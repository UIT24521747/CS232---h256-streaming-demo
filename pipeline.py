"""Pipeline orchestration: the glue between codec/, streaming/, and sr/.

demo.py (CLI) and web/app.py (dashboard) both call into this module so
there is exactly one implementation of "run the whole thing end to end,
for every user." Nothing in here is a stage's real logic -- it only
sequences calls into the modules that are: codec.h256_encoder/h256_decoder
for compression, codec.mv_index for the per-block MV index, streaming.*
for the network hop and per-user session state, sr.simple_sr for the
upscale.

Everything is scoped by `video_id` (a short content hash, see
`video_id_of`) so multiple videos -- and a re-upload of the same bytes --
can share one `outputs/` tree without colliding, and so an already-encoded
video can be reused instead of re-run through the expensive exhaustive
motion search (see `ensure_encoded`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from werkzeug.serving import make_server

from codec import bitstream, mv_index
from codec.frame import Frame, FrameType
from codec.h256_decoder import decode_segment
from codec.h256_encoder import BLOCK_SIZE, DEFAULT_QP, H256Encoder, classify_frames, encode_order
from metrics.metrics import mse, psnr, ssim
from sr.simple_sr import resize_bilinear, simple_super_resolution
from streaming.client import StreamingClient
from streaming.load_balancer import create_app as create_lb_app
from streaming.server import create_app as create_node_app
from streaming.session import UserSession

CHUNK_SIZE = 8  # frames per segment / independent mini-GOP

# name -> (width, height). "540p" is this project's native resolution;
# the others are illustrative ABR-ladder labels, not real broadcast sizes.
QUALITY_TIERS: Dict[str, Tuple[int, int]] = {
    "240p": (80, 48),
    "360p": (160, 96),
    "540p": (256, 144),
}

NUM_STREAMING_NODES = 2
DEFAULT_BANDWIDTH_KBPS = 1800.0
DEFAULT_MAX_FRAMES = 64  # see README section 3: search cost grows linearly with frame count
SR_SCALE = 2

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "data" / "sample.mp4"
UPLOADS_DIR = ROOT / "data" / "uploads"
OUTPUTS_DIR = ROOT / "outputs"

Logger = Callable[[str], None]

# The per-request access log ("GET /segment/... 200 -") just repeats what
# streaming/client.py and streaming/load_balancer.py already print in a
# more explanatory form -- quiet it so the demo's own trace lines aren't
# drowned out.
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _noop(_msg: str) -> None:
    pass


# ------------------------------------------------------------- video_id --

def video_id_of(path: Path) -> str:
    """Short content hash used to key everything under outputs/<video_id>/,
    so uploading the same bytes twice reuses the same encode (README
    section 3)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def video_output_dir(video_id: str) -> Path:
    return OUTPUTS_DIR / video_id


def segments_dir_for(video_id: str) -> Path:
    return video_output_dir(video_id) / "segments"


def mv_index_dir_for(video_id: str) -> Path:
    return video_output_dir(video_id) / "mv_index"


def users_dir_for(video_id: str) -> Path:
    return video_output_dir(video_id) / "users"


def user_dir_for(video_id: str, user_id: str) -> Path:
    return users_dir_for(video_id) / user_id


def manifest_path_for(video_id: str) -> Path:
    return video_output_dir(video_id) / "manifest.json"


# ---------------------------------------------------------------- source --

def read_video_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read any frames from {path}")
    return frames


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else 12.0


def load_source_video(path: Path, max_frames: int = DEFAULT_MAX_FRAMES) -> List[np.ndarray]:
    """read_video_frames + the --max-frames cap (README section 3): motion
    search is exhaustive and CPU-bound, so a long clip is truncated rather
    than silently taking minutes to encode."""
    return read_video_frames(path)[:max_frames]


def build_quality_tiers(frames: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """Resize the source down to each rung of the ABR ladder.

    Uses the same hand-written bilinear resize as sr/simple_sr.py's
    upscale, just run in the other direction -- one primitive, both
    directions, same as a real encoder's downsampler and a real
    player's upsampler often share code.
    """
    tiers: Dict[str, List[np.ndarray]] = {}
    for name, (w, h) in QUALITY_TIERS.items():
        tiers[name] = [resize_bilinear(f, h, w) for f in frames]
    return tiers


# ---------------------------------------------------------------- encode --

@dataclass
class EncodedFrameInfo:
    frame_id: int
    quality: str
    frame_type: str
    ref1: Optional[int]
    ref2: Optional[int]
    raw_size: int
    encoded_size: int
    segment_name: str


def encode_all(
    tiers: Dict[str, List[np.ndarray]], segments_dir: Path, qp: int = DEFAULT_QP, log: Logger = _noop
):
    """Encode every quality tier into segment_*.bin files on disk.

    Each segment is its own independent mini-GOP (fresh I-frame, fresh
    DPB) -- real segmented streaming (HLS/DASH) does the same thing so
    a client can start decoding a segment without needing any other one.

    Returns (infos, traces): a flat per-frame size/type ledger for the
    summary table, and a {(quality, frame_id): EncodeTrace} dict the
    single-frame trace and the MV index builder both pull from.
    """
    if segments_dir.exists():
        shutil.rmtree(segments_dir)

    infos: List[EncodedFrameInfo] = []
    traces: Dict[Tuple[str, int], object] = {}

    for quality, frame_list in tiers.items():
        qdir = segments_dir / quality
        qdir.mkdir(parents=True, exist_ok=True)
        num_chunks = (len(frame_list) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            chunk = frame_list[start : start + CHUNK_SIZE]
            meta = classify_frames(len(chunk), start_id=start)
            meta_by_id = {m["id"]: m for m in meta}
            order = encode_order(meta)

            encoder = H256Encoder(qp=qp)
            dpb: Dict[int, np.ndarray] = {}
            payloads: Dict[int, bytes] = {}

            for fid in order:
                m = meta_by_id[fid]
                local = fid - start
                fr = Frame(frame_id=fid, type=m["type"], data=chunk[local], ref1=m["ref1"], ref2=m["ref2"])
                payload, trace = encoder.encode_frame(fr, dpb)
                payloads[fid] = payload
                traces[(quality, fid)] = trace
                infos.append(
                    EncodedFrameInfo(
                        frame_id=fid,
                        quality=quality,
                        frame_type=m["type"].value,
                        ref1=m["ref1"],
                        ref2=m["ref2"],
                        raw_size=trace.raw_size,
                        encoded_size=trace.encoded_size,
                        segment_name=f"segment_{chunk_idx:03d}.bin",
                    )
                )

            segment_bytes = bitstream.pack_segment_count(len(order)) + b"".join(payloads[fid] for fid in order)
            (qdir / f"segment_{chunk_idx:03d}.bin").write_bytes(segment_bytes)
            log(f"[Encoder] {quality}/segment_{chunk_idx:03d}.bin: {len(order)} frames, {len(segment_bytes)} bytes")

    return infos, traces


# ------------------------------------------------------- encode caching --

@dataclass
class FrameStats:
    """Lightweight, decode-derived stand-in for EncodeTrace, used on an
    encode-cache hit (see `ensure_encoded`). Carries exactly the fields
    `demo.py:print_frame_table` and `trace_frame` read, so both are
    polymorphic over "freshly encoded" vs. "reused from outputs/<video_id>/".
    `residual` here is reconstructed-minus-prediction (i.e. the
    *dequantized* residual) rather than the true pre-quantization one --
    the difference is at most qp/2 per pixel and decoding needs no motion
    search, which is the whole point of a cache hit.
    """

    frame_id: int
    frame_type: FrameType
    ref1: Optional[int]
    ref2: Optional[int]
    raw_size: int
    encoded_size: int
    residual: np.ndarray
    prediction: np.ndarray


def _rebuild_stats_for_quality(video_id: str, quality: str, log: Logger) -> Dict[int, FrameStats]:
    qdir = segments_dir_for(video_id) / quality
    stats: Dict[int, FrameStats] = {}
    for seg_path in sorted(qdir.glob("segment_*.bin")):
        data = seg_path.read_bytes()
        frames, traces, _dpb = decode_segment(data)
        for frame, trace in zip(frames, traces):
            prediction = trace.prediction
            residual = trace.reconstructed.astype(np.int16) - prediction.astype(np.int16)
            stats[frame.frame_id] = FrameStats(
                frame_id=frame.frame_id,
                frame_type=trace.frame_type,
                ref1=trace.header["ref1"],
                ref2=trace.header["ref2"],
                raw_size=trace.header["width"] * trace.header["height"] * 3,
                encoded_size=trace.encoded_size,
                residual=residual,
                prediction=prediction,
            )
    log(f"[Encoder] cache hit for {quality}: reused outputs/{video_id}/segments/{quality}")
    return stats


def _write_mv_index(video_id: str, traces: Dict[Tuple[str, int], object], log: Logger) -> None:
    by_quality: Dict[str, Dict[int, list]] = {}
    for (quality, frame_id), trace in traces.items():
        by_quality.setdefault(quality, {})[frame_id] = mv_index.frame_entries(trace, BLOCK_SIZE)
    for quality, frames in by_quality.items():
        path = mv_index_dir_for(video_id) / f"{quality}.json"
        mv_index.write_index(path, frames)
        log(f"[Encoder] wrote mv_index/{quality}.json ({len(frames)} frames)")


def ensure_encoded(
    video_id: str, tiers: Dict[str, List[np.ndarray]], qp: int, log: Logger = _noop
) -> Tuple[List[EncodedFrameInfo], Dict[Tuple[str, int], object]]:
    """Encode-or-reuse: if outputs/<video_id>/ already holds a matching
    (qp, num_frames) encode for every quality tier, skip the expensive
    exhaustive-search encode and rebuild per-frame stats by decoding the
    cached segments instead (README section 3)."""
    num_frames = len(next(iter(tiers.values())))
    seg_dir = segments_dir_for(video_id)
    manifest_path = manifest_path_for(video_id)

    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = None

    cache_hit = bool(
        manifest
        and manifest.get("qp") == qp
        and manifest.get("num_frames") == num_frames
        and all((seg_dir / q).exists() for q in tiers)
    )

    if cache_hit:
        log(f"[Encoder] cache hit: outputs/{video_id}/ already has qp={qp} num_frames={num_frames}")
        infos: List[EncodedFrameInfo] = []
        traces: Dict[Tuple[str, int], object] = {}
        for quality in tiers:
            stats = _rebuild_stats_for_quality(video_id, quality, log)
            for frame_id, s in stats.items():
                traces[(quality, frame_id)] = s
                infos.append(
                    EncodedFrameInfo(
                        frame_id=frame_id,
                        quality=quality,
                        frame_type=s.frame_type.value,
                        ref1=s.ref1,
                        ref2=s.ref2,
                        raw_size=s.raw_size,
                        encoded_size=s.encoded_size,
                        segment_name=f"segment_{frame_id // CHUNK_SIZE:03d}.bin",
                    )
                )
        return infos, traces

    infos, traces = encode_all(tiers, seg_dir, qp=qp, log=log)
    _write_mv_index(video_id, traces, log)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"video_id": video_id, "qp": qp, "num_frames": num_frames, "qualities": sorted(tiers)}, indent=2),
        encoding="utf-8",
    )
    return infos, traces


def gop_indices(frame_id: int, num_frames: int) -> Tuple[int, int]:
    """(display_index_in_gop, decode_index_in_gop) for one frame -- the
    same local numbering classify_frames/encode_order use, exposed for the
    per-frame history record (README section 7.1)."""
    chunk_start = (frame_id // CHUNK_SIZE) * CHUNK_SIZE
    chunk_len = min(CHUNK_SIZE, num_frames - chunk_start)
    local_id = frame_id - chunk_start
    meta = classify_frames(chunk_len, start_id=chunk_start)
    order = encode_order(meta)
    return local_id, order.index(frame_id)


# -------------------------------------------------------------- streaming --

@dataclass
class StreamingHandles:
    servers: list
    threads: list
    lb_url: str
    node_urls: list

    def shutdown(self) -> None:
        for s in self.servers:
            s.shutdown()
        for t in self.threads:
            t.join(timeout=2)


def start_streaming(outputs_root: Optional[Path] = None, log: Logger = _noop) -> StreamingHandles:
    """Start two streaming nodes plus the load balancer, each as a real
    Flask app bound to its own port in a background thread. The rest of
    the pipeline then talks to them over genuine HTTP, exactly like
    `streaming/server.py` and `streaming/load_balancer.py` do when run
    standalone. Nodes are bound to the whole `outputs/` root (not one
    video's segments dir) so they can serve any video_id already encoded
    there without needing a restart.

    Ports are OS-assigned (bind to port 0), not fixed -- these are purely
    internal to one pipeline run (nothing outside this process ever needs
    to guess them), and a fixed port silently breaks the *next* run if
    anything ever stops this one uncleanly (e.g. the web dashboard's
    process being killed mid-run): the next `start_streaming()` would fail
    to bind with no visible error beyond a generic status/log line, and
    every user's player would sit in BUFFERING forever with no clue why.
    """
    outputs_root = Path(outputs_root) if outputs_root is not None else OUTPUTS_DIR
    servers, threads, node_urls = [], [], []
    try:
        for i in range(1, NUM_STREAMING_NODES + 1):
            name = f"Node-{i}"
            srv = make_server("127.0.0.1", 0, create_node_app(name, outputs_root))
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            servers.append(srv)
            threads.append(th)
            url = f"http://127.0.0.1:{srv.server_port}"
            node_urls.append(url)
            log(f"[Streaming] started {name} on {url}")

        lb_srv = make_server("127.0.0.1", 0, create_lb_app(node_urls))
        lb_th = threading.Thread(target=lb_srv.serve_forever, daemon=True)
        lb_th.start()
        servers.append(lb_srv)
        threads.append(lb_th)
        lb_url = f"http://127.0.0.1:{lb_srv.server_port}"
        log(f"[Streaming] started LoadBalancer on {lb_url}")
    except Exception:
        # a later bind failing (rare, but e.g. OS port exhaustion) must not
        # orphan servers that already started -- they're daemon threads
        # and would otherwise keep running forever, unreachable.
        for s in servers:
            s.shutdown()
        raise

    time.sleep(0.2)  # give the sockets a moment to accept before the client hits them
    return StreamingHandles(servers=servers, threads=threads, lb_url=lb_url, node_urls=node_urls)


@dataclass
class ClientFrameResult:
    frame_id: int
    quality: str
    node: str
    segment_name: str
    frame: Frame
    trace: object
    lb_counter: int = -1


def run_client_download(
    lb_url: str,
    video_id: str,
    user_id: str,
    total_frames: int,
    bandwidth_kbps: Union[float, Callable[[], float]],
    log: Logger = print,
    on_segment: Optional[Callable[[dict], None]] = None,
    on_frame: Optional[Callable[["ClientFrameResult"], None]] = None,
) -> List[ClientFrameResult]:
    """Download every segment through the load balancer, in order, and
    decode it, for one user. `on_segment`/`on_frame` fire as each
    segment/frame actually arrives -- real wall-clock time apart, throttled
    by `bandwidth_kbps` -- so a caller (the web dashboard, run_all_users)
    can react live instead of waiting for the whole download to finish.
    """
    client = StreamingClient(lb_url, video_id, user_id, bandwidth_kbps=bandwidth_kbps, log_fn=log)
    manifest = client.fetch_manifest()
    log(f"[Client] user={user_id} manifest qualities={manifest.get('qualities')}")

    num_chunks = (total_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    results: List[ClientFrameResult] = []
    for chunk_idx in range(num_chunks):
        seg_name = f"segment_{chunk_idx:03d}.bin"
        data, quality, node, sim_time, lb_counter = client.fetch_segment(seg_name)
        if on_segment is not None:
            on_segment(
                {
                    "user_id": user_id,
                    "segment": seg_name,
                    "quality": quality,
                    "node": node,
                    "bytes": len(data),
                    "simulated_seconds": sim_time,
                    "lb_counter": lb_counter,
                }
            )
        frames, traces, _dpb = decode_segment(data)
        for frame, trace in zip(frames, traces):
            result = ClientFrameResult(
                frame_id=frame.frame_id,
                quality=quality,
                node=node,
                segment_name=seg_name,
                frame=frame,
                trace=trace,
                lb_counter=lb_counter,
            )
            results.append(result)
            if on_frame is not None:
                on_frame(result)

    results.sort(key=lambda r: r.frame_id)
    return results


def run_all_users(
    lb_url: str,
    video_id: str,
    sessions: List[UserSession],
    num_frames: int,
    fps: float,
    tiers: Dict[str, List[np.ndarray]],
    mv_index_cache: Dict[str, dict],
    log: Logger = print,
    on_segment: Optional[Callable[[dict], None]] = None,
    on_frame: Optional[Callable[[UserSession, "ClientFrameResult"], None]] = None,
) -> None:
    """Download+decode the video once per user, each in its own thread
    (README section 11, step [6/7]) -- one real StreamingClient and
    UserSession per user, all sharing the same load balancer and encoded
    segments. Populates each session's `.history` (README section 7.1) and
    `.results` (kept for SR/trace reuse) as a side effect.

    `on_segment`/`on_frame` are optional taps for a live caller (the web
    dashboard): `on_segment` fires with the same shape as
    `run_client_download`'s, `on_frame` fires with `(session, result)`
    right after that frame's history record has been pushed.

    A per-user download thread that raises (a bad connection, a decode
    error, ...) is NOT allowed to fail silently -- Python's threading
    module would otherwise just drop the traceback to stderr and leave
    that one user's session stuck with no frames and no explanation, while
    everything else (and the overall run) reports success. Every worker
    exception is collected and the first one is re-raised here, after
    every thread has finished, so callers (demo.py, web/app.py) see it the
    same way they'd see any other pipeline failure.
    """
    errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def _download_one_user(session: UserSession) -> None:
        session.start_playback(fps, num_frames)
        run_start = time.perf_counter()

        def handle_frame(result: ClientFrameResult) -> None:
            quality = result.quality
            index = mv_index_cache.get(quality)
            if index is None:
                index_path = mv_index_dir_for(video_id) / f"{quality}.json"
                index = mv_index.load_index(index_path) if index_path.exists() else {}
                mv_index_cache[quality] = index
            entries = mv_index.lookup(index, result.frame_id)

            orig = tiers[quality][result.frame_id]
            recon = result.frame.data[: orig.shape[0], : orig.shape[1]]
            residual = result.trace.reconstructed.astype(np.int16) - result.trace.prediction.astype(np.int16)
            residual = residual[: orig.shape[0], : orig.shape[1]]

            display_idx, decode_idx = gop_indices(result.frame_id, num_frames)
            header = result.trace.header
            raw_bytes = header["width"] * header["height"] * 3
            encoded_bytes = result.trace.encoded_size
            p = psnr(orig, recon)

            record = {
                "user_id": session.user_id,
                "video_id": video_id,
                "frame_id": result.frame_id,
                "segment": result.segment_name,
                "display_index_in_gop": display_idx,
                "decode_index_in_gop": decode_idx,
                "type": result.trace.frame_type.value,
                "refs": [r for r in (header["ref1"], header["ref2"]) if r is not None],
                "quality": quality,
                "resolution": [orig.shape[1], orig.shape[0]],
                "qp": header["qp"],
                "node": result.node,
                "lb_counter": result.lb_counter,
                "bandwidth_kbps": session.bandwidth_kbps,
                "raw_bytes": raw_bytes,
                "encoded_bytes": encoded_bytes,
                "compression_pct": round(100 * (1 - encoded_bytes / raw_bytes), 1) if raw_bytes else 0.0,
                "residual_range": [int(residual.min()), int(residual.max())],
                "residual_mse": round(mse(orig, recon), 2),
                "psnr_db": None if p == float("inf") else round(p, 2),
                "ssim": round(ssim(orig, recon), 4),
                "n_blocks": header["num_blocks"],
                "mv_ref": f"mv_index/{quality}.json#frame={result.frame_id}",
                "recv_time_ms": round((time.perf_counter() - run_start) * 1000),
            }
            session.push(record)
            if on_frame is not None:
                on_frame(session, result)

        results = run_client_download(
            lb_url,
            video_id,
            session.user_id,
            num_frames,
            session.bandwidth_kbps,
            log=log,
            on_segment=on_segment,
            on_frame=handle_frame,
        )
        session.results = results

        history_path = user_dir_for(video_id, session.user_id) / "history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text("\n".join(json.dumps(r) for r in session.history), encoding="utf-8")

    def worker(session: UserSession) -> None:
        try:
            _download_one_user(session)
        except Exception as exc:  # noqa: BLE001 -- collected below, never silently dropped
            log(f"[User {session.user_id}] ERROR: {exc}")
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(session,), daemon=True) for session in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]


# ------------------------------------------------------------------- SR --

def apply_sr(results: List[ClientFrameResult]) -> Dict[int, np.ndarray]:
    return {r.frame_id: simple_super_resolution(r.frame.data, scale=SR_SCALE) for r in results}


# --------------------------------------------------------------- metrics --

def codec_fidelity_table(results: List[ClientFrameResult], tiers: Dict[str, List[np.ndarray]]) -> List[dict]:
    """Reconstructed-vs-original at the SAME tier resolution -- isolates
    how much error the codec itself introduced, independent of SR."""
    rows = []
    for r in results:
        orig = tiers[r.quality][r.frame_id]
        recon = r.frame.data[: orig.shape[0], : orig.shape[1]]
        rows.append(
            {
                "frame_id": r.frame_id,
                "quality": r.quality,
                "type": r.trace.frame_type.value,
                "mse": mse(orig, recon),
                "psnr": psnr(orig, recon),
                "ssim": ssim(orig, recon),
            }
        )
    return rows


# ----------------------------------------------------------------- trace --

def trace_frame(
    frame_id: int,
    native_frames: List[np.ndarray],
    tiers: Dict[str, List[np.ndarray]],
    encode_traces: Dict[Tuple[str, int], object],
    results: List[ClientFrameResult],
    sr_frames: Dict[int, np.ndarray],
    qp: int,
    *,
    mv_entries: Optional[list] = None,
    user_id: Optional[str] = None,
    video_id: Optional[str] = None,
) -> dict:
    """Assemble every number needed for the 13-section end-to-end trace
    (see demo.py) plus the images tools/visualize_trace.py composites.
    `encode_traces` values may be a real EncodeTrace (fresh encode) or a
    FrameStats (encode-cache hit) -- both expose the same fields used here.
    """
    result = next(r for r in results if r.frame_id == frame_id)
    enc_trace = encode_traces[(result.quality, frame_id)]

    orig_native = native_frames[frame_id]
    orig_tier = tiers[result.quality][frame_id]
    recon = result.frame.data[: orig_tier.shape[0], : orig_tier.shape[1]]
    sr_out = sr_frames[frame_id]

    residual = enc_trace.residual
    frame_type = enc_trace.frame_type

    if frame_type is FrameType.I:
        prediction_desc = "DC intra prediction (left/top neighbor blocks, this frame only)"
    elif frame_type is FrameType.P:
        prediction_desc = f"motion-compensated copy of frame {enc_trace.ref1}"
    else:
        prediction_desc = f"average of motion-compensated frames {enc_trace.ref1} and {enc_trace.ref2}"

    return {
        "frame_id": frame_id,
        "user_id": user_id,
        "video_id": video_id,
        "quality": result.quality,
        "source": {
            "resolution": f"{orig_native.shape[1]}x{orig_native.shape[0]}",
            "original_size_bytes": int(orig_native.nbytes),
        },
        "frame_type": {"type": frame_type.value, "ref1": enc_trace.ref1, "ref2": enc_trace.ref2},
        "prediction": {"description": prediction_desc},
        "residual": {
            "min": int(residual.min()),
            "max": int(residual.max()),
            "mse": float(np.mean(residual.astype(np.float64) ** 2)),
        },
        "quantization": {"qp": qp},
        "h256_encode": {
            "raw_size": enc_trace.raw_size,
            "encoded_size": enc_trace.encoded_size,
            "compression_pct": 100.0 * (1 - enc_trace.encoded_size / enc_trace.raw_size),
        },
        "bitstream": {"segment": result.segment_name, "quality": result.quality},
        "load_balancer": {
            "request": f"{result.quality}/{result.segment_name}",
            "selected_node": result.node,
            "policy": "Round-Robin",
            "lb_counter": result.lb_counter,
        },
        "client": {"received_bytes": enc_trace.encoded_size},
        "h256_decode": {
            "frame_type": result.trace.frame_type.value,
            "ref1": result.trace.header["ref1"],
            "ref2": result.trace.header["ref2"],
        },
        "reconstruction": {
            "resolution": f"{orig_tier.shape[1]}x{orig_tier.shape[0]}",
            "mse": mse(orig_tier, recon),
            "psnr": psnr(orig_tier, recon),
            "ssim": ssim(orig_tier, recon),
        },
        "super_resolution": {
            "input": f"{recon.shape[1]}x{recon.shape[0]}",
            "output": f"{sr_out.shape[1]}x{sr_out.shape[0]}",
            "method": "Bilinear x2 + hand-written sharpen kernel",
        },
        "mv": mv_entries or [],
        "images": {
            "original": orig_tier,
            "prediction": enc_trace.prediction,
            "residual": residual,
            "reconstructed": recon,
            "sr": sr_out,
        },
    }


def trace_user_frame(
    video_id: str,
    session: UserSession,
    frame_id: int,
    native_frames: List[np.ndarray],
    tiers: Dict[str, List[np.ndarray]],
    encode_traces: Dict[Tuple[str, int], object],
    qp: int,
    mv_index_cache: Dict[str, dict],
) -> dict:
    """trace_frame, scoped to one user's actually-received quality for this
    frame -- different users can receive different qualities for the same
    frame_id (README section 6.2)."""
    result = next(r for r in session.results if r.frame_id == frame_id)
    sr_img = simple_super_resolution(result.frame.data, scale=SR_SCALE)

    quality = result.quality
    index = mv_index_cache.get(quality)
    if index is None:
        index_path = mv_index_dir_for(video_id) / f"{quality}.json"
        index = mv_index.load_index(index_path) if index_path.exists() else {}
        mv_index_cache[quality] = index
    entries = mv_index.lookup(index, frame_id)

    return trace_frame(
        frame_id,
        native_frames,
        tiers,
        encode_traces,
        [result],
        {frame_id: sr_img},
        qp,
        mv_entries=entries,
        user_id=session.user_id,
        video_id=video_id,
    )
