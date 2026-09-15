"""Browser dashboard for the H256 demo -- multi-user edition.

A single Flask app, a single HTML page, plain JS polling two JSON status
endpoints every half second -- no build step, no frontend framework. Each
user in the grid is backed by a real `streaming.session.UserSession`
downloading through its own thread (see `pipeline.run_all_users`); dragging
a user's bandwidth slider mid-run mutates that live session's
`bandwidth_kbps`, so the next segment *that user* requests can pick a
different ABR tier while every other user is unaffected -- same "live
slider" trick the original single-user dashboard used, just per user now.

Every user's "Received + SR" cell is a real video player: it only advances
once that user's frame has actually finished download + decode + SR on the
backend (see `_on_frame` below), so a low-bandwidth user visibly stalls
independently of the rest of the grid -- genuine per-user rebuffering, not
a fake progress bar.

Run with:  python demo.py --web
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory

import pipeline
from codec.h256_encoder import DEFAULT_QP
from data.generate_sample import generate as generate_sample
from sr.simple_sr import simple_super_resolution
from streaming.session import SessionManager
from tools.visualize_trace import save_frame_outputs

WEB_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(WEB_DIR / "templates"), static_folder=str(WEB_DIR / "static"))

# How fast each user's "Received + SR" pane advances through its ready
# frames. 30fps reads as actual video instead of a slideshow; a low
# bandwidth user still visibly stalls (BUFFERING) whenever their next
# frame genuinely isn't ready yet -- that part doesn't depend on this
# number, only on the real throttled download time (streaming/client.py).
PLAYBACK_FPS = 30


@dataclass
class AppState:
    video_id: Optional[str] = None
    video_path: Optional[Path] = None
    resolution: str = ""
    num_frames: int = 0
    native_fps: float = 12.0
    status: str = "idle"  # idle | running | done | error
    stage: str = ""
    log: List[str] = field(default_factory=list)
    qp: int = DEFAULT_QP
    lb_log: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    run_seconds: Optional[float] = None
    run_id: int = 0
    sessions: SessionManager = field(default_factory=SessionManager)
    tiers: Dict[str, list] = field(default_factory=dict)
    native_frames: list = field(default_factory=list)
    encode_traces: dict = field(default_factory=dict)
    mv_index_cache: dict = field(default_factory=dict)
    frames_ready: Dict[str, List[int]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log_line(self, msg: str) -> None:
        with self.lock:
            self.log.append(msg)
            self.log = self.log[-300:]

    def add_lb_entry(self, entry: dict) -> None:
        with self.lock:
            self.lb_log.append(entry)
            self.lb_log = self.lb_log[-200:]

    def mark_ready(self, user_id: str, frame_id: int) -> None:
        with self.lock:
            self.frames_ready.setdefault(user_id, []).append(frame_id)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "video_id": self.video_id,
                "resolution": self.resolution,
                "num_frames": self.num_frames,
                "native_fps": self.native_fps,
                "status": self.status,
                "stage": self.stage,
                "log": list(self.log[-80:]),
                "qp": self.qp,
                "lb_log": list(self.lb_log[-40:]),
                "error": self.error,
                "run_seconds": self.run_seconds,
                "run_id": self.run_id,
                "quality_tiers": pipeline.QUALITY_TIERS,
                "playback_fps": PLAYBACK_FPS,
            }

    def users_snapshot(self) -> list:
        with self.lock:
            ready = dict(self.frames_ready)
        rows = []
        for session in self.sessions.all():
            row = session.snapshot()
            row["frames_ready"] = ready.get(session.user_id, [])
            rows.append(row)
        return rows


STATE = AppState()


def _load_video(path: Path) -> dict:
    """Point the dashboard at one video: hash it, read (+ truncate) its
    frames, and build the ABR ladder."""
    video_id = pipeline.video_id_of(path)
    frames = pipeline.load_source_video(path, pipeline.DEFAULT_MAX_FRAMES)
    fps = pipeline.video_fps(path)
    h, w = frames[0].shape[:2]
    tiers = pipeline.build_quality_tiers(frames)

    with STATE.lock:
        STATE.video_id = video_id
        STATE.video_path = path
        STATE.resolution = f"{w}x{h}"
        STATE.num_frames = len(frames)
        STATE.native_fps = fps
        STATE.native_frames = frames
        STATE.tiers = tiers
        STATE.encode_traces = {}
        STATE.mv_index_cache = {}
        STATE.frames_ready = {}
        STATE.stage = "video loaded"

    return {"video_id": video_id, "resolution": STATE.resolution, "num_frames": STATE.num_frames, "fps": fps}


def _run_pipeline(qp: int) -> None:
    t0 = time.perf_counter()
    with STATE.lock:
        STATE.status = "running"
        STATE.stage = "starting"
        STATE.log = []
        STATE.lb_log = []
        STATE.error = None
        STATE.run_seconds = None
        STATE.qp = qp
        STATE.run_id = int(time.time() * 1000)
        STATE.frames_ready = {}
        video_id = STATE.video_id
        tiers = STATE.tiers
        native_fps = STATE.native_fps
        num_frames = STATE.num_frames
    for session in STATE.sessions.all():
        session.results = []
        session.history = []

    try:
        STATE.stage = "H256 encoding"
        infos, encode_traces = pipeline.ensure_encoded(video_id, tiers, qp=qp, log=STATE.log_line)
        with STATE.lock:
            STATE.encode_traces = encode_traces
            STATE.mv_index_cache = {}

        STATE.stage = "starting streaming nodes"
        handles = pipeline.start_streaming(log=STATE.log_line)

        try:
            STATE.stage = "client streaming"

            def on_segment(info: dict) -> None:
                STATE.add_lb_entry(
                    {
                        "user": info["user_id"],
                        "request": f"{info['quality']}/{info['segment']}",
                        "selected_node": info["node"],
                        "policy": "Round-Robin",
                        "lb_counter": info["lb_counter"],
                    }
                )

            def on_frame(session, result) -> None:
                # Fires the instant one user's frame is decoded -- i.e. as
                # soon as *that user's* throttled segment download has
                # actually finished. SR runs here too, so a user's
                # "Received + SR" image only appears once the whole chain
                # (download -> decode -> SR) for that exact frame is done.
                sr_img = simple_super_resolution(result.frame.data, scale=pipeline.SR_SCALE)
                live_dir = pipeline.user_dir_for(video_id, session.user_id) / "live"
                live_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(live_dir / f"frame_{result.frame_id}_final.png"), sr_img)
                STATE.mark_ready(session.user_id, result.frame_id)

            pipeline.run_all_users(
                handles.lb_url,
                video_id,
                STATE.sessions.all(),
                num_frames,
                native_fps,
                tiers,
                STATE.mv_index_cache,
                log=STATE.log_line,
                on_segment=on_segment,
                on_frame=on_frame,
            )
        finally:
            handles.shutdown()

        STATE.run_seconds = round(time.perf_counter() - t0, 1)
        STATE.stage = "done"
        STATE.status = "done"
    except Exception as exc:  # noqa: BLE001 -- surface any failure to the dashboard
        STATE.status = "error"
        STATE.error = str(exc)
        STATE.log_line(f"[Error] {exc}")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def api_upload():
    file = request.files.get("video")
    if file is None or file.filename == "":
        return jsonify({"error": "no file uploaded"}), 400

    pipeline.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = pipeline.UPLOADS_DIR / f"_upload_{int(time.time() * 1000)}.mp4"
    file.save(tmp_path)

    video_id = pipeline.video_id_of(tmp_path)
    final_path = pipeline.UPLOADS_DIR / f"{video_id}.mp4"
    if final_path.exists():
        tmp_path.unlink(missing_ok=True)  # already uploaded before -- reuse it (and its encode cache)
    else:
        tmp_path.replace(final_path)

    info = _load_video(final_path)
    return jsonify(info)


@app.post("/api/users")
def api_create_user():
    body = request.get_json(force=True, silent=True) or {}
    bandwidth = float(body.get("bandwidth", pipeline.DEFAULT_BANDWIDTH_KBPS))
    session = STATE.sessions.create_user(bandwidth)
    return jsonify({"user_id": session.user_id, "bandwidth_kbps": session.bandwidth_kbps})


@app.delete("/api/users/<user_id>")
def api_delete_user(user_id: str):
    if not STATE.sessions.remove_user(user_id):
        return jsonify({"error": "unknown user"}), 404
    with STATE.lock:
        STATE.frames_ready.pop(user_id, None)
    return jsonify({"ok": True})


@app.post("/api/users/<user_id>/bandwidth")
def api_user_bandwidth(user_id: str):
    body = request.get_json(force=True, silent=True) or {}
    bandwidth = float(body.get("bandwidth", 0))
    if not STATE.sessions.set_bandwidth(user_id, bandwidth):
        return jsonify({"error": "unknown user"}), 404
    return jsonify({"ok": True, "bandwidth_kbps": bandwidth})


@app.get("/api/users")
def api_users():
    return jsonify(STATE.users_snapshot())


@app.get("/api/users/<user_id>/frames")
def api_user_frames(user_id: str):
    session = STATE.sessions.get(user_id)
    if session is None:
        return jsonify({"error": "unknown user"}), 404
    return jsonify(session.history)


@app.get("/api/users/<user_id>/frames/<int:frame_id>")
def api_user_frame_detail(user_id: str, frame_id: int):
    session = STATE.sessions.get(user_id)
    if session is None:
        return jsonify({"error": "unknown user"}), 404
    if STATE.video_id is None or not session.results:
        return jsonify({"error": "no run yet"}), 409

    try:
        trace = pipeline.trace_user_frame(
            STATE.video_id,
            session,
            frame_id,
            STATE.native_frames,
            STATE.tiers,
            STATE.encode_traces,
            STATE.qp,
            STATE.mv_index_cache,
        )
    except StopIteration:
        return jsonify({"error": "this user never received that frame"}), 404

    out_dir = pipeline.user_dir_for(STATE.video_id, user_id) / f"frame_{frame_id}"
    save_frame_outputs(trace, out_dir)

    json_safe = {k: v for k, v in trace.items() if k != "images"}
    json_safe["image_base"] = f"/outputs/{STATE.video_id}/users/{user_id}/frame_{frame_id}"
    return jsonify(json_safe)


@app.post("/api/run")
def api_run():
    if STATE.status == "running":
        return jsonify({"status": "busy"}), 409
    if STATE.video_id is None:
        return jsonify({"error": "no video loaded yet"}), 409
    if not STATE.sessions.all():
        return jsonify({"error": "create at least one user first"}), 409

    body = request.get_json(force=True, silent=True) or {}
    requested_video_id = body.get("video_id")
    if requested_video_id and requested_video_id != STATE.video_id:
        return jsonify({"error": f"video_id mismatch: loaded={STATE.video_id} requested={requested_video_id}"}), 409
    qp = int(body.get("qp", STATE.qp))

    thread = threading.Thread(target=_run_pipeline, args=(qp,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.get("/api/state")
def api_state():
    return jsonify(STATE.snapshot())


@app.get("/outputs/<path:subpath>")
def outputs(subpath: str):
    return send_from_directory(pipeline.OUTPUTS_DIR, subpath)


def run_dashboard(port: int = 7000) -> None:
    if not pipeline.DEFAULT_VIDEO.exists():
        generate_sample(pipeline.DEFAULT_VIDEO)
    _load_video(pipeline.DEFAULT_VIDEO)
    print(f"H256 dashboard: http://127.0.0.1:{port}")
    app.run(port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_dashboard()
