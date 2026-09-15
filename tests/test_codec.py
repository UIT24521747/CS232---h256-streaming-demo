"""Tests for codec/: I/P/B classification, prediction, residual,
quantization, entropy coding, and the full encode -> bitstream -> decode
round trip.
"""
from __future__ import annotations

import numpy as np

from codec import bitstream, entropy, mv_index, quantizer
from codec.frame import Frame, FrameType
from codec.h256_decoder import decode_segment
from codec.h256_encoder import BLOCK_SIZE, H256Encoder, classify_frames, encode_order
from tools.inspect_bitstream import parse_segment


def _synthetic_clip(num_frames: int = 8, h: int = 48, w: int = 48):
    """A small clip with a moving block, so motion search has something to find."""
    frames = []
    for t in range(num_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 0] = 40
        f[:, :, 1] = (np.arange(w) * 255 // w).astype(np.uint8)[None, :]
        cx = 4 + t * 2
        f[8:20, cx : cx + 8] = (250, 250, 250)
        frames.append(f)
    return frames


def _encode_chunk(frames, qp: int = 8):
    meta = classify_frames(len(frames), start_id=0)
    meta_by_id = {m["id"]: m for m in meta}
    order = encode_order(meta)

    encoder = H256Encoder(qp=qp)
    dpb: dict = {}
    payloads: dict = {}
    traces: dict = {}
    for fid in order:
        m = meta_by_id[fid]
        fr = Frame(frame_id=fid, type=m["type"], data=frames[fid], ref1=m["ref1"], ref2=m["ref2"])
        payload, trace = encoder.encode_frame(fr, dpb)
        payloads[fid] = payload
        traces[fid] = trace

    segment = bitstream.pack_segment_count(len(order)) + b"".join(payloads[fid] for fid in order)
    return segment, traces, dpb, meta


def test_I_frame():
    frames = _synthetic_clip()
    _segment, traces, _dpb, meta = _encode_chunk(frames)

    assert meta[0]["type"] is FrameType.I
    assert traces[0].ref1 is None and traces[0].ref2 is None
    # DC intra prediction adapts per block from causal neighbors -- it should
    # not degenerate into one flat value across the whole frame.
    assert traces[0].prediction.std() > 0


def test_P_frame():
    frames = _synthetic_clip()
    _segment, traces, _dpb, meta = _encode_chunk(frames)

    p_ids = [m["id"] for m in meta if m["type"] is FrameType.P]
    assert p_ids, "the fixed GOP pattern should always produce P frames"
    fid = p_ids[0]

    assert traces[fid].ref1 is not None
    assert traces[fid].ref2 is None
    mvs = [v[:2] for v in traces[fid].motion_vectors.values()]
    assert any(mv != (0, 0) for mv in mvs), "the moving block should pull in a non-zero motion vector"


def test_B_frame():
    frames = _synthetic_clip()
    _segment, traces, _dpb, meta = _encode_chunk(frames)

    b_ids = [m["id"] for m in meta if m["type"] is FrameType.B]
    assert b_ids, "the fixed GOP pattern should always produce B frames"
    fid = b_ids[0]

    assert traces[fid].ref1 is not None and traces[fid].ref2 is not None
    assert "prev" in traces[fid].motion_vectors and "next" in traces[fid].motion_vectors


def test_encode_order():
    meta = classify_frames(8, start_id=0)
    assert [m["type"].value for m in meta] == ["I", "P", "B", "P", "B", "P", "B", "P"]
    assert encode_order(meta) == [0, 1, 3, 5, 7, 2, 4, 6]


def test_encode_decode():
    frames = _synthetic_clip()
    segment, _traces, dpb, _meta = _encode_chunk(frames)

    decoded_frames, _decode_traces, _dpb2 = decode_segment(segment)
    assert len(decoded_frames) == len(frames)

    for frame in decoded_frames:
        # decoder must reproduce exactly what the encoder's own closed-loop
        # DPB already reconstructed -- that's the whole point of predicting
        # from reconstructed (not original) reference frames.
        assert np.array_equal(frame.data, dpb[frame.frame_id])

        original = frames[frame.frame_id]
        mse = float(np.mean((frame.data.astype(np.int32) - original.astype(np.int32)) ** 2))
        assert mse < 500, "reconstruction should be lossy but nowhere near garbage"


def test_bitstream_integrity():
    frames = _synthetic_clip()
    segment, traces, _dpb, _meta = _encode_chunk(frames)

    n, offset = bitstream.unpack_segment_count(segment)
    assert n == len(frames)

    for _ in range(n):
        header, offset = bitstream.unpack_header(segment, offset)
        assert header["frame_id"] in traces
        for _b in range(header["num_blocks"]):
            (_dx1, _dy1, _dx2, _dy2), offset = bitstream.unpack_block_header(segment, offset)
            for _ch in range(3):
                _pairs, offset = entropy.unpack_pairs(segment, offset)

    assert offset == len(segment), "walking every frame/block/channel should consume the whole segment"


def test_quant_error_bound():
    rng = np.random.default_rng(0)
    residual = rng.integers(-300, 301, size=(64, 64, 3)).astype(np.int16)

    for qp in (1, 2, 8, 16, 32):
        levels = quantizer.quantize(residual, qp)
        dequant = quantizer.dequantize(levels, qp)
        err = np.abs(residual.astype(np.int32) - dequant.astype(np.int32))
        # values clipped by RESIDUAL_CLIP saturate and can exceed qp/2 -- only
        # unclipped levels are guaranteed the round-to-nearest error bound
        unclipped = np.abs(levels) < quantizer.RESIDUAL_CLIP
        assert np.all(err[unclipped] <= qp / 2 + 1e-9)


def test_mv_index():
    frames = _synthetic_clip()
    segment, traces, _dpb, meta = _encode_chunk(frames)

    frames_index = {}
    for m in meta:
        frames_index[m["id"]] = mv_index.frame_entries(traces[m["id"]], BLOCK_SIZE)

    # MVs read straight back out of the bitstream must match the index exactly.
    for header, blocks in parse_segment(segment):
        fid = header["frame_id"]
        entries = {e["block"]: e for e in frames_index[fid]}
        for b in blocks:
            entry = entries[b["index"]]
            if header["type_code"] == 0:  # I-frame: no MV concept
                assert entry["mv"] == []
                continue
            assert entry["mv"][0] == list(b["mv1"])
            if header["ref2"] is not None:
                assert entry["mv"][1] == list(b["mv2"])
