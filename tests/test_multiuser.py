"""Tests for the multi-user layer: streaming/session.py's SessionManager,
and the per-user MV lookup pipeline.run_all_users wires up against
codec/mv_index.py (README section 6.2 -- MV is a property of the encoded
content, looked up through *which quality a user's session picked*, not
computed per user).
"""
from __future__ import annotations

import numpy as np

import pipeline
from streaming.session import SessionManager


def _tiny_clip(num_frames: int = 8, size: int = 48):
    frames = []
    for t in range(num_frames):
        f = np.zeros((size, size, 3), dtype=np.uint8)
        f[:, :, 1] = (np.arange(size) * 255 // size).astype(np.uint8)[None, :]
        cx = 4 + (t * 3) % (size - 12)
        f[10:20, cx : cx + 8] = (255, 255, 255)
        frames.append(f)
    return frames


def test_sessions():
    sessions = SessionManager()
    u1 = sessions.create_user(bandwidth_kbps=800)
    u2 = sessions.create_user(bandwidth_kbps=3000)

    assert u1.user_id != u2.user_id
    assert sessions.get(u1.user_id) is u1
    assert len(sessions.all()) == 2

    # changing one user's bandwidth must not affect the other
    sessions.set_bandwidth(u1.user_id, 200)
    assert sessions.get(u1.user_id).bandwidth_kbps == 200
    assert sessions.get(u2.user_id).bandwidth_kbps == 3000

    assert sessions.remove_user(u1.user_id) is True
    assert sessions.get(u1.user_id) is None
    assert len(sessions.all()) == 1


def test_user_mv_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path)

    video_id = "f00dcafe"
    frames = _tiny_clip()
    tiers = pipeline.build_quality_tiers(frames)
    pipeline.ensure_encoded(video_id, tiers, qp=8, log=lambda m: None)

    handles = pipeline.start_streaming(tmp_path)
    try:
        sessions = SessionManager()
        # force two users onto two different quality tiers via bandwidth
        low = sessions.create_user(bandwidth_kbps=200)  # -> 240p
        high = sessions.create_user(bandwidth_kbps=5000)  # -> 540p
        same_as_high = sessions.create_user(bandwidth_kbps=6000)  # -> 540p too

        mv_cache: dict = {}
        pipeline.run_all_users(
            handles.lb_url, video_id, [low, high, same_as_high], len(frames),
            fps=12.0, tiers=tiers, mv_index_cache=mv_cache, log=lambda m: None,
        )

        assert low.quality == "240p"
        assert high.quality == "540p"
        assert same_as_high.quality == "540p"

        # same quality -> identical MV lookup for the same frame
        rec_high = next(h for h in high.history if h["frame_id"] == 3)
        rec_same = next(h for h in same_as_high.history if h["frame_id"] == 3)
        assert rec_high["mv_ref"] == rec_same["mv_ref"]

        high_index = mv_cache["540p"]
        same_index = mv_cache["540p"]
        assert high_index is same_index  # same quality shares the same loaded index

        # different quality -> a different index source (and generally different MVs)
        low_index = mv_cache["240p"]
        assert low_index is not high_index
    finally:
        handles.shutdown()
