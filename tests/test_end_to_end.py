"""Full pipeline smoke test: encode -> segments -> streaming nodes ->
round-robin load balancer -> N independent users -> decode -> simple SR,
wired together exactly the way demo.py and the web dashboard drive it.
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


def test_end_to_end(tmp_path, monkeypatch):
    # keep this test off the module's default output root so it can run
    # alongside (or after) a manually-started demo.py/web dashboard --
    # streaming ports are OS-assigned (see pipeline.start_streaming), so
    # there's nothing to isolate there.
    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path)

    video_id = "cafef00d"
    frames = _tiny_clip()
    tiers = pipeline.build_quality_tiers(frames)

    infos, encode_traces = pipeline.ensure_encoded(video_id, tiers, qp=8, log=lambda m: None)
    assert infos
    assert (pipeline.segments_dir_for(video_id) / "540p" / "segment_000.bin").exists()
    assert (pipeline.mv_index_dir_for(video_id) / "540p.json").exists()

    handles = pipeline.start_streaming(tmp_path)
    try:
        sessions = SessionManager()
        users = [sessions.create_user(2000) for _ in range(2)]
        mv_cache: dict = {}
        pipeline.run_all_users(
            handles.lb_url, video_id, users, len(frames), fps=12.0, tiers=tiers, mv_index_cache=mv_cache, log=lambda m: None
        )

        for session in users:
            assert len(session.results) == len(frames)
            assert {r.frame_id for r in session.results} == set(range(len(frames)))
            assert len(session.history) == len(frames)
            assert [h["recv_order"] for h in session.history] == list(range(len(frames)))

            sr_frames = pipeline.apply_sr(session.results)
            assert len(sr_frames) == len(frames)
            for r in session.results:
                sr = sr_frames[r.frame_id]
                recon = r.frame.data
                assert sr.shape[0] == recon.shape[0] * pipeline.SR_SCALE
                assert sr.shape[1] == recon.shape[1] * pipeline.SR_SCALE

            fidelity = pipeline.codec_fidelity_table(session.results, tiers)
            assert all(row["psnr"] == float("inf") or row["psnr"] > 15 for row in fidelity)

        session = users[0]
        trace = pipeline.trace_user_frame(
            video_id, session, 0, frames, tiers, encode_traces, qp=8, mv_index_cache=mv_cache
        )
        assert trace["frame_type"]["type"] == "I"
        assert trace["h256_encode"]["encoded_size"] > 0
        assert trace["images"]["sr"].shape[0] == trace["images"]["reconstructed"].shape[0] * pipeline.SR_SCALE
        assert trace["user_id"] == session.user_id
    finally:
        handles.shutdown()
