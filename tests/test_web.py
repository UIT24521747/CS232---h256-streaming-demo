"""Test for web/app.py's upload endpoint: a video's video_id is stable
across re-uploads of the same bytes, and a second upload reuses the first
one's encode cache instead of re-running it (README section 3/9.3).
"""
from __future__ import annotations

import io

import pipeline
from data.generate_sample import generate as generate_sample
from web.app import app


def test_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path / "outputs")

    clip_path = tmp_path / "clip.mp4"
    generate_sample(clip_path, num_frames=8, fps=12)
    clip_bytes = clip_path.read_bytes()

    client = app.test_client()

    resp1 = client.post("/api/upload", data={"video": (io.BytesIO(clip_bytes), "clip.mp4")})
    assert resp1.status_code == 200
    body1 = resp1.get_json()
    assert body1["video_id"] == pipeline.video_id_of(clip_path)
    assert body1["num_frames"] == 8

    manifest_path = pipeline.manifest_path_for(body1["video_id"])
    assert not manifest_path.exists()  # upload alone doesn't encode yet

    infos, _traces = pipeline.ensure_encoded(body1["video_id"], pipeline.build_quality_tiers(
        pipeline.load_source_video(clip_path, pipeline.DEFAULT_MAX_FRAMES)
    ), qp=8, log=lambda m: None)
    assert infos
    mtime_before = manifest_path.stat().st_mtime

    # re-uploading the exact same bytes must resolve to the same video_id
    # and must not disturb the existing encode/manifest on disk.
    resp2 = client.post("/api/upload", data={"video": (io.BytesIO(clip_bytes), "clip_again.mp4")})
    assert resp2.status_code == 200
    body2 = resp2.get_json()
    assert body2["video_id"] == body1["video_id"]
    assert manifest_path.stat().st_mtime == mtime_before

    uploaded_files = list((tmp_path / "uploads").glob("*.mp4"))
    assert len(uploaded_files) == 1  # no duplicate stored for the same content
