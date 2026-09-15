"""One streaming node: a small Flask app serving pre-encoded H256 segments
straight off disk. The demo runs two of these (Node-1, Node-2), each
mirroring the *same* `outputs/` root -- standing in for replicated CDN edge
nodes. The load balancer (load_balancer.py) is the only thing that knows
both exist.

Routes are scoped by `video_id` (README section 4/9.3) so a node can serve
any number of encoded videos without restarting -- it just resolves
`outputs_root/<video_id>/segments/...` per request.

Routes:
    GET /video/<video_id>                    -> manifest: qualities + segment names
    GET /segment/<video_id>/<quality>/<name> -> raw bytes of one .bin segment
    GET /health                               -> liveness check
"""
from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, abort, jsonify, send_file


def create_app(node_name: str, outputs_root: Path) -> Flask:
    app = Flask(node_name)
    outputs_root = Path(outputs_root)

    def _segments_dir(video_id: str) -> Path:
        return outputs_root / video_id / "segments"

    @app.get("/video/<video_id>")
    def manifest(video_id: str):
        segments_dir = _segments_dir(video_id)
        qualities = sorted(p.name for p in segments_dir.iterdir() if p.is_dir()) if segments_dir.exists() else []
        segments = {q: sorted(p.name for p in (segments_dir / q).glob("*.bin")) for q in qualities}
        return jsonify({"node": node_name, "video_id": video_id, "qualities": qualities, "segments": segments})

    @app.get("/segment/<video_id>/<quality>/<name>")
    def segment(video_id: str, quality: str, name: str):
        path = _segments_dir(video_id) / quality / name
        if not path.exists():
            abort(404)
        resp = send_file(path, mimetype="application/octet-stream")
        resp.headers["X-Node-Name"] = node_name
        return resp

    @app.get("/health")
    def health():
        return jsonify({"node": node_name, "status": "ok"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one H256 streaming node")
    parser.add_argument("--name", default="Node-1")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--dir", default="outputs", help="outputs/ root (holds one subdir per video_id)")
    args = parser.parse_args()
    create_app(args.name, Path(args.dir)).run(port=args.port)


if __name__ == "__main__":
    main()
