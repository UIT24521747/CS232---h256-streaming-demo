"""Round-robin load balancer sitting in front of the streaming nodes.

Every client (client.py, one per user) only ever talks to the load
balancer, never directly to a node. Every request advances one *shared*
counter and picks `node[counter % N]` -- simple on purpose, so the log line
underneath is the whole explanation:

    user=u2 request=360p/segment_001.bin -> node[1]=Node-2 (Round-Robin)

Because the counter is shared across every user, requests from different
users interleave -- one user can get two segments in a row from the same
node purely by chance (README section 5.10). The raw (non-modulo) counter
value is also handed back as `X-LB-Counter` so a client can record exactly
which request index it was (README section 7.1's `lb_counter` field).
"""
from __future__ import annotations

import argparse
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from flask import Flask, Response, abort, jsonify, request


@dataclass
class LoadBalancer:
    node_urls: List[str]
    log: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def pick_node(self, request_name: str, user_id: Optional[str] = None) -> tuple[int, str]:
        with self._lock:
            counter = next(self._counter)
        i = counter % len(self.node_urls)
        node = self.node_urls[i]
        entry = {
            "user": user_id,
            "request": request_name,
            "node_index": i,
            "selected_node": node,
            "policy": "Round-Robin",
            "lb_counter": counter,
            "time": time.time(),
        }
        self.log.append(entry)
        who = f"user={user_id} " if user_id else ""
        print(f"[LoadBalancer] {who}request={request_name} -> node[{i}]={node} (Round-Robin)")
        return counter, node


def create_app(node_urls: List[str]) -> Flask:
    app = Flask("load-balancer")
    lb = LoadBalancer(node_urls=node_urls)
    app.config["lb"] = lb

    @app.get("/segment/<video_id>/<quality>/<name>")
    def segment(video_id: str, quality: str, name: str):
        user_id = request.args.get("user")
        counter, node = lb.pick_node(f"{quality}/{name}", user_id)
        try:
            r = requests.get(f"{node}/segment/{video_id}/{quality}/{name}", timeout=5)
        except requests.RequestException:
            abort(502)
        if r.status_code != 200:
            abort(r.status_code)
        resp = Response(r.content, mimetype="application/octet-stream")
        resp.headers["X-Selected-Node"] = r.headers.get("X-Node-Name", node)
        resp.headers["X-LB-Counter"] = str(counter)
        return resp

    @app.get("/video/<video_id>")
    def manifest(video_id: str):
        _, node = lb.pick_node("manifest")
        r = requests.get(f"{node}/video/{video_id}", timeout=5)
        return Response(r.content, mimetype="application/json")

    @app.get("/log")
    def log():
        return jsonify(lb.log[-100:])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the H256 round-robin load balancer")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--nodes", nargs="+", default=["http://127.0.0.1:6001", "http://127.0.0.1:6002"])
    args = parser.parse_args()
    create_app(args.nodes).run(port=args.port)


if __name__ == "__main__":
    main()
