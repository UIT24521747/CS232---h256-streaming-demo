"""Tests for streaming/: one node server, and the round-robin load balancer
sitting in front of two of them -- both run as real Flask apps on real
ports, exercised over real HTTP. Routes are scoped by video_id (see
streaming/server.py, streaming/load_balancer.py) so one node can serve
segments for any number of encoded videos.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import requests
from werkzeug.serving import make_server

from streaming.client import choose_quality
from streaming.load_balancer import create_app as create_lb_app
from streaming.server import create_app as create_node_app

NODE_PORTS = [17101, 17102]
LB_PORT = 17100
VIDEO_ID = "deadbeef"


def _fake_outputs_root(tmp_path: Path) -> Path:
    segments_dir = tmp_path / VIDEO_ID / "segments"
    for quality in ("240p", "360p"):
        qdir = segments_dir / quality
        qdir.mkdir(parents=True)
        for i in range(2):
            (qdir / f"segment_{i:03d}.bin").write_bytes(bytes([i]) * 100)
    return tmp_path


@pytest.fixture
def running_stack(tmp_path):
    outputs_root = _fake_outputs_root(tmp_path)
    servers, threads, node_urls = [], [], []

    for i, port in enumerate(NODE_PORTS, start=1):
        srv = make_server("127.0.0.1", port, create_node_app(f"Node-{i}", outputs_root))
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        servers.append(srv)
        threads.append(th)
        node_urls.append(f"http://127.0.0.1:{port}")

    lb_srv = make_server("127.0.0.1", LB_PORT, create_lb_app(node_urls))
    lb_th = threading.Thread(target=lb_srv.serve_forever, daemon=True)
    lb_th.start()
    servers.append(lb_srv)
    threads.append(lb_th)

    time.sleep(0.2)
    yield f"http://127.0.0.1:{LB_PORT}", node_urls

    for s in servers:
        s.shutdown()
    for t in threads:
        t.join(timeout=2)


def test_streaming(running_stack):
    _lb_url, node_urls = running_stack

    r = requests.get(f"{node_urls[0]}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = requests.get(f"{node_urls[0]}/segment/{VIDEO_ID}/240p/segment_000.bin", timeout=5)
    assert r.status_code == 200
    assert r.content == bytes([0]) * 100
    assert r.headers["X-Node-Name"] == "Node-1"


def test_load_balancer(running_stack):
    lb_url, _node_urls = running_stack

    selected = []
    for i in range(4):
        r = requests.get(
            f"{lb_url}/segment/{VIDEO_ID}/240p/segment_{i % 2:03d}.bin", params={"user": "u1"}, timeout=5
        )
        assert r.status_code == 200
        assert "X-LB-Counter" in r.headers
        selected.append(r.headers["X-Selected-Node"])

    # strict alternation -- the definition of round-robin
    assert selected[0] != selected[1]
    assert selected[0] == selected[2]
    assert selected[1] == selected[3]

    log = requests.get(f"{lb_url}/log", timeout=5).json()
    assert len(log) >= 4
    assert all(entry["policy"] == "Round-Robin" for entry in log)
    assert any(entry["user"] == "u1" for entry in log)


def test_choose_quality_thresholds():
    assert choose_quality(0) == "240p"
    assert choose_quality(1200) == "360p"
    assert choose_quality(2500) == "540p"
    assert choose_quality(10_000) == "540p"
