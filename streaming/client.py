"""Streaming client: talks only to the load balancer, simulates a limited
network, and picks a quality tier the way real adaptive streaming does --
by comparing the simulated bandwidth against a small fixed ladder.

One `StreamingClient` exists per user (see streaming/session.py and
pipeline.run_all_users) -- each carries its own `user_id` and its own
(possibly time-varying) bandwidth, and both are attached to every request
so the load balancer can log which user asked for what.

Loopback HTTP on one machine is essentially instant, which would make a
"bandwidth" slider pointless. So after every segment download we compute
how long that download *should* have taken at the target bandwidth and
sleep off the difference. That's the only thing standing in for a real,
slow network in this demo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Union

import requests

# (quality name, minimum simulated bandwidth in kbps to select it)
QUALITY_LADDER = [
    ("240p", 0),
    ("360p", 1200),
    ("540p", 2500),
]


def choose_quality(bandwidth_kbps: float) -> str:
    chosen = QUALITY_LADDER[0][0]
    for name, min_kbps in QUALITY_LADDER:
        if bandwidth_kbps >= min_kbps:
            chosen = name
    return chosen


@dataclass
class BufferState:
    segments: List[str] = field(default_factory=list)

    def as_ascii(self, now_playing: str | None = None) -> str:
        """Render like: [seg01][seg02][seg03]  (playing: seg03)"""
        chips = "".join(f"[{s}]" for s in self.segments[-6:])
        return f"{chips}  (playing: {now_playing})" if now_playing else chips


class StreamingClient:
    def __init__(
        self,
        lb_url: str,
        video_id: str,
        user_id: str,
        bandwidth_kbps: Union[float, Callable[[], float]] = 2000,
        log_fn: Callable[[str], None] = print,
    ):
        self.lb_url = lb_url.rstrip("/")
        self.video_id = video_id
        self.user_id = user_id
        self._bandwidth = bandwidth_kbps
        self.buffer = BufferState()
        self.log_fn = log_fn

    @property
    def bandwidth_kbps(self) -> float:
        return float(self._bandwidth()) if callable(self._bandwidth) else float(self._bandwidth)

    def fetch_manifest(self) -> dict:
        r = requests.get(f"{self.lb_url}/video/{self.video_id}", timeout=5)
        r.raise_for_status()
        return r.json()

    def fetch_segment(self, name: str) -> tuple[bytes, str, str, float, int]:
        bandwidth = self.bandwidth_kbps
        quality = choose_quality(bandwidth)

        t0 = time.perf_counter()
        r = requests.get(
            f"{self.lb_url}/segment/{self.video_id}/{quality}/{name}",
            params={"user": self.user_id},
            timeout=15,
        )
        r.raise_for_status()
        data = r.content
        elapsed = time.perf_counter() - t0

        simulated_seconds = (len(data) * 8) / max(bandwidth, 1) / 1000
        if simulated_seconds > elapsed:
            time.sleep(simulated_seconds - elapsed)

        node = r.headers.get("X-Selected-Node", "unknown")
        lb_counter = int(r.headers.get("X-LB-Counter", -1))
        self.buffer.segments.append(name)
        self.log_fn(
            f"[Client] user={self.user_id} segment={name} quality={quality} node={node} "
            f"bytes={len(data)} bandwidth={bandwidth:.0f}kbps simulated_time={simulated_seconds:.3f}s"
        )
        return data, quality, node, simulated_seconds, lb_counter
