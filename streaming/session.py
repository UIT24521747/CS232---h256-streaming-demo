"""Per-user session state for the multi-user demo (README section 6).

`SessionManager` holds one `UserSession` per `user_id`: bandwidth, current
ABR quality, buffer, and the full receive history the web dashboard's user
grid and per-frame detail views read from. Everyone shares one encoded
video; only the session state (and, through it, which quality/MVs a user
actually sees) differs between users -- see pipeline.run_all_users.

Buffer level is simulated against real wall-clock time: `start_playback`
records when a user "starts watching", and each `push` works out how many
frames a player would have consumed by now at the video's native fps and
drops them from the pending count. That makes `rebuffer` (buffer hits zero
while still mid-playback) a real, checkable condition instead of a fake
number -- see README section 6.3.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UserSession:
    user_id: str
    bandwidth_kbps: float
    quality: Optional[str] = None
    buffer: List[str] = field(default_factory=list)  # recent segment names, most-recent last
    history: List[dict] = field(default_factory=list)  # one record per received frame (README 7.1)
    results: List[Any] = field(default_factory=list)  # pipeline.ClientFrameResult, kept for SR/trace reuse
    total_frames: int = 0
    _start_time: Optional[float] = field(default=None, repr=False)
    _fps: float = field(default=12.0, repr=False)
    _pending_frame_ids: List[int] = field(default_factory=list, repr=False)
    _played_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start_playback(self, fps: float, total_frames: int) -> None:
        with self._lock:
            self._start_time = time.perf_counter()
            self._fps = max(fps, 1e-6)
            self._pending_frame_ids = []
            self._played_count = 0
            self.total_frames = total_frames
            self.buffer = []
            self.history = []
            self.results = []
            self.quality = None

    @property
    def buffer_level_s(self) -> float:
        with self._lock:
            return len(self._pending_frame_ids) / self._fps

    @property
    def rebuffering(self) -> bool:
        with self._lock:
            unfinished = self._played_count + len(self._pending_frame_ids) < self.total_frames
            return unfinished and not self._pending_frame_ids and self._played_count > 0

    def push(self, record: dict) -> dict:
        """Record one received/decoded frame; fills in recv_order and
        buffer_level_s, appends to history, and returns the completed
        record (the same dict the caller passed in)."""
        with self._lock:
            now = time.perf_counter()
            if self._start_time is None:
                self._start_time = now
            elapsed = now - self._start_time
            consumed = int(elapsed * self._fps)
            self._pending_frame_ids.append(record["frame_id"])
            while self._played_count < consumed and self._pending_frame_ids:
                self._pending_frame_ids.pop(0)
                self._played_count += 1

            record["recv_order"] = len(self.history)
            record["buffer_level_s"] = round(len(self._pending_frame_ids) / self._fps, 3)
            self.history.append(record)
            self.quality = record.get("quality", self.quality)
            if record.get("segment") and (not self.buffer or self.buffer[-1] != record["segment"]):
                self.buffer.append(record["segment"])
        return record

    def snapshot(self) -> dict:
        with self._lock:
            last = self.history[-1] if self.history else None
            psnr_values = [h["psnr_db"] for h in self.history if h.get("psnr_db") is not None]
            total_bytes = sum(h.get("encoded_bytes", 0) for h in self.history)
            return {
                "user_id": self.user_id,
                "bandwidth_kbps": self.bandwidth_kbps,
                "quality": self.quality,
                "frames_received": len(self.history),
                "total_frames": self.total_frames,
                "buffer_level_s": round(len(self._pending_frame_ids) / self._fps, 3),
                "rebuffering": (
                    self._played_count + len(self._pending_frame_ids) < self.total_frames
                    and not self._pending_frame_ids
                    and self._played_count > 0
                ),
                "avg_psnr": round(sum(psnr_values) / len(psnr_values), 2) if psnr_values else None,
                "total_bytes": total_bytes,
                "last_frame": last,
            }


@dataclass
class SessionManager:
    sessions: Dict[str, UserSession] = field(default_factory=dict)
    _next_id: int = field(default=1, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def create_user(self, bandwidth_kbps: float, user_id: Optional[str] = None) -> UserSession:
        with self._lock:
            while user_id is None or user_id in self.sessions:
                user_id = f"u{self._next_id}"
                self._next_id += 1
            session = UserSession(user_id=user_id, bandwidth_kbps=bandwidth_kbps)
            self.sessions[user_id] = session
            return session

    def remove_user(self, user_id: str) -> bool:
        with self._lock:
            return self.sessions.pop(user_id, None) is not None

    def get(self, user_id: str) -> Optional[UserSession]:
        return self.sessions.get(user_id)

    def all(self) -> List[UserSession]:
        return list(self.sessions.values())

    def set_bandwidth(self, user_id: str, bandwidth_kbps: float) -> bool:
        session = self.sessions.get(user_id)
        if session is None:
            return False
        session.bandwidth_kbps = bandwidth_kbps
        return True
