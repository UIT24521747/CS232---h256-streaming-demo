"""Generate a small synthetic test clip so the demo needs no external
video file or download. A panning gradient background plus a bouncing
ball give the motion-compensated predictor (P/B frames) real motion to
search for -- a static image would make every motion vector (0, 0) and
the whole point of predictor.py would be invisible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 256, 144
FPS = 12
NUM_FRAMES = 32


def make_frame(t: int, num_frames: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Panning gradient background: green sweeps one way, red the other.
    # Kept gentle (a few px/frame) on purpose -- it has to stay inside the
    # codec's motion search window (see codec/h256_encoder.py SEARCH_RANGE),
    # the same way a real encoder's search range bounds how fast motion it
    # can track.
    shift = int((t / num_frames) * WIDTH * 0.3)
    xx = (np.arange(WIDTH) + shift) % WIDTH
    gradient = (xx * 255 // WIDTH).astype(np.uint8)
    frame[:, :, 0] = 40
    frame[:, :, 1] = gradient[None, :]
    frame[:, :, 2] = 255 - gradient[None, :]

    # Ball drifting left-right and gently bobbing -- the thing motion search tracks.
    cx = int(WIDTH * 0.3 + (WIDTH * 0.4) * (t / num_frames))
    cy = int(HEIGHT * 0.5 + HEIGHT * 0.15 * np.sin(t * 0.12))
    cv2.circle(frame, (cx, cy), 14, (245, 245, 245), -1)
    cv2.circle(frame, (cx, cy), 14, (20, 20, 20), 2)

    # Static high-contrast block: gives I-frame intra coding a hard edge to keep.
    cv2.rectangle(frame, (10, 10), (46, 30), (30, 200, 30), -1)
    return frame


def generate(path: Path, num_frames: int = NUM_FRAMES, fps: int = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    for t in range(num_frames):
        writer.write(make_frame(t, num_frames))
    writer.release()
    print(f"wrote {num_frames} frames ({WIDTH}x{HEIGHT} @ {fps}fps) -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic CS232 sample clip")
    parser.add_argument("--out", default=str(Path(__file__).parent / "sample.mp4"))
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    generate(Path(args.out), args.frames, args.fps)


if __name__ == "__main__":
    main()
