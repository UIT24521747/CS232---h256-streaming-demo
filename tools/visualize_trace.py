"""Save one traced frame's intermediate images + a single composite grid.

    outputs/<video_id>/users/<user_id>/frame_<id>/
        01_original.png
        02_prediction.png
        03_residual.png
        04_reconstructed.png
        05_sr.png
        mv_overlay.png     <- MV of every block drawn as arrows
        trace.json
        summary.png        <- the 2x3 grid, the one image to show a grader

Run directly to rebuild summary.png (+ mv_overlay.png if the trace has MV
entries) from an existing frame_<id>/ folder:
    python tools/visualize_trace.py outputs/<video_id>/users/<user_id>/frame_12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from codec.h256_encoder import BLOCK_SIZE

LABEL_COLOR = (30, 30, 30)
LABEL_BG = (235, 235, 230)
MV_COLORS = [(0, 200, 255), (255, 140, 0)]  # BGR: ref1 arrows, ref2 arrows (B-frames)


def _residual_to_image(residual: np.ndarray) -> np.ndarray:
    """Signed residual -> viewable image, centered on mid-gray (128)."""
    return np.clip(residual.astype(np.int16) + 128, 0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = max(16, h // 8)
    cv2.rectangle(out, (0, 0), (w, bar_h), LABEL_BG, -1)
    cv2.putText(out, text, (4, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, LABEL_COLOR, 1, cv2.LINE_AA)
    return out


def draw_mv_overlay(image: np.ndarray, mv_entries: list, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """One arrow per block per reference, from the block's own center to
    the reference offset motion_search() found for it (codec/mv_index.py
    entries) -- makes the "which block came from where" of README section
    6.2/7.2 visible instead of just numbers."""
    out = image.copy()
    for entry in mv_entries:
        x, y = entry["pos"]
        cx, cy = x + block_size // 2, y + block_size // 2
        for i, mv in enumerate(entry.get("mv") or []):
            dx, dy = mv
            if dx == 0 and dy == 0:
                continue
            end = (cx + dx, cy + dy)
            cv2.arrowedLine(out, (cx, cy), end, MV_COLORS[i % len(MV_COLORS)], 1, tipLength=0.35)
    return out


def save_frame_outputs(trace: dict, out_dir: Path) -> Path:
    """Write the five stage images + mv_overlay.png + trace.json +
    summary.png for one traced frame into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    images = trace["images"]
    cv2.imwrite(str(out_dir / "01_original.png"), images["original"])
    cv2.imwrite(str(out_dir / "02_prediction.png"), images["prediction"])
    cv2.imwrite(str(out_dir / "03_residual.png"), _residual_to_image(images["residual"]))
    cv2.imwrite(str(out_dir / "04_reconstructed.png"), images["reconstructed"])
    cv2.imwrite(str(out_dir / "05_sr.png"), images["sr"])

    overlay = draw_mv_overlay(images["reconstructed"], trace.get("mv") or [])
    cv2.imwrite(str(out_dir / "mv_overlay.png"), overlay)

    json_safe = {k: v for k, v in trace.items() if k != "images"}
    (out_dir / "trace.json").write_text(json.dumps(json_safe, indent=2), encoding="utf-8")

    summary = make_composite(trace)
    cv2.imwrite(str(out_dir / "summary.png"), summary)
    return out_dir


def make_composite(trace: dict) -> np.ndarray:
    """2x3 grid: original | prediction | residual / reconstructed | SR | (info panel)."""
    images = trace["images"]
    cell_w, cell_h = 220, 160

    def fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min((cell_w - 8) / w, (cell_h - 28) / h)
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
        rh, rw = resized.shape[:2]
        y0 = 24 + (cell_h - 24 - rh) // 2
        x0 = (cell_w - rw) // 2
        canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
        return canvas

    info_panel = np.full((cell_h, cell_w, 3), 250, dtype=np.uint8)
    lines = [
        f"Frame {trace['frame_id']}  [{trace['frame_type']['type']}]",
        f"quality: {trace['quality']}",
        f"ref1={trace['frame_type']['ref1']} ref2={trace['frame_type']['ref2']}",
        f"QP={trace['quantization']['qp']}",
        f"raw={trace['h256_encode']['raw_size']}B",
        f"enc={trace['h256_encode']['encoded_size']}B",
        f"comp={trace['h256_encode']['compression_pct']:.1f}%",
        f"PSNR={trace['reconstruction']['psnr']:.2f}dB",
        f"node={trace['load_balancer']['selected_node']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info_panel, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_COLOR, 1, cv2.LINE_AA)

    cells = [
        _label(fit(images["original"]), "ORIGINAL"),
        _label(fit(images["prediction"]), "PREDICTION"),
        _label(fit(_residual_to_image(images["residual"])), "RESIDUAL"),
        _label(fit(images["reconstructed"]), "RECONSTRUCTED"),
        _label(fit(images["sr"]), "SR OUTPUT"),
        info_panel,
    ]

    row1 = np.hstack(cells[0:3])
    row2 = np.hstack(cells[3:6])
    grid = np.vstack([row1, row2])
    border = np.full((grid.shape[0] + 4, grid.shape[1] + 4, 3), 200, dtype=np.uint8)
    border[2:-2, 2:-2] = grid
    return border


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild summary.png (+ mv_overlay.png) for an already-traced frame")
    parser.add_argument("frame_dir", help="e.g. outputs/<video_id>/users/<user_id>/frame_12")
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    trace_path = frame_dir / "trace.json"
    if not trace_path.exists():
        print(f"no trace.json in {frame_dir}", file=sys.stderr)
        sys.exit(1)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["images"] = {
        "original": cv2.imread(str(frame_dir / "01_original.png")),
        "prediction": cv2.imread(str(frame_dir / "02_prediction.png")),
        "residual": cv2.imread(str(frame_dir / "03_residual.png")).astype(np.int16) - 128,
        "reconstructed": cv2.imread(str(frame_dir / "04_reconstructed.png")),
        "sr": cv2.imread(str(frame_dir / "05_sr.png")),
    }
    summary = make_composite(trace)
    cv2.imwrite(str(frame_dir / "summary.png"), summary)
    print(f"wrote {frame_dir / 'summary.png'}")

    if trace.get("mv"):
        overlay = draw_mv_overlay(trace["images"]["reconstructed"], trace["mv"])
        cv2.imwrite(str(frame_dir / "mv_overlay.png"), overlay)
        print(f"wrote {frame_dir / 'mv_overlay.png'}")


if __name__ == "__main__":
    main()
