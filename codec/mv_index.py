"""Per-video, per-quality index of block-level motion vectors + residual
stats, written once per encode so a per-user MV lookup (README section 6.2)
never has to touch the bitstream: `mv_index/<quality>.json`, keyed by
frame_id, one entry per block.

The index deliberately mirrors the bitstream's own block order exactly: all
three quality tiers (240p/360p/540p) are exact multiples of BLOCK_SIZE, so
`EncodeTrace.quantized` (already cropped to the frame's own width/height)
has the identical shape `h256_encoder._pack` used when writing the
bitstream, and `frame.block_grid` walks both in the same raster order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import entropy
from .frame import FrameType, block_grid
from .h256_encoder import EncodeTrace

FrameEntries = List[dict]
Index = Dict[int, FrameEntries]


def frame_entries(trace: EncodeTrace, block_size: int) -> FrameEntries:
    """Build this one frame's block records for the MV index."""
    h, w = trace.quantized.shape[:2]
    mv_map = trace.motion_vectors
    is_b = trace.frame_type is FrameType.B

    entries: FrameEntries = []
    for i, (row, col, y, x) in enumerate(block_grid(h, w, block_size)):
        block_levels = trace.quantized[y : y + block_size, x : x + block_size]
        nonzero = int(np.count_nonzero(block_levels))
        rle_bytes = sum(
            len(entropy.pack_pairs(entropy.rle_encode(block_levels[:, :, ch]))) for ch in range(3)
        )

        if trace.frame_type is FrameType.I:
            mv, match_mse = [], []
        elif is_b:
            dx1, dy1, mse1 = mv_map["prev"][(row, col)]
            dx2, dy2, mse2 = mv_map["next"][(row, col)]
            mv = [[dx1, dy1], [dx2, dy2]]
            match_mse = [round(mse1, 2), round(mse2, 2)]
        else:
            dx, dy, mse1 = mv_map[(row, col)]
            mv = [[dx, dy]]
            match_mse = [round(mse1, 2)]

        entries.append(
            {
                "frame_id": trace.frame_id,
                "block": i,
                "pos": [x, y],
                "mv": mv,
                "match_mse": match_mse,
                "nonzero_levels": nonzero,
                "rle_bytes": rle_bytes,
            }
        )
    return entries


def write_index(path: Path, frames: Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(fid): entries for fid, entries in sorted(frames.items())}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(path: Path) -> Index:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(fid): entries for fid, entries in data.items()}


def lookup(index: Index, frame_id: int, block: int | None = None):
    """Look up either every block of `frame_id` (block=None) or one block."""
    entries = index.get(frame_id, [])
    if block is None:
        return entries
    for e in entries:
        if e["block"] == block:
            return e
    return None
