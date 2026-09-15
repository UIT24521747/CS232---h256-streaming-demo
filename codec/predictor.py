"""Prediction: the single idea that makes video compression work.

Instead of storing every pixel of every frame, we store a guess
(the "prediction") plus how wrong the guess was (the "residual",
see residual.py). If the guess is good, the residual is mostly
zeros and compresses very well.

Three kinds of guess are implemented here, one per frame type:

  I-frame : no other frame to look at, so guess each block from the
            blocks immediately to its left and above, *within the same
            frame* (already reconstructed, since blocks are coded in
            raster-scan order). Falls back to mid-gray (128) at the
            top-left corner where no neighbor exists yet. This is a
            simplified version of real intra-prediction (H.264/H.265's
            "DC" mode) -- it's what actually makes flat regions (like a
            gradient background) compress well without a transform.
  P-frame : guess the previous reconstructed frame, but let each
            block slide a little first (motion compensation) so a
            moving object still gets a good match.
  B-frame : guess the *average* of a motion-compensated previous and
            a motion-compensated future frame.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .frame import block_grid

INTRA_CONSTANT = 128  # mid-gray fallback -- same level-shift trick JPEG/MPEG use for DC


def predict_intra_block(
    reconstructed_so_far: np.ndarray, row: int, col: int, y: int, x: int, block_size: int
) -> np.ndarray:
    """DC intra prediction for one block: the flat value equal to the
    average of the reconstructed pixels immediately left and above it.

    `reconstructed_so_far` is the *current frame's own* reconstruction
    buffer, filled in raster-scan order as encoding/decoding proceeds --
    only pixels above the current row, or to the left in the current
    row, are valid when this is called. First block of the frame has
    neither neighbor, so it falls back to mid-gray.
    """
    samples = []
    if col > 0:
        samples.append(reconstructed_so_far[y : y + block_size, x - 1, :].astype(np.float64))
    if row > 0:
        samples.append(reconstructed_so_far[y - 1, x : x + block_size, :].astype(np.float64))

    if not samples:
        dc = np.full(3, INTRA_CONSTANT, dtype=np.float64)
    else:
        dc = np.concatenate(samples, axis=0).mean(axis=0)

    block = np.broadcast_to(dc, (block_size, block_size, 3))
    return block.astype(np.int16)


def _block_mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.int32) - b.astype(np.int32)
    return float(np.mean(diff * diff))


def motion_search(
    current_block: np.ndarray,
    ref_frame: np.ndarray,
    y: int,
    x: int,
    block_size: int,
    search_range: int,
) -> Tuple[int, int, np.ndarray, float]:
    """Full (exhaustive) search block matching in a small window.

    Tries every offset (dx, dy) in [-search_range, +search_range] around
    the block's own position in the reference frame, and keeps whichever
    offset gives the lowest mean-squared-error match. This is the
    simplest possible motion estimation -- real codecs use much faster
    search patterns, but exhaustive search is the one that's obvious to
    read and impossible to get subtly wrong.

    Returns (dx, dy, matched_block, mse).
    """
    h, w = ref_frame.shape[:2]
    best_mse = None
    best_dx, best_dy = 0, 0
    best_block = ref_frame[y : y + block_size, x : x + block_size]

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ry, rx = y + dy, x + dx
            if ry < 0 or rx < 0 or ry + block_size > h or rx + block_size > w:
                continue
            candidate = ref_frame[ry : ry + block_size, rx : rx + block_size]
            mse = _block_mse(current_block, candidate)
            if best_mse is None or mse < best_mse:
                best_mse, best_dx, best_dy, best_block = mse, dx, dy, candidate

    return best_dx, best_dy, best_block, float(best_mse)


MotionVectors = Dict[Tuple[int, int], Tuple[int, int, float]]


def predict_inter(
    current: np.ndarray, ref: np.ndarray, block_size: int, search_range: int
) -> Tuple[np.ndarray, MotionVectors]:
    """Motion-compensated prediction of `current` from one reference frame.

    Returns (predicted_frame, motion_vectors), where motion_vectors maps
    (block_row, block_col) -> (dx, dy, mse) for logging/inspection.
    """
    h, w = current.shape[:2]
    predicted = np.zeros_like(current, dtype=np.int16)
    motion_vectors: MotionVectors = {}

    for row, col, y, x in block_grid(h, w, block_size):
        cur_block = current[y : y + block_size, x : x + block_size]
        dx, dy, matched, mse = motion_search(cur_block, ref, y, x, block_size, search_range)
        predicted[y : y + block_size, x : x + block_size] = matched
        motion_vectors[(row, col)] = (dx, dy, mse)

    return predicted, motion_vectors


def predict_bidirectional(
    current: np.ndarray, ref_prev: np.ndarray, ref_next: np.ndarray, block_size: int, search_range: int
):
    """B-frame prediction: average of two independently motion-compensated guesses."""
    pred_prev, mv_prev = predict_inter(current, ref_prev, block_size, search_range)
    pred_next, mv_next = predict_inter(current, ref_next, block_size, search_range)
    predicted = ((pred_prev.astype(np.int32) + pred_next.astype(np.int32)) // 2).astype(np.int16)
    return predicted, mv_prev, mv_next
