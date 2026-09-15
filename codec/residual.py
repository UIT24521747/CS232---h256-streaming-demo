"""Residual = original - prediction.

This is the payload that actually gets compressed. If the prediction
in predictor.py was good, the residual is close to zero everywhere and
the quantizer/entropy stages downstream can squash it hard.
"""
from __future__ import annotations

import numpy as np


def compute_residual(original: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """original and predicted are pixel arrays (0-255-ish); residual can be negative."""
    return original.astype(np.int16) - predicted.astype(np.int16)


def reconstruct(predicted: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Inverse of compute_residual: prediction + residual, clipped back to a valid pixel."""
    out = predicted.astype(np.int16) + residual.astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def residual_mse(residual: np.ndarray) -> float:
    """Mean squared residual -- a quick 'how wrong was the prediction' number."""
    return float(np.mean(residual.astype(np.float64) ** 2))
