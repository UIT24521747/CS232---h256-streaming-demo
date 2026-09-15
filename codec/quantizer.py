"""Uniform scalar quantization of the residual -- the actual lossy step.

Real codecs quantize residuals in the *frequency* domain, after a DCT,
because that concentrates energy into a few coefficients. We skip the
transform entirely and quantize pixel values directly: level = round(
residual / QP). It is less efficient, but it is one line of code and
the lossy step (division that throws away the remainder) is exactly
the same idea a real quantizer uses.

Bigger QP -> coarser steps -> smaller numbers to encode -> smaller
bitstream -> lower quality. That's the whole rate/quality knob.
"""
from __future__ import annotations

import numpy as np

QP_MIN, QP_MAX = 1, 32
RESIDUAL_CLIP = 127  # keeps quantized levels inside a signed byte, see entropy.py


def quantize(residual: np.ndarray, qp: int) -> np.ndarray:
    levels = np.round(residual.astype(np.float32) / qp)
    levels = np.clip(levels, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return levels.astype(np.int16)


def dequantize(levels: np.ndarray, qp: int) -> np.ndarray:
    return (levels.astype(np.int32) * qp).astype(np.int16)
