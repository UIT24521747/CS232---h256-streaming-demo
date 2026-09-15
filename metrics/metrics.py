"""Small, dependency-free image quality metrics.

Papers quote these three numbers constantly, so it's worth knowing
exactly what formula produces them instead of importing a library that
hides it. All three take two same-shape uint8 images (HxWx3, BGR).
"""
from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean((a64 - b64) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_value: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10 * np.log10((max_value**2) / m))


def _to_luma(img: np.ndarray) -> np.ndarray:
    """BGR -> luma (Y), same ITU-R BT.601 weights real codecs measure quality on."""
    if img.ndim == 2:
        return img.astype(np.float64)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return 0.114 * b.astype(np.float64) + 0.587 * g.astype(np.float64) + 0.299 * r.astype(np.float64)


def ssim(a: np.ndarray, b: np.ndarray, window: int = 8) -> float:
    """Simplified SSIM: the textbook local formula, averaged over non-overlapping
    windows instead of a Gaussian-weighted sliding window. Close enough to be a
    useful number for this demo without pulling in a stats/image library."""
    ya, yb = _to_luma(a), _to_luma(b)
    h, w = ya.shape
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    h_use, w_use = h - h % window, w - w % window
    if h_use == 0 or w_use == 0:
        window = min(h, w) or 1
        h_use, w_use = h, w

    scores = []
    for y in range(0, h_use, window):
        for x in range(0, w_use, window):
            pa = ya[y : y + window, x : x + window]
            pb = yb[y : y + window, x : x + window]
            mu_a, mu_b = pa.mean(), pb.mean()
            var_a, var_b = pa.var(), pb.var()
            cov = ((pa - mu_a) * (pb - mu_b)).mean()
            num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
            den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
            scores.append(num / den)
    return float(np.mean(scores)) if scores else 1.0


def compression_ratio(raw_size: int, encoded_size: int) -> float:
    if encoded_size == 0:
        return float("inf")
    return raw_size / encoded_size
