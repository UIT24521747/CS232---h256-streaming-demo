"""Simple Super Resolution: bilinear upscaling + a hand-written sharpen pass.

No learned model, no GAN, no pretrained checkpoint. The point of this
module is that both operations are formulas you can read in ten
seconds, not a black box you have to trust:

  1. resize_bilinear -- for every *output* pixel, find where it would
     land in the *input* image (which is almost always between four
     input pixels), then blend those four neighbors weighted by how
     close the output pixel is to each one. This is the same routine
     used to shrink frames for the streaming quality ladder (see
     pipeline.py) and to grow them back for SR -- one primitive, both
     directions.

  2. sharpen -- an unsharp mask: blend the image with a 3x3 kernel that
     boosts a pixel relative to its neighbors ([[0,-1,0],[-1,5,-1],[0,-1,0]]).
     Upscaling alone looks soft; this restores some perceived detail
     without inventing any new content.
"""
from __future__ import annotations

import numpy as np


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    src = img.astype(np.float64)
    scale_y, scale_x = h / out_h, w / out_w

    # Map each output coordinate back into source-image space.
    ys = np.clip((np.arange(out_h) + 0.5) * scale_y - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * scale_x - 0.5, 0, w - 1)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = (ys - y0)[:, None, None]  # how far down towards the *next* row
    wx = (xs - x0)[None, :, None]  # how far right towards the *next* column

    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.clip(out, 0, 255).astype(np.uint8)


def bilinear_upscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    return resize_bilinear(img, h * scale, w * scale)


def sharpen(img: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Unsharp mask via a hand-written 3x3 convolution (no cv2.filter2D)."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    h, w = img.shape[:2]
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge").astype(np.float64)

    filtered = np.zeros((h, w, img.shape[2]), dtype=np.float64)
    for ky in range(3):
        for kx in range(3):
            weight = kernel[ky, kx]
            if weight == 0:
                continue
            filtered += weight * padded[ky : ky + h, kx : kx + w]

    blended = img.astype(np.float64) * (1 - amount) + filtered * amount
    return np.clip(blended, 0, 255).astype(np.uint8)


def simple_super_resolution(img: np.ndarray, scale: int = 2, sharpen_amount: float = 0.6) -> np.ndarray:
    """LR frame -> bilinear upscale x`scale` -> sharpen -> HR frame."""
    upscaled = bilinear_upscale(img, scale)
    return sharpen(upscaled, sharpen_amount)
