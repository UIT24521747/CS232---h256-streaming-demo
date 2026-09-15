"""Tests for sr/simple_sr.py -- the hand-written bilinear resize and
unsharp-mask sharpen used for Simple Super Resolution."""
from __future__ import annotations

import numpy as np

from sr.simple_sr import bilinear_upscale, resize_bilinear, sharpen, simple_super_resolution


def test_sr():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:12, 4:12] = (200, 100, 50)

    upscaled = bilinear_upscale(img, scale=2)
    assert upscaled.shape == (32, 32, 3)
    assert upscaled.dtype == np.uint8

    # Downscaling the upscaled image back should land close to the
    # original -- proves the interpolation is a real geometric mapping,
    # not noise.
    roundtrip = resize_bilinear(upscaled, 16, 16)
    mse = float(np.mean((roundtrip.astype(np.int32) - img.astype(np.int32)) ** 2))
    assert mse < 200

    sharpened = sharpen(img, amount=0.6)
    assert sharpened.shape == img.shape
    assert sharpened.dtype == np.uint8

    sr_out = simple_super_resolution(img, scale=2)
    assert sr_out.shape == (32, 32, 3)
