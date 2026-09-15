"""Frame container and block-grid utilities shared by every codec stage."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import numpy as np


class FrameType(Enum):
    I = "I"   # Intra: encoded on its own, no reference to any other frame
    P = "P"   # Predicted: encoded as a delta from one earlier reference frame
    B = "B"   # Bi-directional: delta from one earlier AND one later reference frame


@dataclass
class Frame:
    """One raw video frame plus the codec bookkeeping around it.

    `data` is HxWx3 uint8 in OpenCV's BGR channel order. `ref1`/`ref2`
    hold the frame_id(s) this frame is predicted from (None for I).
    """

    frame_id: int
    type: FrameType
    data: np.ndarray
    ref1: Optional[int] = None
    ref2: Optional[int] = None

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]


def pad_to_block_size(image: np.ndarray, block_size: int) -> np.ndarray:
    """Pad an image so both dimensions are a multiple of `block_size`.

    Real codecs do this too (edge-extend the last row/column) so the
    block grid always divides evenly; we crop back to the original size
    on the way out.
    """
    h, w = image.shape[:2]
    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def block_grid(height: int, width: int, block_size: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield (row, col, y, x) for every block in raster-scan order.

    Raster scan (left to right, top to bottom) is what real codecs use
    too -- it's the order blocks are written to the bitstream in, and
    the order `inspect_bitstream.py` prints them back out.
    """
    rows = height // block_size
    cols = width // block_size
    for row in range(rows):
        for col in range(cols):
            yield row, col, row * block_size, col * block_size


def num_blocks(height: int, width: int, block_size: int) -> int:
    return (height // block_size) * (width // block_size)
