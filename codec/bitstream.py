"""Byte-exact layout of one encoded H256 frame.

This is the piece that makes the bitstream "readable by eye": every
field below has a fixed, documented position, packed with Python's
`struct` module -- no hidden container format, no external library.
`tools/inspect_bitstream.py` parses these exact same functions to
print a segment back out as text.

Frame record layout::

    [FRAME HEADER]                              (FRAME_HEADER_SIZE bytes)
      magic        4s    b"H256"                sanity check
      frame_id     I     uint32                 display-order frame number
      type_code    B     0=I 1=P 2=B
      ref1         i     int32 (-1 = none)       reference frame_id #1
      ref2         i     int32 (-1 = none)       reference frame_id #2 (B only)
      width        H     uint16 (padded width)
      height       H     uint16 (padded height)
      qp           B     quantization step
      block_size   B     block edge length (pixels)
      num_blocks   H     uint16, blocks that follow, raster-scan order

    [BLOCK] x num_blocks
      mv1_dx, mv1_dy   b, b   motion vector vs. ref1 (0,0 for I-frames)
      mv2_dx, mv2_dy   b, b   motion vector vs. ref2 (B-frames only)
      [RLE residual, channel 0 (B)]   see entropy.pack_pairs
      [RLE residual, channel 1 (G)]
      [RLE residual, channel 2 (R)]
"""
from __future__ import annotations

import struct
from typing import Optional

MAGIC = b"H256"

FRAME_HEADER_FMT = ">4sIBiiHHBBH"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)

BLOCK_MV_FMT = ">bbbb"
BLOCK_MV_SIZE = struct.calcsize(BLOCK_MV_FMT)


def pack_header(
    frame_id: int,
    type_code: int,
    ref1: Optional[int],
    ref2: Optional[int],
    width: int,
    height: int,
    qp: int,
    block_size: int,
    num_blocks: int,
) -> bytes:
    return struct.pack(
        FRAME_HEADER_FMT,
        MAGIC,
        frame_id,
        type_code,
        -1 if ref1 is None else ref1,
        -1 if ref2 is None else ref2,
        width,
        height,
        qp,
        block_size,
        num_blocks,
    )


def unpack_header(data: bytes, offset: int = 0):
    fields = struct.unpack_from(FRAME_HEADER_FMT, data, offset)
    magic, frame_id, type_code, ref1, ref2, width, height, qp, block_size, n_blocks = fields
    if magic != MAGIC:
        raise ValueError(f"bad H256 frame magic at offset {offset}: {magic!r}")
    header = {
        "frame_id": frame_id,
        "type_code": type_code,
        "ref1": None if ref1 == -1 else ref1,
        "ref2": None if ref2 == -1 else ref2,
        "width": width,
        "height": height,
        "qp": qp,
        "block_size": block_size,
        "num_blocks": n_blocks,
    }
    return header, offset + FRAME_HEADER_SIZE


def pack_block_header(dx1: int, dy1: int, dx2: int, dy2: int) -> bytes:
    return struct.pack(BLOCK_MV_FMT, dx1, dy1, dx2, dy2)


def unpack_block_header(data: bytes, offset: int):
    dx1, dy1, dx2, dy2 = struct.unpack_from(BLOCK_MV_FMT, data, offset)
    return (dx1, dy1, dx2, dy2), offset + BLOCK_MV_SIZE


def pack_segment_count(num_frames: int) -> bytes:
    return struct.pack(">H", num_frames)


def unpack_segment_count(data: bytes, offset: int = 0):
    (n,) = struct.unpack_from(">H", data, offset)
    return n, offset + 2
