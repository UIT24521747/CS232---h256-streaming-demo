"""The last compression stage: zero run-length coding.

After quantization most residual values are exactly 0 (flat regions,
a good motion match). Run-length coding a mostly-zero integer stream
is the same core idea real codecs use before a fancier entropy stage
(Huffman / CABAC) -- we stop at RLE because it's the point where you
can still read every number with your own eyes.

Encoding: walk the flattened block. Count consecutive zeros ("run").
The moment a non-zero value `v` appears, emit the pair (run, v) and
reset the run to zero. Trailing zeros at the end of a block are never
written at all -- the decoder just pads with zeros up to the known
block size. A run longer than 255 (fits a byte) is split into several
(255, 0) "skip" pairs; a pair with value 0 is unambiguous because a
*real* value is never zero by construction.
"""
from __future__ import annotations

import struct
from typing import List, Tuple

import numpy as np

MAX_RUN = 255  # fits in one unsigned byte

Pairs = List[Tuple[int, int]]


def rle_encode(values: np.ndarray) -> Pairs:
    flat = values.flatten().astype(int).tolist()
    pairs: Pairs = []
    run = 0
    for v in flat:
        if v == 0:
            run += 1
            while run > MAX_RUN:
                pairs.append((MAX_RUN, 0))  # long zero-run split, not a real value
                run -= MAX_RUN
        else:
            pairs.append((run, v))
            run = 0
    return pairs


def rle_decode(pairs: Pairs, count: int) -> np.ndarray:
    out: List[int] = []
    for run, val in pairs:
        out.extend([0] * run)
        if val != 0:
            out.append(val)
    if len(out) < count:
        out.extend([0] * (count - len(out)))
    return np.array(out[:count], dtype=np.int16)


def pack_pairs(pairs: Pairs) -> bytes:
    """(run: unsigned byte, value: signed byte) pairs, length-prefixed."""
    body = b"".join(struct.pack(">Bb", run, val) for run, val in pairs)
    return struct.pack(">H", len(pairs)) + body


def unpack_pairs(data: bytes, offset: int = 0) -> Tuple[Pairs, int]:
    (n,) = struct.unpack_from(">H", data, offset)
    offset += 2
    pairs: Pairs = []
    for _ in range(n):
        run, val = struct.unpack_from(">Bb", data, offset)
        pairs.append((run, val))
        offset += 2
    return pairs, offset
