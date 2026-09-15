"""Human-readable dump of an H256 segment .bin file.

Parses the exact same struct layout the encoder/decoder use (see
codec/bitstream.py) but only to print it -- no pixel reconstruction
happens here. This is the file that proves the bitstream isn't a black
box: every byte it reads has a name.

Usage:
    python tools/inspect_bitstream.py outputs/<video_id>/segments/540p/segment_000.bin
    python tools/inspect_bitstream.py outputs/<video_id>/segments/540p/segment_000.bin --frame 1
    python tools/inspect_bitstream.py outputs/<video_id>/segments/540p/segment_000.bin --frame 1 --block 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codec import bitstream, entropy  # noqa: E402

TYPE_NAME = {0: "I", 1: "P", 2: "B"}


def parse_segment(data: bytes):
    """Yield (header, blocks) for every frame record in the segment.

    Each block is {row, col, x, y, mv1, mv2, residual: {channel: rle_pairs}}.
    """
    n, offset = bitstream.unpack_segment_count(data)
    for _ in range(n):
        header, offset = bitstream.unpack_header(data, offset)
        cols = header["width"] // header["block_size"]
        blocks = []
        for i in range(header["num_blocks"]):
            row, col = divmod(i, cols)
            (dx1, dy1, dx2, dy2), offset = bitstream.unpack_block_header(data, offset)
            residual = {}
            for ch in range(3):
                pairs, offset = entropy.unpack_pairs(data, offset)
                residual[ch] = pairs
            blocks.append(
                {
                    "index": i,
                    "row": row,
                    "col": col,
                    "x": col * header["block_size"],
                    "y": row * header["block_size"],
                    "mv1": (dx1, dy1),
                    "mv2": (dx2, dy2),
                    "residual": residual,
                }
            )
        yield header, blocks


def _residual_summary(pairs_by_channel: dict) -> str:
    values = [val for pairs in pairs_by_channel.values() for _run, val in pairs if val != 0]
    return f"nonzero_levels={len(values)} sample={values[:8]}"


def print_segment(data: bytes, only_frame: Optional[int], only_block: Optional[int]) -> None:
    for header, blocks in parse_segment(data):
        if only_frame is not None and header["frame_id"] != only_frame:
            continue

        print("=" * 50)
        print(f"Frame ID       : {header['frame_id']}")
        print(f"Frame Type     : {TYPE_NAME[header['type_code']]}")
        ref_str = "-" if header["ref1"] is None else str(header["ref1"])
        if header["ref2"] is not None:
            ref_str += f", {header['ref2']}"
        print(f"Reference      : {ref_str}")
        print(f"Resolution     : {header['width']}x{header['height']}")
        print(f"QP             : {header['qp']}")
        print(f"Block size     : {header['block_size']}")
        print(f"Blocks         : {header['num_blocks']}")
        print()

        if only_block is not None:
            preview = [b for b in blocks if b["index"] == only_block]
        else:
            preview = blocks[:8]

        for b in preview:
            print(f"Block #{b['index']}")
            print(f"  position    : ({b['x']},{b['y']})")
            print(f"  MV (ref1)   : {b['mv1']}")
            if header["ref2"] is not None:
                print(f"  MV (ref2)   : {b['mv2']}")
            print(f"  residual    : {_residual_summary(b['residual'])}")
            print()

        if only_block is None and len(blocks) > 8:
            print(f"  ... ({len(blocks) - 8} more blocks not shown; use --block N to inspect one)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print an H256 segment .bin file")
    parser.add_argument("path")
    parser.add_argument("--frame", type=int, default=None, help="only show this frame_id")
    parser.add_argument("--block", type=int, default=None, help="show this block index in full detail")
    args = parser.parse_args()

    data = Path(args.path).read_bytes()
    print_segment(data, args.frame, args.block)


if __name__ == "__main__":
    main()
