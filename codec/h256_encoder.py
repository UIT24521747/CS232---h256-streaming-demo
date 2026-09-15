"""H256 encoder: frame classification + the encode side of the pipeline.

Pipeline for one frame::

    Frame -> [predict] -> prediction -> [residual] -> residual
          -> [quantize] -> levels -> [pack: header + per-block RLE] -> bytes

The encoder also keeps a DPB ("decoded picture buffer"): after encoding
a frame it immediately *decodes its own output* (dequantize + reconstruct)
and stores that reconstructed frame, not the original, as the reference
for later frames. This "closed-loop" prediction is what real codecs do
too -- predicting from the original would let quantization error pile up
silently across a GOP, since the real decoder only ever has the lossy
reconstructed version to predict from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import bitstream, entropy, predictor
from . import quantizer as quant
from . import residual as residual_mod
from .frame import Frame, FrameType, block_grid, num_blocks, pad_to_block_size

BLOCK_SIZE = 16
SEARCH_RANGE = 8
DEFAULT_QP = 8

TYPE_CODE = {FrameType.I: 0, FrameType.P: 1, FrameType.B: 2}
CODE_TYPE = {v: k for k, v in TYPE_CODE.items()}


def classify_frames(count: int, start_id: int = 0) -> List[dict]:
    """Assign I/P/B types with a small fixed GOP pattern.

    Local frame 0 is I. Odd local frames are P, each referencing the
    previous anchor (I or P). Even local frames (except 0) are B,
    referencing the anchor before *and* the anchor after them -- which
    means the anchor after must be compressed first (see encode_order).
    The final frame of a chunk falls back to P if it would otherwise be
    a B frame with no future anchor to reference.

    `start_id` offsets every id/ref field so this can be called once
    per independent segment (each segment/GOP starts fresh with an I
    frame) while still reporting the true, global frame number.
    """
    frames: List[dict] = []
    anchors = [0]
    frames.append({"id": 0, "type": FrameType.I, "ref1": None, "ref2": None})

    for local_id in range(1, count):
        if local_id % 2 == 1:
            ref1 = anchors[-1]
            anchors.append(local_id)
            frames.append({"id": local_id, "type": FrameType.P, "ref1": ref1, "ref2": None})
        else:
            has_future_anchor = local_id + 1 < count
            if has_future_anchor:
                frames.append(
                    {"id": local_id, "type": FrameType.B, "ref1": anchors[-1], "ref2": local_id + 1}
                )
            else:
                ref1 = anchors[-1]
                anchors.append(local_id)
                frames.append({"id": local_id, "type": FrameType.P, "ref1": ref1, "ref2": None})

    for meta in frames:
        meta["id"] += start_id
        if meta["ref1"] is not None:
            meta["ref1"] += start_id
        if meta["ref2"] is not None:
            meta["ref2"] += start_id
    return frames


def encode_order(classified: List[dict]) -> List[int]:
    """Anchors (I/P) first in display order, then B frames.

    By the time any B frame is encoded, both of the anchors it points
    to already exist in the DPB -- real codecs call this "frame
    reordering" and it's why decode order and display order differ.
    """
    anchors = [f["id"] for f in classified if f["type"] in (FrameType.I, FrameType.P)]
    bs = [f["id"] for f in classified if f["type"] == FrameType.B]
    return anchors + bs


@dataclass
class EncodeTrace:
    frame_id: int
    frame_type: FrameType
    ref1: Optional[int]
    ref2: Optional[int]
    prediction: np.ndarray
    residual: np.ndarray
    quantized: np.ndarray
    qp: int
    raw_size: int
    encoded_size: int
    motion_vectors: dict = field(default_factory=dict)


class H256Encoder:
    def __init__(self, block_size: int = BLOCK_SIZE, search_range: int = SEARCH_RANGE, qp: int = DEFAULT_QP):
        self.block_size = block_size
        self.search_range = search_range
        self.qp = qp

    def encode_frame(self, frame: Frame, dpb: Dict[int, np.ndarray]) -> tuple[bytes, EncodeTrace]:
        data = pad_to_block_size(frame.data, self.block_size)
        h, w = data.shape[:2]
        bs = self.block_size

        if frame.type is FrameType.I:
            prediction = self._predict_intra_frame(data, bs)
            mv_map: dict = {}
        elif frame.type is FrameType.P:
            ref = dpb[frame.ref1]
            prediction, mv_map = predictor.predict_inter(data, ref, bs, self.search_range)
        else:
            ref_prev, ref_next = dpb[frame.ref1], dpb[frame.ref2]
            prediction, mv_prev, mv_next = predictor.predict_bidirectional(
                data, ref_prev, ref_next, bs, self.search_range
            )
            mv_map = {"prev": mv_prev, "next": mv_next}

        res = residual_mod.compute_residual(data, prediction)
        levels = quant.quantize(res, self.qp)

        payload = self._pack(frame, w, h, levels, mv_map)

        dequant = quant.dequantize(levels, self.qp)
        reconstructed = residual_mod.reconstruct(prediction, dequant)
        dpb[frame.frame_id] = reconstructed  # closed-loop: predict from *this*, not the original

        trace = EncodeTrace(
            frame_id=frame.frame_id,
            frame_type=frame.type,
            ref1=frame.ref1,
            ref2=frame.ref2,
            prediction=np.clip(prediction, 0, 255).astype(np.uint8)[: frame.height, : frame.width],
            residual=res[: frame.height, : frame.width],
            quantized=levels[: frame.height, : frame.width],
            qp=self.qp,
            raw_size=data.size,
            encoded_size=len(payload),
            motion_vectors=mv_map,
        )
        return payload, trace

    def _predict_intra_frame(self, data: np.ndarray, bs: int) -> np.ndarray:
        """Run DC intra prediction block-by-block in raster order.

        Each block needs the *reconstructed* left/top neighbor, so this
        does its own tiny predict -> residual -> quantize -> dequantize
        -> reconstruct round trip per block, purely to build the causal
        neighbor buffer. The real (whole-frame) residual/quantize used
        for the actual bitstream happens once, back in encode_frame,
        using the `prediction` this returns.
        """
        h, w = data.shape[:2]
        prediction = np.zeros_like(data, dtype=np.int16)
        recon_so_far = np.zeros_like(data, dtype=np.uint8)
        for row, col, y, x in block_grid(h, w, bs):
            pred_block = predictor.predict_intra_block(recon_so_far, row, col, y, x, bs)
            prediction[y : y + bs, x : x + bs] = pred_block

            cur_block = data[y : y + bs, x : x + bs]
            res_block = residual_mod.compute_residual(cur_block, pred_block)
            levels_block = quant.quantize(res_block, self.qp)
            deq_block = quant.dequantize(levels_block, self.qp)
            recon_so_far[y : y + bs, x : x + bs] = residual_mod.reconstruct(pred_block, deq_block)
        return prediction

    def _pack(self, frame: Frame, width: int, height: int, levels: np.ndarray, mv_map: dict) -> bytes:
        bs = self.block_size
        n = num_blocks(height, width, bs)
        parts = [
            bitstream.pack_header(
                frame.frame_id, TYPE_CODE[frame.type], frame.ref1, frame.ref2, width, height, self.qp, bs, n
            )
        ]
        for row, col, y, x in block_grid(height, width, bs):
            if frame.type is FrameType.I:
                dx1 = dy1 = dx2 = dy2 = 0
            elif frame.type is FrameType.P:
                dx1, dy1, _mse = mv_map[(row, col)]
                dx2 = dy2 = 0
            else:
                dx1, dy1, _ = mv_map["prev"][(row, col)]
                dx2, dy2, _ = mv_map["next"][(row, col)]
            parts.append(bitstream.pack_block_header(dx1, dy1, dx2, dy2))

            block_levels = levels[y : y + bs, x : x + bs]
            for ch in range(3):
                pairs = entropy.rle_encode(block_levels[:, :, ch])
                parts.append(entropy.pack_pairs(pairs))
        return b"".join(parts)
