"""H256 decoder: the exact mirror image of h256_encoder.py.

    bytes -> [unpack header + per-block RLE] -> quantized levels
          -> [dequantize] -> residual -> [motion-compensate / predict]
          -> [reconstruct] -> pixel frame

Because motion vectors are stored explicitly in the bitstream, the
decoder never needs to search for a match -- it just reads (dx, dy)
and copies pixels from the reference frame at that offset. That's the
entire point of storing motion vectors: search is the encoder's
(expensive) problem, applying one is nearly free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import bitstream, entropy, predictor
from . import quantizer as quant
from . import residual as residual_mod
from .frame import Frame, FrameType, block_grid

from .h256_encoder import CODE_TYPE


@dataclass
class DecodeTrace:
    header: dict
    frame_type: FrameType
    mv_log: List[dict]
    prediction: np.ndarray
    reconstructed: np.ndarray
    encoded_size: int = 0


class H256Decoder:
    def decode_frame(self, payload: bytes, dpb: Dict[int, np.ndarray]) -> tuple[Frame, DecodeTrace]:
        header, offset = bitstream.unpack_header(payload)
        bs = header["block_size"]
        w, h = header["width"], header["height"]
        cols = w // bs
        frame_type = CODE_TYPE[header["type_code"]]

        levels = np.zeros((h, w, 3), dtype=np.int16)
        mv_log: List[dict] = []
        for i in range(header["num_blocks"]):
            row, col = divmod(i, cols)
            y, x = row * bs, col * bs
            (dx1, dy1, dx2, dy2), offset = bitstream.unpack_block_header(payload, offset)
            for ch in range(3):
                pairs, offset = entropy.unpack_pairs(payload, offset)
                flat = entropy.rle_decode(pairs, bs * bs)
                levels[y : y + bs, x : x + bs, ch] = flat.reshape(bs, bs)
            mv_log.append({"row": row, "col": col, "x": x, "y": y, "mv1": (dx1, dy1), "mv2": (dx2, dy2)})

        dequant = quant.dequantize(levels, header["qp"])

        if frame_type is FrameType.I:
            prediction, reconstructed = self._reconstruct_intra(dequant, h, w, bs)
        elif frame_type is FrameType.P:
            ref = dpb[header["ref1"]]
            prediction = self._motion_compensate(ref, mv_log, bs, "mv1")
            reconstructed = residual_mod.reconstruct(prediction, dequant)
        else:
            ref_prev, ref_next = dpb[header["ref1"]], dpb[header["ref2"]]
            pred_prev = self._motion_compensate(ref_prev, mv_log, bs, "mv1")
            pred_next = self._motion_compensate(ref_next, mv_log, bs, "mv2")
            prediction = ((pred_prev.astype(np.int32) + pred_next.astype(np.int32)) // 2).astype(np.int16)
            reconstructed = residual_mod.reconstruct(prediction, dequant)

        dpb[header["frame_id"]] = reconstructed

        frame = Frame(
            frame_id=header["frame_id"],
            type=frame_type,
            data=reconstructed,
            ref1=header["ref1"],
            ref2=header["ref2"],
        )
        trace = DecodeTrace(
            header=header,
            frame_type=frame_type,
            mv_log=mv_log,
            prediction=np.clip(prediction, 0, 255).astype(np.uint8),
            reconstructed=reconstructed,
            encoded_size=len(payload),
        )
        return frame, trace

    @staticmethod
    def _reconstruct_intra(dequant: np.ndarray, h: int, w: int, bs: int):
        """Mirror of H256Encoder._predict_intra_frame: walk blocks in raster
        order, predicting each one from the reconstruction-so-far so the
        causal left/top DC prediction lines up exactly with what the
        encoder used."""
        prediction = np.zeros((h, w, 3), dtype=np.int16)
        reconstructed = np.zeros((h, w, 3), dtype=np.uint8)
        for row, col, y, x in block_grid(h, w, bs):
            pred_block = predictor.predict_intra_block(reconstructed, row, col, y, x, bs)
            prediction[y : y + bs, x : x + bs] = pred_block
            deq_block = dequant[y : y + bs, x : x + bs]
            reconstructed[y : y + bs, x : x + bs] = residual_mod.reconstruct(pred_block, deq_block)
        return prediction, reconstructed

    @staticmethod
    def _motion_compensate(ref: np.ndarray, mv_log: List[dict], bs: int, key: str) -> np.ndarray:
        h, w = ref.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.int16)
        for entry in mv_log:
            dx, dy = entry[key]
            y, x = entry["y"], entry["x"]
            ry, rx = y + dy, x + dx
            out[y : y + bs, x : x + bs] = ref[ry : ry + bs, rx : rx + bs]
        return out


def decode_segment(data: bytes, dpb: Optional[Dict[int, np.ndarray]] = None):
    """Decode every frame in one segment file. Returns (frames, traces, dpb)."""
    if dpb is None:
        dpb = {}
    n, offset = bitstream.unpack_segment_count(data)
    decoder = H256Decoder()
    frames, traces = [], []
    for _ in range(n):
        end = frame_byte_length(data, offset)
        frame, trace = decoder.decode_frame(data[offset:end], dpb)
        frames.append(frame)
        traces.append(trace)
        offset = end
    return frames, traces, dpb


def frame_byte_length(data: bytes, offset: int) -> int:
    """Walk one frame record purely to find where it ends (no side effects)."""
    header, cursor = bitstream.unpack_header(data, offset)
    for _ in range(header["num_blocks"]):
        _, cursor = bitstream.unpack_block_header(data, cursor)
        for _ in range(3):
            _, cursor = entropy.unpack_pairs(data, cursor)
    return cursor
