"""H256 -- a small, from-scratch educational video codec.

Not H.265. Not any real codec. The name is a deliberate wink: same
high-level idea (I/P/B frame types, prediction, residual, quantization,
entropy coding, a bitstream you can parse), none of the real-world
complexity (no DCT, no CABAC, no deblocking filter, no RDO). Every
module in this package is short enough to read start to finish.
"""
