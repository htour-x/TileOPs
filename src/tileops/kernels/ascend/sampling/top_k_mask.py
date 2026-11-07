"""Streaming radix TopK mask for Ascend 950.

One kernel handles every vocabulary and batch size. It selects the kth-largest
threshold with a bitwise rank scan over the row, then writes the masked row in a
second pass. No per-case kernel variants are generated.
"""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.sampling.radix_reduce import _rank_tile
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall, TopKMaskFwdInterface


def _key(value, shift):
    value = T.cast(value, "float32")
    bits = T.reinterpret(T.if_then_else(value == 0, T.float32(0), value), "uint32")
    return (
        T.if_then_else(
            T.isnan(value),
            T.uint32(0xFFFFFFFF),
            T.if_then_else(bits >> 31 != 0, ~bits, bits ^ T.uint32(0x80000000)),
        )
        >> shift
    )


@functools.lru_cache(maxsize=256)
def _build(batch: int, vocab: int, dtype: str):
    # A row tile plus the SIMD scratch must fit in UB for every supported dtype.
    width = min(8192, max(64, tilelang.next_power_of_2(vocab)))
    chunks = (vocab + width - 1) // width
    blocks = min(batch, 64)
    shift = 16 if dtype == "bfloat16" else 13 if dtype == "float16" else 0
    bits = 32 - shift
    max_key = 0xFFFFFFFF >> shift

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        K: T.Tensor((batch,), "int32"),
        Y: T.Tensor((batch, vocab), dtype),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), dtype)
            kval = T.alloc_shared((8,), "int32")
            count_out = T.alloc_shared((8,), "int32")
            for r in T.serial(T.ceildiv(batch, blocks)):
                row = r * blocks + bx
                if row < batch:
                    T.copy(K[row : row + 1], kval[:1])
                    cutoff = T.alloc_var("uint32", init=0)
                    if kval[0] < vocab:
                        for step in T.serial(bits):
                            pivot = cutoff | (T.uint32(1) << (bits - 1 - step))
                            rank = T.alloc_var("int32", init=0)
                            for chunk in T.serial(chunks):
                                start = chunk * width
                                count = T.min(width, vocab - start)
                                T.copy(X[row, start : start + count], x[:count])
                                _rank_tile(x, count_out, count, pivot, shift, dtype)
                                rank = rank + count_out[0]
                            if rank >= kval[0]:
                                cutoff = pivot
                    for chunk in T.serial(chunks):
                        start = chunk * width
                        count = T.min(width, vocab - start)
                        T.copy(X[row, start : start + count], x[:count])
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(width):
                                if i < count:
                                    value = T.cast(x[i], "float32")
                                    keep = _key(value, shift) >= cutoff
                                    keep = keep and (
                                        kval[0] >= vocab
                                        or (not T.isnan(value) and cutoff != T.uint32(max_key))
                                    )
                                    y[i] = T.if_then_else(
                                        keep,
                                        x[i],
                                        T.cast(-T.infinity("float32"), dtype),
                                    )
                        T.copy(y[:count], Y[row, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendTopKMaskKernel(Kernel, TopKMaskFwdInterface):
    """General streaming radix TopK mask for float16, bfloat16 and float32."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: SamplingCall):
        return call, lambda: cls(call)

    def __init__(self, call: SamplingCall):
        super().__init__()
        self.call = call
        self.kernel = (
            _build(call.batch, call.vocab, str(call.dtype).split(".")[-1])
            if call.batch and call.vocab
            else None
        )

    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._require_devices(logits=logits, k=k)
        if logits.shape != (self.call.batch, self.call.vocab):
            raise ValueError("logits shape differs from the compiled call")
        if logits.dtype != self.call.dtype or k.dtype != torch.int32:
            raise TypeError("input dtype differs from the compiled call")
        if self.kernel is None:
            return torch.empty_like(logits)
        return self.kernel(logits.contiguous(), k.contiguous())


register_kernel_type("TopKMaskFwdOp", "ascend", AscendTopKMaskKernel)
