"""Byte-histogram TopK mask for Ascend 950.

One kernel selects the kth-largest threshold with hardware byte histograms and
writes the masked row. Wide rows are split into 8192-element chunks; every chunk
contributes a 256-bin histogram, then the per-row histograms are merged and the
selected byte is refined. The same kernel handles every vocab, batch and dtype in
the supported range.
"""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.reduction.cumsum import _scan
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall, TopKMaskFwdInterface


@T.macro
def _histogram(x, state, hist, chunk, count, dtype: str, shift: int):
    with T.SimdVF():
        full = T.simd.pset(32)
        half_mask = T.simd.pset(16)
        low = T.simd.alloc_local(1, "uint16")
        high = T.simd.alloc_local(1, "uint16")
        low[0] = T.simd.vdup(T.uint16(0), "uint16")
        high[0] = T.simd.vdup(T.uint16(0), "uint16")
        prefix = T.simd.vld(state[0], dist="BRC_B32")
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(x[j * 64], dist="UNPK_B16"), "float32", part=0)
            )
            value = T.simd.vsel(
                T.simd.vdup(T.float32(0), "float32"),
                value,
                T.simd.vcmps(value, T.float32(0), full, op="eq"),
            )
            raw = T.reinterpret(value, "uint32x64")
            signed = T.reinterpret(value, "int32x64")
            negative = T.simd.vcmps(signed, T.int32(0), full, op="lt")
            flip = T.simd.vsel(
                T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"),
                T.simd.vdup(T.uint32(0x80000000), "uint32"),
                negative,
            )
            key = T.simd.vxor(raw, flip, full)
            absolute = T.simd.vand(raw, T.simd.vdup(T.uint32(0x7FFFFFFF), "uint32"), full)
            key = T.simd.vsel(
                T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"),
                key,
                T.simd.vcmps(absolute, T.uint32(0x7F800000), full, op="gt"),
            )
            if shift < 24:
                upper = T.simd.vand(
                    key,
                    T.simd.vdup(T.uint32((0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF), "uint32"),
                    full,
                )
                valid = T.simd.pand(valid, T.simd.vcmp(upper, prefix, full, op="eq"), full)
            byte = T.simd.vand(
                T.simd.vshrs(key, shift, full), T.simd.vdup(T.uint32(255), "uint32"), full
            )
            # A b32 predicate selects only byte zero of each reinterpreted word.
            T.simd.dhistv2(low[0], T.reinterpret(byte, "uint8x256"), valid, bin=0)
            T.simd.dhistv2(high[0], T.reinterpret(byte, "uint8x256"), valid, bin=1)
        T.simd.vsts(hist[chunk, 0], low[0], half_mask, extent=128)
        T.simd.vsts(hist[chunk, 128], high[0], half_mask, extent=128)


@functools.lru_cache(maxsize=128)
def _build(batch: int, vocab: int, dtype: str):
    width = 8192
    chunks = (vocab + width - 1) // width
    blocks = min(batch, 64)
    shifts = (24, 16) if dtype == "bfloat16" else (24, 16, 8, 0)

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        K: T.Tensor((batch,), "int32"),
        Y: T.Tensor((batch, vocab), dtype),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), dtype)
            hist = T.alloc_shared((chunks, 256), "uint16")
            counts = T.alloc_shared((1, 256), "float32")
            prefix = T.alloc_shared((1, 256), "float32")
            carry = T.alloc_shared((1, 8), "float32")
            state = T.alloc_shared((8,), "uint32")
            kval = T.alloc_shared((8,), "int32")
            for r in T.serial(T.ceildiv(batch, blocks)):
                row = r * blocks + bx
                if row < batch:
                    T.copy(K[row : row + 1], kval[:1])
                    state[0] = T.uint32(0)
                    state[1] = T.cast(kval[0], "uint32")
                    if kval[0] < vocab:
                        for shift in shifts:
                            for chunk in T.serial(chunks):
                                start = chunk * width
                                count = T.min(width, vocab - start)
                                T.copy(X[row, start : start + count], x[:count])
                                _histogram(x, state, hist, chunk, count, dtype, shift)
                            with T.SimtVF(threads=256):
                                totals = T.alloc_fragment((256,), "float32")
                                for i in T.Parallel(256):
                                    totals[i] = T.float32(0)
                                    for part in T.serial(chunks):
                                        totals[i] = totals[i] + T.cast(
                                            hist[part, 255 - i], "float32"
                                        )
                                    counts[0, i] = totals[i]
                            T.fill(carry, 0)
                            _scan(counts, prefix, carry, 1, 256, "float32")
                            with T.SimtVF(threads=256):
                                index = T.alloc_reducer((1,), "int32", op="min")
                                T.reducer_init(index)
                                for i in T.Parallel(256):
                                    if prefix[0, i] >= T.cast(state[1], "float32"):
                                        T.reducer_update(index[0], i)
                                found = T.alloc_fragment((1,), "int32")
                                T.finalize_reducer(index, found)
                                loc = T.min(found[0], 255)
                                state[0] = state[0] | (T.cast(255 - loc, "uint32") << shift)
                                state[1] = state[1] - T.cast(
                                    prefix[0, loc] - counts[0, loc], "uint32"
                                )
                    for chunk in T.serial(chunks):
                        start = chunk * width
                        count = T.min(width, vocab - start)
                        T.copy(X[row, start : start + count], x[:count])
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(width):
                                if i < count:
                                    value = T.cast(x[i], "float32")
                                    bits = T.reinterpret(
                                        T.if_then_else(value == 0, T.float32(0), value),
                                        "uint32",
                                    )
                                    key = T.if_then_else(
                                        T.isnan(value),
                                        T.uint32(0xFFFFFFFF),
                                        T.if_then_else(
                                            bits >> 31 != 0,
                                            ~bits,
                                            bits ^ T.uint32(0x80000000),
                                        ),
                                    )
                                    keep = kval[0] >= vocab or (
                                        key >= state[0]
                                        and not T.isnan(value)
                                        and state[0] < T.uint32(0xFFC00000)
                                    )
                                    y[i] = T.if_then_else(
                                        keep, x[i], T.cast(-T.infinity("float32"), dtype)
                                    )
                        T.copy(y[:count], Y[row, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendTopKMaskKernel(Kernel, TopKMaskFwdInterface):
    """General byte-histogram TopK mask for float16, bfloat16 and float32."""

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
