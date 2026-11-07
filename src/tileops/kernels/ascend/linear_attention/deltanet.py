"""Ascend DeltaNet forward with chunked prefill and single-token decode."""

from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch

import tileops.backend
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import DeltaNetCall, DeltaNetFwdInterface

_CHUNK = 32
_CUBE_BLOCKS = 32
_THREADS = 128
_SOLVE_BLOCKS = 64
_SOLVE_THREADS = 128
_SOLVE_PANEL = 8
_IDENTITY_ROWS = 16
_STATE_VALUE_TILE = 64
_DENSE_BULK_STATE_VALUE_TILE = 128
_DECODE_BLOCKS = 64
_DECODE_THREADS = 128
_K128_HF32 = "nearest_even"


def _build_dense_decode(
    batch, heads, dim_k, dim_v, dtype, scale, normalize, has_initial, max_blocks, simt_threads
):
    jobs = batch * heads
    blocks = min(max_blocks, jobs)
    rounds = (jobs + blocks - 1) // blocks
    threads = min(simt_threads, dim_k)
    value_vectors = dim_v // 64

    @T.prim_func
    def main(
        Q: T.Tensor((batch, 1, heads, dim_k), dtype),
        Keys: T.Tensor((batch, 1, heads, dim_k), dtype),
        Values: T.Tensor((batch, 1, heads, dim_v), dtype),
        Beta: T.Tensor((batch, 1, heads), dtype),
        Initial: T.Tensor((batch, heads, dim_k, dim_v), "float32"),
        Output: T.Tensor((batch, 1, heads, dim_v), dtype),
        Final: T.Tensor((batch, heads, dim_k, dim_v), "float32"),
    ):
        with T.Kernel(blocks) as block:
            state_kv = T.alloc_shared((dim_k, dim_v), "float32")
            q_raw = T.alloc_shared((dim_k,), dtype)
            k_raw = T.alloc_shared((dim_k,), dtype)
            v_raw = T.alloc_shared((dim_v,), dtype)
            beta_raw = T.alloc_shared((16,), dtype)
            q_f32 = T.alloc_shared((dim_k,), "float32")
            k_f32 = T.alloc_shared((dim_k,), "float32")
            v_f32 = T.alloc_shared((dim_v,), "float32")
            beta_f32 = T.alloc_shared((8,), "float32")
            output_f32 = T.alloc_shared((dim_v,), "float32")
            output_ub = T.alloc_shared((dim_v,), dtype)
            T.annotate_buffer_versions(
                {
                    state_kv: 1,
                    q_raw: 1,
                    k_raw: 1,
                    v_raw: 1,
                    beta_raw: 1,
                    q_f32: 1,
                    k_f32: 1,
                    v_f32: 1,
                    beta_f32: 1,
                    output_f32: 1,
                    output_ub: 1,
                }
            )
            for work_round in T.serial(rounds):
                job = work_round * blocks + block
                if job < jobs:
                    sequence = job // heads
                    head = job % heads
                    T.copy(Q[sequence, 0, head, 0:dim_k], q_raw)
                    T.copy(Keys[sequence, 0, head, 0:dim_k], k_raw)
                    T.copy(Values[sequence, 0, head, 0:dim_v], v_raw)
                    T.copy(Beta[sequence, 0, head : head + 1], beta_raw[0:1])
                    if has_initial:
                        T.copy(Initial[sequence, head, 0:dim_k, 0:dim_v], state_kv)
                    with T.SimtVF(threads=threads):
                        q_frag = T.alloc_fragment((dim_k,), "float32")
                        k_frag = T.alloc_fragment((dim_k,), "float32")
                        for i in T.Parallel(dim_k, coalesced_width=T.int32(1)):
                            q_frag[i] = T.cast(q_raw[i], "float32")
                            k_frag[i] = T.cast(k_raw[i], "float32")
                        if normalize:
                            q_sum = T.alloc_reducer((1,), "float32", op="sum")
                            k_sum = T.alloc_reducer((1,), "float32", op="sum")
                            q_inv = T.alloc_fragment((1,), "float32")
                            k_inv = T.alloc_fragment((1,), "float32")
                            T.reducer_init(q_sum)
                            T.reducer_init(k_sum)
                            for i in T.Parallel(dim_k, coalesced_width=T.int32(1)):
                                T.reducer_update(q_sum[0], q_frag[i] * q_frag[i])
                                T.reducer_update(k_sum[0], k_frag[i] * k_frag[i])
                            T.finalize_reducer(q_sum, q_inv)
                            T.finalize_reducer(k_sum, k_inv)
                            for r in T.Parallel(1):
                                q_inv[r] = T.rsqrt(q_inv[r] + T.float32(1e-6))
                                k_inv[r] = T.rsqrt(k_inv[r] + T.float32(1e-6))
                            for i in T.Parallel(dim_k, coalesced_width=T.int32(1)):
                                q_frag[i] = T.cast(T.cast(q_frag[i] * q_inv[0], dtype), "float32")
                                k_frag[i] = T.cast(T.cast(k_frag[i] * k_inv[0], dtype), "float32")
                        for i in T.Parallel(dim_k, coalesced_width=T.int32(1)):
                            q_f32[i] = q_frag[i] * T.float32(scale)
                            k_f32[i] = k_frag[i]
                        for j in T.Parallel(dim_v):
                            v_f32[j] = T.cast(v_raw[j], "float32")
                        for r in T.Parallel(1):
                            beta_f32[r] = T.cast(beta_raw[r], "float32")
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        prediction = T.simd.alloc_local(4 * value_vectors, "float32")
                        result = T.simd.alloc_local(4 * value_vectors, "float32")
                        delta = T.simd.alloc_local(value_vectors, "float32")
                        for r in T.Unroll(4, explicit=True):
                            for c in T.Unroll(value_vectors, explicit=True):
                                result[r * value_vectors + c] = T.simd.vdup(
                                    T.float32(0), "float32", mask
                                )
                                if has_initial:
                                    prediction[r * value_vectors + c] = T.simd.vdup(
                                        T.float32(0), "float32", mask
                                    )
                        if has_initial:
                            for g in range(dim_k // 4):
                                for r in T.Unroll(4, explicit=True):
                                    key_reg = T.simd.vld(k_f32[g * 4 + r], dist="BRC_B32")
                                    for c in T.Unroll(value_vectors, explicit=True):
                                        previous = T.simd.vld(state_kv[g * 4 + r, c * 64])
                                        product = T.simd.vmul(previous, key_reg, mask)
                                        prediction[r * value_vectors + c] = T.simd.vadd(
                                            prediction[r * value_vectors + c], product, mask
                                        )
                        beta_reg = T.simd.vld(beta_f32[0], dist="BRC_B32")
                        for c in T.Unroll(value_vectors, explicit=True):
                            value_reg = T.simd.vld(v_f32[c * 64])
                            if has_initial:
                                prediction01 = T.simd.vadd(
                                    prediction[c], prediction[value_vectors + c], mask
                                )
                                prediction23 = T.simd.vadd(
                                    prediction[2 * value_vectors + c],
                                    prediction[3 * value_vectors + c],
                                    mask,
                                )
                                prediction_sum = T.simd.vadd(prediction01, prediction23, mask)
                                residual = T.simd.vsub(value_reg, prediction_sum, mask)
                                delta[c] = T.simd.vmul(residual, beta_reg, mask)
                            else:
                                delta[c] = T.simd.vmul(value_reg, beta_reg, mask)
                        for g in range(dim_k // 4):
                            for r in T.Unroll(4, explicit=True):
                                update_key = T.simd.vld(k_f32[g * 4 + r], dist="BRC_B32")
                                query_reg = T.simd.vld(q_f32[g * 4 + r], dist="BRC_B32")
                                for c in T.Unroll(value_vectors, explicit=True):
                                    update_reg = T.simd.vmul(update_key, delta[c], mask)
                                    if has_initial:
                                        previous_state = T.simd.vld(state_kv[g * 4 + r, c * 64])
                                        next_state = T.simd.vadd(previous_state, update_reg, mask)
                                    else:
                                        next_state = update_reg
                                    T.simd.vsts(state_kv[g * 4 + r, c * 64], next_state, mask)
                                    output_product = T.simd.vmul(next_state, query_reg, mask)
                                    result[r * value_vectors + c] = T.simd.vadd(
                                        result[r * value_vectors + c], output_product, mask
                                    )
                        for c in T.Unroll(value_vectors, explicit=True):
                            result01 = T.simd.vadd(result[c], result[value_vectors + c], mask)
                            result23 = T.simd.vadd(
                                result[2 * value_vectors + c], result[3 * value_vectors + c], mask
                            )
                            result_sum = T.simd.vadd(result01, result23, mask)
                            T.simd.vsts(output_f32[c * 64], result_sum, mask)
                    with T.SimtVF(threads=threads):
                        for j in T.Parallel(dim_v):
                            output_ub[j] = T.cast(output_f32[j], dtype)
                    T.copy(state_kv, Final[sequence, head, 0:dim_k, 0:dim_v])
                    T.copy(output_ub, Output[sequence, 0, head, 0:dim_v])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6])


def _build_pack_grams(
    batch,
    seq_len,
    heads,
    dim_k,
    dim_v,
    num_sequences,
    dtype,
    scale,
    normalize,
    packed,
    chunk,
    slots,
    max_blocks,
    threads,
    *,
    hf32,
):
    dense_chunks = (seq_len + chunk - 1) // chunk
    jobs = slots * heads
    blocks = min(max_blocks, jobs)
    rounds = (jobs + blocks - 1) // blocks
    offset_storage = ((num_sequences + 4) // 4) * 4
    meta_storage = ((num_sequences + 8) // 8) * 8
    features = dim_k + dim_v
    rows = chunk // 2
    cast_threads = min(threads, dim_k)
    key_vectors = dim_k // 64
    value_vectors = dim_v // 64

    @T.prim_func
    def pack_grams(
        Q: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        Keys: T.Tensor((batch, seq_len, heads, dim_k), dtype),
        Values: T.Tensor((batch, seq_len, heads, dim_v), dtype),
        Beta: T.Tensor((batch, seq_len, heads), dtype),
        Offsets: T.Tensor((num_sequences + 1,), "int64"),
        QP: T.Tensor((slots, heads, chunk, dim_k), "float32"),
        KV: T.Tensor((slots, heads, chunk, features), "float32"),
        BP: T.Tensor((slots, heads, chunk), "float32"),
        Meta: T.Tensor((num_sequences + 1,), "int32"),
        Gram: T.Tensor((slots, heads, chunk, chunk), "float32"),
        QK: T.Tensor((slots, heads, chunk, chunk), "float32"),
    ):
        with T.MixedKernel(blocks) as (block, sid):
            T.set_hf32_mode(hf32)
            q_l1 = T.alloc_l1((chunk, dim_k), "float32")
            k_l1 = T.alloc_l1((chunk, dim_k), "float32")
            result = T.alloc_l0c((chunk, chunk), "float32")
            offsets_ub = T.alloc_shared((offset_storage,), "int64")
            meta_ub = T.alloc_shared((meta_storage,), "int32")
            q_raw = T.alloc_shared((rows, dim_k), dtype)
            k_raw = T.alloc_shared((rows, dim_k), dtype)
            v_raw = T.alloc_shared((rows, dim_v), dtype)
            b_raw = T.alloc_shared((rows, heads), dtype)
            q_ub = T.alloc_shared((rows, dim_k), "float32")
            k_ub = T.alloc_shared((rows, dim_k), "float32")
            v_ub = T.alloc_shared((rows, dim_v), "float32")
            b_ub = T.alloc_shared((rows,), "float32")
            versions = {
                offsets_ub: 1,
                meta_ub: 1,
                q_raw: 1,
                k_raw: 1,
                v_raw: 1,
                b_raw: 1,
                q_ub: 1,
                k_ub: 1,
                v_ub: 1,
                b_ub: 1,
                q_l1: 1,
                k_l1: 1,
                result: 1,
            }
            T.annotate_buffer_versions(versions)
            if packed:
                T.copy(Offsets[0 : num_sequences + 1], offsets_ub[0 : num_sequences + 1])
                with T.SimtVF(threads=32):
                    prefix = T.alloc_fragment((num_sequences + 1,), "int32")
                    for r in T.Parallel(num_sequences + 1):
                        prefix[r] = 0
                    for s in T.serial(num_sequences):
                        for r in T.Parallel(num_sequences + 1):
                            if r > s:
                                prefix[r] = (
                                    prefix[r]
                                    + (
                                        T.cast(offsets_ub[s + 1] - offsets_ub[s], "int32")
                                        + chunk
                                        - 1
                                    )
                                    // chunk
                                )
                    T.copy(prefix, meta_ub[0 : num_sequences + 1])
            else:
                with T.SimtVF(threads=32):
                    for r in T.Parallel(num_sequences + 1):
                        meta_ub[r] = r * dense_chunks
            if block == 0 and sid == 0:
                T.copy(meta_ub[0 : num_sequences + 1], Meta[0 : num_sequences + 1])
            for work_round in T.serial(rounds):
                job = work_round * blocks + block
                if job < jobs:
                    slot = job // heads
                    head = job % heads
                    token_begin = T.alloc_var("int32")
                    valid = T.alloc_var("int32")
                    if packed:
                        batch_index = T.int32(0)
                        token_begin = 0
                        valid = 0
                        for sequence in T.serial(num_sequences):
                            if slot >= meta_ub[sequence] and slot < meta_ub[sequence + 1]:
                                token_begin = (
                                    T.cast(offsets_ub[sequence], "int32")
                                    + (slot - meta_ub[sequence]) * chunk
                                )
                                valid = T.min(
                                    chunk, T.cast(offsets_ub[sequence + 1], "int32") - token_begin
                                )
                    else:
                        batch_index = slot // dense_chunks
                        token_begin = (slot % dense_chunks) * chunk
                        valid = T.min(chunk, seq_len - token_begin)
                    row_base = sid * rows
                    local_begin = token_begin + row_base
                    local_valid = T.max(0, T.min(rows, valid - row_base))
                    if local_valid > 0:
                        T.copy(
                            Q[batch_index, local_begin : local_begin + local_valid, head, 0:dim_k],
                            q_raw[0:local_valid, 0:dim_k],
                        )
                        T.copy(
                            Keys[
                                batch_index, local_begin : local_begin + local_valid, head, 0:dim_k
                            ],
                            k_raw[0:local_valid, 0:dim_k],
                        )
                        T.copy(
                            Values[
                                batch_index, local_begin : local_begin + local_valid, head, 0:dim_v
                            ],
                            v_raw[0:local_valid, 0:dim_v],
                        )
                        T.copy(
                            Beta[batch_index, local_begin : local_begin + local_valid, 0:heads],
                            b_raw[0:local_valid, 0:heads],
                        )
                    with T.SimtVF(threads=cast_threads):
                        for t in T.Parallel(rows):
                            b_ub[t] = T.if_then_else(
                                t < local_valid, T.cast(b_raw[t, head], "float32"), T.float32(0)
                            )
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        q_vec = T.simd.alloc_local(key_vectors, "float32")
                        k_vec = T.simd.alloc_local(key_vectors, "float32")
                        sums = T.simd.alloc_local(2, "float32")
                        if normalize:
                            epsilon = T.simd.vdup(T.float32(1e-6), "float32", mask)
                            one = T.simd.vdup(T.float32(1), "float32", mask)
                        for t in range(local_valid):
                            for c in T.Unroll(key_vectors, explicit=True):
                                q_input = T.simd.vld(q_raw[t, c * 64], dist="UNPK_B16")
                                k_input = T.simd.vld(k_raw[t, c * 64], dist="UNPK_B16")
                                q_vec[c] = T.simd.vcvt(q_input, "float32", part=0)
                                k_vec[c] = T.simd.vcvt(k_input, "float32", part=0)
                            if normalize:
                                sums[0] = T.simd.vdup(T.float32(0), "float32", mask)
                                sums[1] = T.simd.vdup(T.float32(0), "float32", mask)
                                for c in T.Unroll(key_vectors, explicit=True):
                                    q_squared = T.simd.vmul(q_vec[c], q_vec[c], mask)
                                    k_squared = T.simd.vmul(k_vec[c], k_vec[c], mask)
                                    sums[0] = T.simd.vadd(sums[0], q_squared, mask)
                                    sums[1] = T.simd.vadd(sums[1], k_squared, mask)
                                q_sum = T.simd.vcadd(sums[0], mask)
                                k_sum = T.simd.vcadd(sums[1], mask)
                                q_broadcast = T.simd.vdupv(q_sum, mask)
                                k_broadcast = T.simd.vdupv(k_sum, mask)
                                q_eps = T.simd.vadd(q_broadcast, epsilon, mask)
                                k_eps = T.simd.vadd(k_broadcast, epsilon, mask)
                                q_root = T.simd.vsqrt(q_eps, mask)
                                k_root = T.simd.vsqrt(k_eps, mask)
                                q_inverse = T.simd.vdiv(one, q_root, mask, precision="exact")
                                k_inverse = T.simd.vdiv(one, k_root, mask, precision="exact")
                                for c in T.Unroll(key_vectors, explicit=True):
                                    q_normalized = T.simd.vmul(q_vec[c], q_inverse, mask)
                                    k_normalized = T.simd.vmul(k_vec[c], k_inverse, mask)
                                    q_rounded = T.simd.vcvt(
                                        q_normalized, dtype, round="ROUND_R", part=0
                                    )
                                    k_rounded = T.simd.vcvt(
                                        k_normalized, dtype, round="ROUND_R", part=0
                                    )
                                    q_vec[c] = T.simd.vcvt(q_rounded, "float32", part=0)
                                    k_vec[c] = T.simd.vcvt(k_rounded, "float32", part=0)
                            for c in T.Unroll(key_vectors, explicit=True):
                                q_scaled = T.simd.vmuls(q_vec[c], T.float32(scale), mask)
                                T.simd.vsts(q_ub[t, c * 64], q_scaled, mask)
                                T.simd.vsts(k_ub[t, c * 64], k_vec[c], mask)
                            for c in T.Unroll(value_vectors, explicit=True):
                                value_input = T.simd.vld(v_raw[t, c * 64], dist="UNPK_B16")
                                value_reg = T.simd.vcvt(value_input, "float32", part=0)
                                T.simd.vsts(v_ub[t, c * 64], value_reg, mask)
                        zero = T.simd.vdup(T.float32(0), "float32", mask)
                        for t in range(local_valid, rows):
                            for c in T.Unroll(key_vectors, explicit=True):
                                T.simd.vsts(q_ub[t, c * 64], zero, mask)
                                T.simd.vsts(k_ub[t, c * 64], zero, mask)
                            for c in T.Unroll(value_vectors, explicit=True):
                                T.simd.vsts(v_ub[t, c * 64], zero, mask)
                    T.dual_copy(k_ub, k_l1)
                    T.gemm(k_l1, k_l1, result, transpose_B=True, clear_accum=True)
                    T.copy(result, Gram[slot, head, 0:chunk, 0:chunk])
                    T.dual_copy(q_ub, q_l1)
                    T.gemm(q_l1, k_l1, result, transpose_B=True, clear_accum=True)
                    T.copy(result, QK[slot, head, 0:chunk, 0:chunk])
                    T.copy(q_ub, QP[slot, head, row_base : row_base + rows, 0:dim_k])
                    T.copy(k_ub, KV[slot, head, row_base : row_base + rows, 0:dim_k])
                    T.copy(v_ub, KV[slot, head, row_base : row_base + rows, dim_k:features])
                    T.copy(b_ub, BP[slot, head, row_base : row_base + rows])

    return tilelang.compile(pack_grams, target="ascend", out_idx=[5, 6, 7, 8, 9, 10])


@T.macro
def _forward_substitution(rhs, lower, beta, chunk, panel, vectors, weight_rhs):
    # The lower coefficients are negated for fused subtraction into each row.
    with T.SimdVF():
        mask = T.simd.pset(32)
        rows = T.simd.alloc_local(panel * vectors, "float32")
        pivot = T.simd.alloc_local(vectors, "float32")
        for p in range(chunk // panel):
            for i in T.Unroll(panel, explicit=True):
                if weight_rhs:
                    # Only a new row needs beta. Earlier pivots are already solved.
                    beta_reg = T.simd.vld(beta[p * panel + i], dist="BRC_B32")
                for c in T.Unroll(vectors, explicit=True):
                    row = T.simd.vld(rhs[p * panel + i, c * 64])
                    if weight_rhs:
                        row = T.simd.vmul(row, beta_reg, mask)
                    rows[i * vectors + c] = row
            for j in range(p * panel):
                for c in T.Unroll(vectors, explicit=True):
                    pivot[c] = T.simd.vld(rhs[j, c * 64])
                for i in T.Unroll(panel, explicit=True):
                    coefficient = T.simd.vld(lower[p * panel + i, j], dist="BRC_B32")
                    for c in T.Unroll(vectors, explicit=True):
                        T.simd.vmula(rows[i * vectors + c], coefficient, pivot[c], mask)
            for j in T.Unroll(panel, explicit=True):
                for i in T.Unroll(panel, explicit=True):
                    if i > j:
                        coefficient = T.simd.vld(
                            lower[p * panel + i, p * panel + j], dist="BRC_B32"
                        )
                        for c in T.Unroll(vectors, explicit=True):
                            T.simd.vmula(
                                rows[i * vectors + c], coefficient, rows[j * vectors + c], mask
                            )
            for i in T.Unroll(panel, explicit=True):
                for c in T.Unroll(vectors, explicit=True):
                    T.simd.vsts(rhs[p * panel + i, c * 64], rows[i * vectors + c], mask)
            T.simd.mem_bar("VST_VLD")


def _build_solve(slots, heads, dim_k, dim_v, chunk, max_blocks, threads, panel, hf32):
    inverse = hf32 is not None
    features = dim_k + dim_v
    rhs_width = 64 if inverse else features
    vectors = rhs_width // 64
    jobs = slots * heads
    blocks = min(max_blocks, jobs)
    rounds = (jobs + blocks - 1) // blocks

    @T.prim_func
    def main(
        KV: T.Tensor((slots, heads, chunk, features), "float32"),
        Gram: T.Tensor((slots, heads, chunk, chunk), "float32"),
        BP: T.Tensor((slots, heads, chunk), "float32"),
        W: T.Tensor((slots, heads, chunk, dim_k), "float32"),
        UT: T.Tensor((slots, heads, dim_v, chunk), "float32"),
    ):
        with T.Kernel(blocks) as block:
            rhs_ub = T.alloc_shared((chunk, rhs_width), "float32")
            gram_ub = T.alloc_shared((chunk, chunk), "float32")
            beta_ub = T.alloc_shared((chunk,), "float32")
            if inverse:
                T.set_hf32_mode(hf32)
                inverse_ub = T.alloc_shared((chunk, chunk), "float32")
                inverse_l1 = T.alloc_l1((chunk, chunk), "float32")
                rhs_t_l1 = T.alloc_l1((dim_k, chunk), "float32")
                w_acc = T.alloc_l0c((chunk, dim_k), "float32")
                ut_acc = T.alloc_l0c((dim_v, chunk), "float32")
                T.annotate_buffer_versions(
                    {
                        rhs_ub: 1,
                        gram_ub: 1,
                        beta_ub: 1,
                        inverse_ub: 1,
                        inverse_l1: 1,
                        rhs_t_l1: 1,
                        w_acc: 1,
                        ut_acc: 1,
                    }
                )
            else:
                w_ub = T.alloc_shared((chunk, dim_k), "float32")
                ut_ub = T.alloc_shared((dim_v, chunk), "float32")
                T.annotate_buffer_versions({rhs_ub: 1, gram_ub: 1, beta_ub: 1, w_ub: 1, ut_ub: 1})
            for work_round in T.serial(rounds):
                job = work_round * blocks + block
                if job < jobs:
                    slot = job // heads
                    head = job % heads
                    if not inverse:
                        T.copy(KV[slot, head, 0:chunk, 0:features], rhs_ub)
                    T.copy(Gram[slot, head, 0:chunk, 0:chunk], gram_ub)
                    T.copy(BP[slot, head, 0:chunk], beta_ub)
                    with T.SimtVF(threads=threads):
                        for i, j in T.Parallel(chunk, chunk):
                            gram_ub[i, j] = -(gram_ub[i, j] * beta_ub[i])
                        if inverse:
                            for i, j in T.Parallel(chunk, rhs_width):
                                rhs_ub[i, j] = T.if_then_else(i == j, T.float32(1), T.float32(0))
                    _forward_substitution(
                        rhs_ub, gram_ub, beta_ub, chunk, panel, vectors, not inverse
                    )
                    if inverse:
                        with T.SimtVF(threads=threads):
                            for i, j in T.Parallel(chunk, chunk):
                                inverse_ub[i, j] = rhs_ub[i, j] * beta_ub[j]
                        T.copy(inverse_ub, inverse_l1)
                        T.copy(KV[slot, head, 0:chunk, 0:dim_k], rhs_t_l1, transpose=True)
                        T.gemm(inverse_l1, rhs_t_l1, w_acc, transpose_B=True, clear_accum=True)
                        T.copy(w_acc, W[slot, head, 0:chunk, 0:dim_k])
                        T.copy(KV[slot, head, 0:chunk, dim_k:features], rhs_t_l1, transpose=True)
                        T.gemm(rhs_t_l1, inverse_l1, ut_acc, transpose_B=True, clear_accum=True)
                        T.copy(ut_acc, UT[slot, head, 0:dim_v, 0:chunk])
                    else:
                        with T.SimtVF(threads=threads):
                            for t, i in T.Parallel(chunk, dim_k):
                                w_ub[t, i] = rhs_ub[t, i]
                            for j, t in T.Parallel(dim_v, chunk):
                                ut_ub[j, t] = rhs_ub[t, dim_k + j]
                        T.copy(w_ub, W[slot, head, 0:chunk, 0:dim_k])
                        T.copy(ut_ub, UT[slot, head, 0:dim_v, 0:chunk])

    return tilelang.compile(main, target="ascend", out_idx=[3, 4])


def _build_fused_states_project(
    batch,
    slots,
    heads,
    dim_k,
    dim_v,
    num_sequences,
    dtype,
    chunk,
    has_initial,
    max_blocks,
    threads,
    identity_rows,
    value_tile,
    hf32,
    key_width,
    packed,
    aligned_dense,
    seq_len,
):
    if dim_k not in (64, 128) or dim_v != dim_k or chunk != 32:
        raise NotImplementedError("fused state/output requires K=V=64 or 128 and chunk=32")
    if value_tile not in (32, 64, 128) or dim_v % value_tile:
        raise NotImplementedError("value tile must be 32, 64 or 128 and divide V")
    if identity_rows < 16 or identity_rows % 16 or dim_k % identity_rows:
        raise NotImplementedError("identity strips must be aligned to 16 rows and divide K")
    value_tiles = dim_v // value_tile
    jobs = num_sequences * heads * value_tiles
    blocks = min(max_blocks, jobs)
    rounds = (jobs + blocks - 1) // blocks
    half_v = value_tile // 2
    key_vectors = dim_k // 64
    identity_tiles = dim_k // identity_rows

    @T.prim_func
    def main(
        W: T.Tensor((slots, heads, chunk, dim_k), "float32"),
        UT: T.Tensor((slots, heads, dim_v, chunk), "float32"),
        KP: T.Tensor((slots, heads, chunk, key_width), "float32"),
        QP: T.Tensor((slots, heads, chunk, dim_k), "float32"),
        QK: T.Tensor((slots, heads, chunk, chunk), "float32"),
        Initial: T.Tensor((num_sequences, heads, dim_k, dim_v), "float32"),
        Meta: T.Tensor((num_sequences + 1,), "int32"),
        Offsets: T.Tensor((num_sequences + 1,), "int64"),
        OP: T.Tensor((batch, seq_len, heads, dim_v), dtype),
        Final: T.Tensor((num_sequences, heads, dim_k, dim_v), "float32"),
    ):
        with T.MixedKernel(blocks) as (block, sid):
            T.set_hf32_mode(hf32)
            state_l1 = T.alloc_l1((value_tile, dim_k), "float32")
            w_l1 = T.alloc_l1((chunk, dim_k), "float32")
            q_l1 = T.alloc_l1((chunk, dim_k), "float32")
            d_l1 = T.alloc_l1((value_tile, chunk), "float32")
            kt_l1 = T.alloc_l1((dim_k, chunk), "float32")
            g_l1 = T.alloc_l1((chunk, chunk), "float32")
            identity_l1 = T.alloc_l1((dim_k, dim_k), "float32")
            left_l0 = T.alloc_l0a((value_tile, dim_k), "float32")
            right_l0 = T.alloc_l0b((dim_k, dim_k), "float32")
            projected = T.alloc_l0c((value_tile, chunk), "float32")
            output_acc = T.alloc_l0c((chunk, value_tile), "float32")
            state_acc = T.alloc_l0c((value_tile, dim_k), "float32")
            state_ub = T.alloc_shared((half_v, dim_k), "float32")
            final_ub = T.alloc_shared((dim_k, half_v), "float32")
            projected_ub = T.alloc_shared((half_v, chunk), "float32")
            u_ub = T.alloc_shared((half_v, chunk), "float32")
            d_ub = T.alloc_shared((half_v, chunk), "float32")
            g_ub = T.alloc_shared((chunk, chunk), "float32")
            g_masked = T.alloc_shared((chunk, chunk), "float32")
            identity_ub = T.alloc_shared((identity_rows, dim_k), "float32")
            T.annotate_buffer_versions(
                {
                    state_l1: 1,
                    w_l1: 1,
                    q_l1: 1,
                    d_l1: 1,
                    kt_l1: 1,
                    g_l1: 1,
                    identity_l1: 1,
                    left_l0: 1,
                    right_l0: 1,
                    projected: 1,
                    output_acc: 1,
                    state_acc: 1,
                    state_ub: 1,
                    final_ub: 1,
                    projected_ub: 1,
                    u_ub: 1,
                    d_ub: 1,
                    g_ub: 1,
                    g_masked: 1,
                    identity_ub: 1,
                }
            )
            if has_initial:
                for identity_tile in T.serial(identity_tiles):
                    with T.SimtVF(threads=threads):
                        for i, j in T.Parallel(identity_rows, dim_k):
                            identity_ub[i, j] = T.if_then_else(
                                identity_tile * identity_rows + i == j, T.float32(1), T.float32(0)
                            )
                    T.copy(
                        identity_ub,
                        identity_l1[
                            identity_tile * identity_rows : (identity_tile + 1) * identity_rows,
                            0:dim_k,
                        ],
                    )
            for work_round in T.serial(rounds):
                job = work_round * blocks + block
                if job < jobs:
                    sequence = job // (heads * value_tiles)
                    head = job // value_tiles % heads
                    value_begin = (job % value_tiles) * value_tile
                    first_slot = Meta[sequence]
                    end_slot = Meta[sequence + 1]
                    if packed:
                        batch_index = 0
                        first_token = T.cast(Offsets[sequence], "int32")
                        end_token = T.cast(Offsets[sequence + 1], "int32")
                    else:
                        batch_index = sequence
                        first_token = 0
                        end_token = seq_len
                    if has_initial:
                        T.copy(
                            Initial[
                                sequence, head, 0:dim_k, value_begin : value_begin + value_tile
                            ],
                            state_l1,
                            transpose=True,
                        )
                        T.copy(state_l1, left_l0)
                        T.copy(identity_l1, right_l0)
                        T.gemm(left_l0, right_l0, state_acc, transpose_B=True, clear_accum=True)
                        T.dual_copy(state_acc, state_ub)
                    else:
                        with T.SimdVF():
                            zero_mask = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), "float32", zero_mask)
                            for j in range(half_v):
                                for c in T.Unroll(key_vectors, explicit=True):
                                    T.simd.vsts(state_ub[j, c * 64], zero, zero_mask)
                    for slot in T.serial(first_slot, end_slot):
                        T.dual_copy(state_ub, state_l1)
                        T.copy(W[slot, head, 0:chunk, 0:dim_k], w_l1)
                        T.copy(QP[slot, head, 0:chunk, 0:dim_k], q_l1)
                        T.copy(state_l1, left_l0)
                        T.copy(w_l1, right_l0[0:chunk, 0:dim_k])
                        T.gemm(
                            left_l0,
                            right_l0[0:chunk, 0:dim_k],
                            projected,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.dual_copy(projected, projected_ub)
                        T.copy(q_l1, left_l0[0:chunk, 0:dim_k])
                        T.copy(state_l1, right_l0[0:value_tile, 0:dim_k])
                        T.gemm(
                            left_l0[0:chunk, 0:dim_k],
                            right_l0[0:value_tile, 0:dim_k],
                            output_acc,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.dual_copy(
                            UT[slot, head, value_begin : value_begin + value_tile, 0:chunk], u_ub
                        )
                        with T.SimdVF():
                            delta_mask = T.simd.pset(32)
                            for x in range(half_v * chunk // 64):
                                u_reg = T.simd.vld(u_ub[x * 64 // chunk, x * 64 % chunk])
                                prediction_reg = T.simd.vld(
                                    projected_ub[x * 64 // chunk, x * 64 % chunk]
                                )
                                delta_reg = T.simd.vsub(u_reg, prediction_reg, delta_mask)
                                T.simd.vsts(
                                    d_ub[x * 64 // chunk, x * 64 % chunk], delta_reg, delta_mask
                                )
                        T.dual_copy(d_ub, d_l1)
                        T.copy(QK[slot, head, 0:chunk, 0:chunk], g_ub)
                        with T.SimtVF(threads=threads):
                            for i, j in T.Parallel(chunk, chunk):
                                g_masked[i, j] = T.if_then_else(j <= i, g_ub[i, j], T.float32(0))
                        T.copy(g_masked, g_l1)
                        T.copy(g_l1, left_l0[0:chunk, 0:chunk])
                        T.copy(d_l1, right_l0[0:value_tile, 0:chunk])
                        T.gemm(
                            left_l0[0:chunk, 0:chunk],
                            right_l0[0:value_tile, 0:chunk],
                            output_acc,
                            transpose_B=True,
                            clear_accum=False,
                        )
                        token_begin = first_token + (slot - first_slot) * chunk
                        valid = chunk if aligned_dense else T.min(chunk, end_token - token_begin)
                        # The destination clips rows while the source keeps its full pitch.
                        T.copy(
                            output_acc,
                            OP[
                                batch_index,
                                token_begin : token_begin + valid,
                                head,
                                value_begin : value_begin + value_tile,
                            ],
                        )
                        T.copy(KP[slot, head, 0:chunk, 0:dim_k], kt_l1, transpose=True)
                        T.copy(d_l1, left_l0[0:value_tile, 0:chunk])
                        T.copy(kt_l1, right_l0[0:dim_k, 0:chunk])
                        if has_initial:
                            T.gemm(
                                left_l0[0:value_tile, 0:chunk],
                                right_l0[0:dim_k, 0:chunk],
                                state_acc,
                                transpose_B=True,
                                clear_accum=False,
                            )
                        else:
                            T.gemm(
                                left_l0[0:value_tile, 0:chunk],
                                right_l0[0:dim_k, 0:chunk],
                                state_acc,
                                transpose_B=True,
                                clear_accum=slot == first_slot,
                            )
                        T.dual_copy(state_acc, state_ub)
                    if packed and has_initial:
                        if first_slot == end_slot:
                            local_value_begin = value_begin + sid * half_v
                            T.copy(
                                Initial[
                                    sequence,
                                    head,
                                    0:dim_k,
                                    local_value_begin : local_value_begin + half_v,
                                ],
                                final_ub,
                            )
                        else:
                            with T.SimtVF(threads=threads):
                                for i, j in T.Parallel(dim_k, half_v):
                                    final_ub[i, j] = state_ub[j, i]
                    else:
                        with T.SimtVF(threads=threads):
                            for i, j in T.Parallel(dim_k, half_v):
                                final_ub[i, j] = state_ub[j, i]
                    T.dual_copy(
                        final_ub,
                        Final[sequence, head, 0:dim_k, value_begin : value_begin + value_tile],
                    )

    return tilelang.compile(main, target="ascend", out_idx=[8, 9])


class AscendDeltaNetChunkedKernel(Kernel, DeltaNetFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: DeltaNetCall) -> Optional[str]:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 token tensors"
        if call.dim_k not in (64, 128) or call.dim_v != call.dim_k:
            return "requires a square 64-wide or 128-wide state"
        if call.batch < 1 or call.seq_len < 1 or call.heads < 1:
            return "requires positive batch, sequence length and head count"
        if call.seq_len > 2147483647:
            return "requires sequence positions representable by int32"
        if call.num_sequences < 1:
            return "requires at least one sequence"
        if call.varlen and call.batch != 1:
            return "packed sequences require batch=1"
        if not call.varlen and call.num_sequences != call.batch:
            return "dense calls require num_sequences=batch"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: DeltaNetCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: DeltaNetCall):
        reason = type(self).refusal(call)
        if reason is not None:
            raise NotImplementedError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self._has_initial = bool(call.has_initial_state)
        self._packed = bool(call.varlen)
        self._decode = call.seq_len == 1 and not self._packed
        self._state_shape = (call.num_sequences, call.heads, call.dim_k, call.dim_v)
        self._offset_shape = (call.num_sequences + 1,)
        batch = int(call.batch)
        seq_len = int(call.seq_len)
        heads = int(call.heads)
        dim_k = int(call.dim_k)
        dim_v = int(call.dim_v)
        num_sequences = int(call.num_sequences)
        dtype = str(call.dtype).split(".")[-1]
        scale = float(call.scale)
        normalize = bool(call.l2norm)
        if self._decode:
            self.config = {
                "max_blocks": _DECODE_BLOCKS,
                "simt_threads": _DECODE_THREADS,
                "state_layout": "KV",
                "reduction_stripes": 4,
                "value_vector_width": 64,
            }
            self._compiled_fn = _build_dense_decode(
                batch,
                heads,
                dim_k,
                dim_v,
                dtype,
                scale,
                normalize,
                self._has_initial,
                _DECODE_BLOCKS,
                _DECODE_THREADS,
            )
        else:
            chunk = _CHUNK
            dense_chunks = (seq_len + chunk - 1) // chunk
            slots = dense_chunks + num_sequences - 1 if self._packed else batch * dense_chunks
            if slots > 2147483647:
                raise NotImplementedError("requires chunk indices representable by int32")
            inverse_solve = dim_k == 128
            hf32 = _K128_HF32 if inverse_solve else None
            key_width = dim_k + dim_v
            aligned_dense = not self._packed and seq_len % chunk == 0
            if dim_v == 64:
                # Split independent value columns while there are unoccupied Cube cores.
                state_value_tile = 32 if num_sequences * heads < _CUBE_BLOCKS else dim_v
            else:
                state_value_tile = (
                    _DENSE_BULK_STATE_VALUE_TILE if aligned_dense else _STATE_VALUE_TILE
                )
            self.config = {
                "chunk": chunk,
                "cube_blocks": _CUBE_BLOCKS,
                "threads": _THREADS,
                "solve_blocks": _CUBE_BLOCKS if inverse_solve else _SOLVE_BLOCKS,
                "solve_threads": _SOLVE_THREADS,
                "solve_panel": _SOLVE_PANEL,
                "solve_algorithm": "fused_inverse_direct_gemm" if inverse_solve else "direct_panel",
                "identity_rows": _IDENTITY_ROWS,
                "pack_arithmetic": "explicit_simd",
                "pack_token_widening": "unpack_b16_vcvt",
                "state_accumulation": "persistent_fp32_l0c",
                "fused_state_output": True,
                "state_value_tile": state_value_tile,
                "materialize_kt": False,
                "hf32": hf32,
                "compact_kv": True,
                "fused_pack_gram": True,
                "weighted_inverse": inverse_solve,
                "direct_public_output": True,
                "direct_final_state": True,
            }
            self._pack = _build_pack_grams(
                batch,
                seq_len,
                heads,
                dim_k,
                dim_v,
                num_sequences,
                dtype,
                scale,
                normalize,
                self._packed,
                chunk,
                slots,
                _CUBE_BLOCKS,
                _THREADS,
                hf32=hf32,
            )
            self._solve = _build_solve(
                slots,
                heads,
                dim_k,
                dim_v,
                chunk,
                _CUBE_BLOCKS if inverse_solve else _SOLVE_BLOCKS,
                _SOLVE_THREADS,
                _SOLVE_PANEL,
                hf32,
            )
            self._states_project = _build_fused_states_project(
                batch,
                slots,
                heads,
                dim_k,
                dim_v,
                num_sequences,
                dtype,
                chunk,
                self._has_initial,
                _CUBE_BLOCKS,
                _THREADS,
                _IDENTITY_ROWS,
                state_value_tile,
                hf32,
                key_width,
                self._packed,
                aligned_dense,
                seq_len,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (initial_state is not None) != self._has_initial:
            raise ValueError("initial_state presence differs from the specialized call")
        if (cu_seqlens is not None) != self._packed:
            raise ValueError("cu_seqlens presence differs from the specialized call")
        state_arg = initial_state
        if state_arg is None:
            state_arg = torch.empty(self._state_shape, dtype=torch.float32, device=q.device)
        if self._decode:
            output, final_state = self._compiled_fn(q, k, v, beta, state_arg)
            return output, final_state
        offsets_arg = cu_seqlens
        if offsets_arg is None:
            offsets_arg = torch.empty(self._offset_shape, dtype=torch.int64, device=q.device)
        # Device offsets are authoritative; workspaces are invocation-local.
        # Packing feeds Gram matrices directly into the triangular solve.
        qp, kv, bp, meta, gram, qk = self._pack(q, k, v, beta, offsets_arg)
        w, ut = self._solve(kv, gram, bp)
        return self._states_project(w, ut, kv, qp, qk, state_arg, meta, offsets_arg)


tileops.backend.register_kernel_type("DeltaNetFwdOp", "ascend", AscendDeltaNetChunkedKernel)
