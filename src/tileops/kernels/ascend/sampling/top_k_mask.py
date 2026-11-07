from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall, TopKMaskFwdInterface


def build(specialization):
    batch, vocab = map(int, specialization["shape"]["logits"])
    dtype = specialization["dtype"]["logits"]
    if batch == 0 or vocab == 0:

        def empty_call(logits, k):
            return (torch.empty_like(logits),)

        return empty_call

    if dtype == "float32":
        bit_dtype = "uint32"
        bit_width = 32
        sign_bit = 0x80000000
        word_mask = 0xFFFFFFFF
        magnitude_mask = 0x7FFFFFFF
        infinity_magnitude = 0x7F800000
        pack = 1
    elif dtype == "float16":
        bit_dtype = "uint16"
        bit_width = 16
        sign_bit = 0x8000
        word_mask = 0xFFFF
        magnitude_mask = 0x7FFF
        infinity_magnitude = 0x7C00
        pack = 2
    elif dtype == "bfloat16":
        bit_dtype = "uint16"
        bit_width = 16
        sign_bit = 0x8000
        word_mask = 0xFFFF
        magnitude_mask = 0x7FFF
        infinity_magnitude = 0x7F80
        pack = 2
    else:
        raise ValueError("Unsupported logits dtype")

    wide = batch <= 17
    digit_bits = 8 if wide else 4
    radix = 1 << digit_bits
    frequency_bins = 256 if wide else 128
    sentinel = 255 if wide else 128
    if wide:
        byte_padding = 0xFFFFFF00 if bit_width == 32 else 0xFF00
        excluded_word = word_mask
    else:
        byte_padding = 0x80808000 if bit_width == 32 else 0x8000
        excluded_word = 0x80808080 if bit_width == 32 else 0x8080

    threads = 128
    export_lanes = min(radix, threads)
    export_groups = radix // export_lanes
    chunk = 4096
    pieces = (vocab + chunk - 1) // chunk
    last_piece = pieces - 1
    tail = vocab - last_piece * chunk
    merge_rounds = (pieces + threads - 1) // threads
    encode_rounds = chunk // (threads * 4)
    histogram_rounds = chunk // 256
    vector_lanes = 2048 // bit_width
    bit_vector_dtype = bit_dtype + "x" + str(vector_lanes)
    value_vector_dtype = dtype + "x" + str(vector_lanes)
    store_dist = "NORM_B32" if bit_width == 32 else "NORM_B16"
    negative_infinity_bits = sign_bit | infinity_magnitude
    aligned_rows = vocab % vector_lanes == 0
    fused_path = batch == 1 and aligned_rows
    total_elements = batch * vocab
    output_tile = 1024
    output_programs = (total_elements + output_tile - 1) // output_tile

    if fused_path and pieces <= 32:
        passes = bit_width // 8

        @T.prim_func
        def persistent_radix(
            X: T.Tensor((batch, vocab), dtype),
            K: T.Tensor((batch,), "int32"),
            Counts: T.Tensor((passes, pieces, 256), "int32"),
            Y: T.Tensor((batch, vocab), dtype),
        ):
            with T.Kernel(pieces) as piece:
                values = T.alloc_shared((chunk,), dtype)
                output_values = T.alloc_shared((chunk,), dtype)
                frequencies = T.alloc_shared((256,), "uint16")
                local_counts = T.alloc_shared((256,), "int32")
                merge_counts = T.alloc_shared((pieces, 256), "int32")
                cumulative_ub = T.alloc_shared((256,), "int32")
                rank_ub = T.alloc_shared((8,), "int32")
                prefix_ub = T.alloc_shared((8,), "int32")
                residual_ub = T.alloc_shared((8,), "int32")

                T.copy(K[0:1], rank_ub[0:1])
                if piece < last_piece:
                    T.copy(X[0, piece * chunk : piece * chunk + chunk], values)
                else:
                    T.copy(X[0, last_piece * chunk : vocab], values[0:tail])
                with T.SimtVF(threads=threads):
                    for lane in T.Parallel(1):
                        prefix_ub[lane] = 0
                        residual_ub[lane] = rank_ub[lane]
                T.ascend_pipe_barrier("PIPE_ALL")

                for phase in T.serial(passes):
                    shift = bit_width - 8 - phase * 8
                    higher_mask = T.uint32(word_mask) ^ (T.uint32(word_mask) >> (phase * 8))
                    with T.SimdVF():
                        mask8 = T.simd.pset(8)
                        mask16 = T.simd.pset(16)
                        mask = T.simd.pset(bit_width)
                        frequency0 = T.simd.alloc_var("uint16")
                        frequency1 = T.simd.alloc_var("uint16")
                        frequency0 = T.simd.vdup(T.uint16(0), "uint16", mask16)
                        frequency1 = T.simd.vdup(T.uint16(0), "uint16", mask16)
                        selected_prefix = T.cast(T.reinterpret(prefix_ub[0], "uint32"), bit_dtype)
                        rank_value = rank_ub[0]
                        sign_vector = T.simd.vdup(T.cast(sign_bit, bit_dtype), bit_dtype, mask)
                        word_vector = T.simd.vdup(T.cast(word_mask, bit_dtype), bit_dtype, mask)
                        magnitude_vector = T.simd.vdup(
                            T.cast(magnitude_mask, bit_dtype), bit_dtype, mask
                        )
                        higher_vector = T.simd.vdup(T.cast(higher_mask, bit_dtype), bit_dtype, mask)
                        bit0_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << shift, bit_dtype), bit_dtype, mask
                        )
                        bit1_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 1), bit_dtype), bit_dtype, mask
                        )
                        bit2_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 2), bit_dtype), bit_dtype, mask
                        )
                        bit3_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 3), bit_dtype), bit_dtype, mask
                        )
                        bit4_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 4), bit_dtype), bit_dtype, mask
                        )
                        bit5_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 5), bit_dtype), bit_dtype, mask
                        )
                        bit6_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 6), bit_dtype), bit_dtype, mask
                        )
                        bit7_vector = T.simd.vdup(
                            T.cast(T.uint32(1) << (shift + 7), bit_dtype), bit_dtype, mask
                        )
                        zero_vector = T.simd.vdup(T.cast(0, bit_dtype), bit_dtype, mask)
                        weight0_vector = T.simd.vdup(T.cast(1, bit_dtype), bit_dtype, mask)
                        weight1_vector = T.simd.vdup(T.cast(2, bit_dtype), bit_dtype, mask)
                        weight2_vector = T.simd.vdup(T.cast(4, bit_dtype), bit_dtype, mask)
                        weight3_vector = T.simd.vdup(T.cast(8, bit_dtype), bit_dtype, mask)
                        weight4_vector = T.simd.vdup(T.cast(16, bit_dtype), bit_dtype, mask)
                        weight5_vector = T.simd.vdup(T.cast(32, bit_dtype), bit_dtype, mask)
                        weight6_vector = T.simd.vdup(T.cast(64, bit_dtype), bit_dtype, mask)
                        weight7_vector = T.simd.vdup(T.cast(128, bit_dtype), bit_dtype, mask)
                        reverse_vector = T.simd.vdup(T.cast(255, bit_dtype), bit_dtype, mask)
                        padding_vector = T.simd.vdup(
                            T.cast(byte_padding, bit_dtype), bit_dtype, mask
                        )
                        excluded_vector = T.simd.vdup(
                            T.cast(excluded_word, bit_dtype), bit_dtype, mask
                        )
                        if rank_value < vocab:
                            for vector_index in T.serial(chunk // vector_lanes):
                                if (piece < last_piece) | (vector_index < tail // vector_lanes):
                                    original = T.reinterpret(
                                        T.simd.vld(
                                            values[vector_index * vector_lanes], dist="NORM"
                                        ),
                                        bit_vector_dtype,
                                    )
                                    magnitude = T.simd.vand(original, magnitude_vector, mask)
                                    positive_key = T.simd.vxor(original, sign_vector, mask)
                                    negative_key = T.simd.vxor(original, word_vector, mask)
                                    signed_key = T.simd.vsel(
                                        negative_key,
                                        positive_key,
                                        T.simd.vcmps(
                                            original, T.cast(sign_bit, bit_dtype), mask, op="ge"
                                        ),
                                    )
                                    zero_key = T.simd.vsel(
                                        sign_vector,
                                        signed_key,
                                        T.simd.vcmps(
                                            magnitude, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    key = T.simd.vsel(
                                        word_vector,
                                        zero_key,
                                        T.simd.vcmps(
                                            magnitude,
                                            T.cast(infinity_magnitude, bit_dtype),
                                            mask,
                                            op="gt",
                                        ),
                                    )
                                    key_prefix = T.simd.vand(key, higher_vector, mask)
                                    selected_bit0 = T.simd.vand(key, bit0_vector, mask)
                                    digit0 = T.simd.vsel(
                                        zero_vector,
                                        weight0_vector,
                                        T.simd.vcmps(
                                            selected_bit0, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit1 = T.simd.vand(key, bit1_vector, mask)
                                    digit1 = T.simd.vsel(
                                        zero_vector,
                                        weight1_vector,
                                        T.simd.vcmps(
                                            selected_bit1, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit2 = T.simd.vand(key, bit2_vector, mask)
                                    digit2 = T.simd.vsel(
                                        zero_vector,
                                        weight2_vector,
                                        T.simd.vcmps(
                                            selected_bit2, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit3 = T.simd.vand(key, bit3_vector, mask)
                                    digit3 = T.simd.vsel(
                                        zero_vector,
                                        weight3_vector,
                                        T.simd.vcmps(
                                            selected_bit3, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit4 = T.simd.vand(key, bit4_vector, mask)
                                    digit4 = T.simd.vsel(
                                        zero_vector,
                                        weight4_vector,
                                        T.simd.vcmps(
                                            selected_bit4, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit5 = T.simd.vand(key, bit5_vector, mask)
                                    digit5 = T.simd.vsel(
                                        zero_vector,
                                        weight5_vector,
                                        T.simd.vcmps(
                                            selected_bit5, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit6 = T.simd.vand(key, bit6_vector, mask)
                                    digit6 = T.simd.vsel(
                                        zero_vector,
                                        weight6_vector,
                                        T.simd.vcmps(
                                            selected_bit6, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    selected_bit7 = T.simd.vand(key, bit7_vector, mask)
                                    digit7 = T.simd.vsel(
                                        zero_vector,
                                        weight7_vector,
                                        T.simd.vcmps(
                                            selected_bit7, T.cast(0, bit_dtype), mask, op="eq"
                                        ),
                                    )
                                    digit01 = T.simd.vor(digit0, digit1, mask)
                                    digit23 = T.simd.vor(digit2, digit3, mask)
                                    digit45 = T.simd.vor(digit4, digit5, mask)
                                    digit67 = T.simd.vor(digit6, digit7, mask)
                                    low_digit = T.simd.vor(digit01, digit23, mask)
                                    high_digit = T.simd.vor(digit45, digit67, mask)
                                    full_digit = T.simd.vor(low_digit, high_digit, mask)
                                    descending_digit = T.simd.vxor(full_digit, reverse_vector, mask)
                                    padded_digit = T.simd.vor(
                                        descending_digit, padding_vector, mask
                                    )
                                    encoded = T.simd.vsel(
                                        padded_digit,
                                        excluded_vector,
                                        T.simd.vcmps(key_prefix, selected_prefix, mask, op="eq"),
                                    )
                                    digit_bytes = T.reinterpret(encoded, "uint8x256")
                                    T.simd.chistv2(frequency0, digit_bytes, mask8, bin=0)
                                    T.simd.chistv2(frequency1, digit_bytes, mask8, bin=1)
                        T.simd.vsts(frequencies[0], frequency0, mask16, dist="NORM_B16")
                        T.simd.vsts(frequencies[128], frequency1, mask16, dist="NORM_B16")

                    with T.SimtVF(threads=threads):
                        for export_group in T.serial(2):
                            for export_lane in T.Parallel(128):
                                export_digit = export_group * 128 + export_lane
                                local_counts[export_digit] = T.cast(
                                    frequencies[export_digit], "int32"
                                )
                    T.ascend_pipe_barrier("PIPE_ALL")
                    with T.Stage(0), T.Task():
                        T.copy(local_counts, Counts[phase, piece, 0:256])
                        T.ascend_pipe_barrier("PIPE_MTE3")
                    with T.Stage(0), T.Task():
                        T.ascend_sync_inter_arrive("PIPE_MTE3", 8)
                    with T.Stage(0), T.Task():
                        T.ascend_sync_inter_wait("PIPE_MTE2", 8)
                    with T.Stage(0), T.Task():
                        T.copy(Counts[phase, 0:pieces, 0:256], merge_counts)
                    with T.SimdVF():
                        merge_mask = T.simd.pset(32)
                        merged_counts = T.simd.alloc_var("int32")
                        for digit_group in T.serial(4):
                            merged_counts = T.simd.vdup(T.int32(0), "int32", merge_mask)
                            for source_piece in T.serial(pieces):
                                piece_counts = T.simd.vld(
                                    merge_counts[source_piece, digit_group * 64], dist="NORM"
                                )
                                merged_counts = T.simd.vadd(merged_counts, piece_counts, merge_mask)
                            T.simd.vsts(
                                cumulative_ub[digit_group * 64],
                                merged_counts,
                                merge_mask,
                                dist="NORM_B32",
                            )
                    with T.SimtVF(threads=threads):
                        cumulative = T.alloc_fragment((256,), "int32")
                        choice = T.alloc_reducer((1,), "int32", op="min")
                        choice_result = T.alloc_fragment((1,), "int32")
                        before = T.alloc_reducer((1,), "int32", op="sum")
                        before_result = T.alloc_fragment((1,), "int32")
                        next_prefix = T.alloc_fragment((1,), "uint32")
                        next_rank = T.alloc_fragment((1,), "int32")
                        for digit in T.Parallel(256):
                            cumulative[digit] = cumulative_ub[digit]
                        T.reducer_init(choice)
                        for digit in T.Parallel(256):
                            T.reducer_update(
                                choice[0],
                                T.if_then_else(cumulative[digit] >= residual_ub[0], digit, 255),
                            )
                        T.finalize_reducer(choice, choice_result)
                        T.reducer_init(before)
                        for digit in T.Parallel(256):
                            if digit + 1 == choice_result[0]:
                                T.reducer_update(before[0], cumulative[digit])
                        T.finalize_reducer(before, before_result)
                        next_prefix[0] = T.reinterpret(prefix_ub[0], "uint32") | (
                            T.cast(255 - choice_result[0], "uint32") << shift
                        )
                        next_rank[0] = residual_ub[0] - before_result[0]
                        for lane in T.Parallel(1):
                            prefix_ub[lane] = T.reinterpret(next_prefix[lane], "int32")
                            residual_ub[lane] = next_rank[lane]
                    T.ascend_pipe_barrier("PIPE_ALL")

                with T.SimdVF():
                    mask = T.simd.pset(bit_width)
                    threshold_key = T.reinterpret(prefix_ub[0], "uint32")
                    rank_value = rank_ub[0]
                    masked = T.simd.vdup(T.cast(negative_infinity_bits, bit_dtype), bit_dtype, mask)
                    filtered = T.simd.alloc_var(bit_dtype)
                    for vector_index in T.serial(chunk // vector_lanes):
                        if (piece < last_piece) | (vector_index < tail // vector_lanes):
                            original = T.reinterpret(
                                T.simd.vld(values[vector_index * vector_lanes], dist="NORM"),
                                bit_vector_dtype,
                            )
                            filtered = original
                            if rank_value < vocab:
                                if threshold_key >= T.uint32(sign_bit):
                                    positive_cutoff = T.cast(
                                        threshold_key - T.uint32(sign_bit), bit_dtype
                                    )
                                    filtered = T.simd.vsel(
                                        original,
                                        masked,
                                        T.simd.vcmps(original, positive_cutoff, mask, op="ge"),
                                    )
                                    filtered = T.simd.vsel(
                                        filtered,
                                        masked,
                                        T.simd.vcmps(
                                            original,
                                            T.cast(infinity_magnitude, bit_dtype),
                                            mask,
                                            op="le",
                                        ),
                                    )
                                    if threshold_key == T.uint32(sign_bit):
                                        filtered = T.simd.vsel(
                                            original,
                                            filtered,
                                            T.simd.vcmps(
                                                original, T.cast(sign_bit, bit_dtype), mask, op="eq"
                                            ),
                                        )
                                else:
                                    negative_cutoff = T.cast(
                                        T.uint32(word_mask) - threshold_key, bit_dtype
                                    )
                                    filtered = T.simd.vsel(
                                        original,
                                        masked,
                                        T.simd.vcmps(
                                            original,
                                            T.cast(infinity_magnitude, bit_dtype),
                                            mask,
                                            op="le",
                                        ),
                                    )
                                    filtered = T.simd.vsel(
                                        original,
                                        filtered,
                                        T.simd.vcmps(
                                            original, T.cast(sign_bit, bit_dtype), mask, op="ge"
                                        ),
                                    )
                                    filtered = T.simd.vsel(
                                        filtered,
                                        masked,
                                        T.simd.vcmps(original, negative_cutoff, mask, op="le"),
                                    )
                            T.simd.vsts(
                                output_values[vector_index * vector_lanes],
                                T.reinterpret(filtered, value_vector_dtype),
                                mask,
                                dist=store_dist,
                            )
                if piece < last_piece:
                    T.copy(output_values, Y[0, piece * chunk : piece * chunk + chunk])
                else:
                    T.copy(output_values[0:tail], Y[0, last_piece * chunk : vocab])

        persistent_kernel = tilelang.compile(persistent_radix, target="ascend", out_idx=[])

        def persistent_run(logits, k):
            counts = torch.empty((passes, pieces, 256), dtype=torch.int32, device=logits.device)
            output = torch.empty_like(logits)
            persistent_kernel(logits, k, counts, output)
            return (output,)

        return persistent_run

    def make_stage(shift, merge_shift):
        hist_shift = max(shift, 0)
        higher_mask = word_mask ^ ((1 << (hist_shift + digit_bits)) - 1)
        merge_higher_mask = (
            word_mask ^ ((1 << (merge_shift + digit_bits)) - 1) if merge_shift >= 0 else 0
        )
        safe_merge_shift = max(merge_shift, 0)
        bit0_mask = 1 << hist_shift
        bit1_mask = 1 << (hist_shift + 1)
        bit2_mask = 1 << (hist_shift + 2)
        bit3_mask = 1 << (hist_shift + 3)
        bit4_mask = (1 << (hist_shift + 4)) & word_mask
        bit5_mask = (1 << (hist_shift + 5)) & word_mask
        bit6_mask = (1 << (hist_shift + 6)) & word_mask
        bit7_mask = (1 << (hist_shift + 7)) & word_mask

        @T.prim_func
        def radix_stage(
            X: T.Tensor((batch, vocab), dtype),
            K: T.Tensor((batch,), "int32"),
            PreviousCounts: T.Tensor((batch, pieces, radix), "int32"),
            PreviousPrefix: T.Tensor((batch,), "int32"),
            PreviousRank: T.Tensor((batch,), "int32"),
            Counts: T.Tensor((batch, pieces, radix), "int32"),
            NextPrefix: T.Tensor((batch,), "int32"),
            NextRank: T.Tensor((batch,), "int32"),
            Y: T.Tensor((batch, vocab), dtype),
        ):
            with T.Kernel(batch * pieces) as block:
                row = block // pieces
                piece = block % pieces
                values = T.alloc_shared((chunk,), dtype)
                output_values = T.alloc_shared((chunk,), dtype)
                frequencies = T.alloc_shared((frequency_bins,), "uint16")
                digits = T.alloc_shared((chunk,), "uint8")
                rank_ub = T.alloc_shared((8,), "int32")
                prefix_ub = T.alloc_shared((8,), "int32")
                T.copy(K[row : row + 1], rank_ub[0:1])

                if merge_shift >= 0:
                    merge_counts_ub = T.alloc_shared((pieces, radix), "int32")
                    cumulative_ub = T.alloc_shared((radix,), "int32")
                    T.copy(PreviousCounts[row, 0:pieces, 0:radix], merge_counts_ub)
                    with T.SimdVF():
                        merge_mask = T.simd.pset(32)
                        merged_counts = T.simd.alloc_var("int32")
                        for digit_group in T.serial(radix // 64):
                            merged_counts = T.simd.vdup(T.int32(0), "int32", merge_mask)
                            for source_piece in T.serial(pieces):
                                piece_counts = T.simd.vld(
                                    merge_counts_ub[source_piece, digit_group * 64], dist="NORM"
                                )
                                merged_counts = T.simd.vadd(merged_counts, piece_counts, merge_mask)
                            T.simd.vsts(
                                cumulative_ub[digit_group * 64],
                                merged_counts,
                                merge_mask,
                                dist="NORM_B32",
                            )
                    with T.SimtVF(threads=threads):
                        cumulative = T.alloc_fragment((radix,), "int32")
                        choice = T.alloc_reducer((1,), "int32", op="min")
                        choice_result = T.alloc_fragment((1,), "int32")
                        before = T.alloc_reducer((1,), "int32", op="sum")
                        before_result = T.alloc_fragment((1,), "int32")
                        prefix = T.alloc_fragment((1,), "uint32")
                        residual = T.alloc_fragment((1,), "int32")
                        for digit in T.Parallel(radix):
                            cumulative[digit] = cumulative_ub[digit]
                        T.reducer_init(choice)
                        for digit in T.Parallel(radix):
                            T.reducer_update(
                                choice[0],
                                T.if_then_else(
                                    cumulative[digit] >= PreviousRank[row], digit, radix - 1
                                ),
                            )
                        T.finalize_reducer(choice, choice_result)
                        T.reducer_init(before)
                        for digit in T.Parallel(radix):
                            if digit + 1 == choice_result[0]:
                                T.reducer_update(before[0], cumulative[digit])
                        T.finalize_reducer(before, before_result)
                        prefix[0] = (
                            T.reinterpret(PreviousPrefix[row], "uint32")
                            & T.uint32(merge_higher_mask)
                        ) | (T.cast(radix - 1 - choice_result[0], "uint32") << safe_merge_shift)
                        residual[0] = PreviousRank[row] - before_result[0]
                        for lane in T.Parallel(1):
                            prefix_ub[lane] = T.reinterpret(prefix[lane], "int32")
                            if piece == 0:
                                NextPrefix[row] = T.reinterpret(prefix[lane], "int32")
                                NextRank[row] = residual[lane]
                else:
                    T.copy(PreviousPrefix[row : row + 1], prefix_ub[0:1])
                    if fused_path:
                        with T.SimtVF(threads=threads):
                            for _lane in T.Parallel(1):
                                if piece == 0:
                                    NextPrefix[row] = 0
                                    NextRank[row] = PreviousRank[row]

                if piece < last_piece:
                    T.copy(X[row, piece * chunk : piece * chunk + chunk], values)
                else:
                    T.copy(X[row, last_piece * chunk : vocab], values[0:tail])

                if shift >= 0:
                    if not aligned_rows:
                        with T.SimtVF(threads=threads):
                            selected_prefix = T.reinterpret(prefix_ub[0], "uint32") & T.uint32(
                                higher_mask
                            )
                            for encode_round in T.serial(encode_rounds):
                                for lane in T.Parallel(threads):
                                    for member in T.serial(4):
                                        local_col = (encode_round * threads + lane) * 4 + member
                                        col = piece * chunk + local_col
                                        digits[local_col] = T.cast(sentinel, "uint8")
                                        if col < vocab:  # noqa: SIM102
                                            if K[row] < vocab:
                                                bits = T.cast(
                                                    T.reinterpret(values[local_col], bit_dtype),
                                                    "uint32",
                                                )
                                                magnitude = bits & T.uint32(magnitude_mask)
                                                key = T.if_then_else(
                                                    magnitude > T.uint32(infinity_magnitude),
                                                    T.uint32(word_mask),
                                                    T.if_then_else(
                                                        magnitude == T.uint32(0),
                                                        T.uint32(sign_bit),
                                                        T.if_then_else(
                                                            (bits & T.uint32(sign_bit))
                                                            != T.uint32(0),
                                                            bits ^ T.uint32(word_mask),
                                                            bits ^ T.uint32(sign_bit),
                                                        ),
                                                    ),
                                                )
                                                if (key & T.uint32(higher_mask)) == selected_prefix:
                                                    if wide:
                                                        digits[local_col] = T.cast(
                                                            T.uint32(radix - 1)
                                                            - (
                                                                (key >> hist_shift)
                                                                & T.uint32(radix - 1)
                                                            ),
                                                            "uint8",
                                                        )
                                                    else:
                                                        digits[local_col] = T.cast(
                                                            (key >> hist_shift)
                                                            & T.uint32(radix - 1),
                                                            "uint8",
                                                        )

                    with T.SimdVF():
                        mask8 = T.simd.pset(8)
                        mask16 = T.simd.pset(16)
                        frequency0 = T.simd.alloc_var("uint16")
                        frequency0 = T.simd.vdup(T.uint16(0), "uint16", mask16)
                        if wide:
                            frequency1 = T.simd.alloc_var("uint16")
                            frequency1 = T.simd.vdup(T.uint16(0), "uint16", mask16)

                        if aligned_rows:
                            mask = T.simd.pset(bit_width)
                            rank_value = rank_ub[0]
                            selected_prefix = T.cast(
                                T.reinterpret(prefix_ub[0], "uint32") & T.uint32(higher_mask),
                                bit_dtype,
                            )
                            sign_vector = T.simd.vdup(T.cast(sign_bit, bit_dtype), bit_dtype, mask)
                            word_vector = T.simd.vdup(T.cast(word_mask, bit_dtype), bit_dtype, mask)
                            magnitude_vector = T.simd.vdup(
                                T.cast(magnitude_mask, bit_dtype), bit_dtype, mask
                            )
                            higher_vector = T.simd.vdup(
                                T.cast(higher_mask, bit_dtype), bit_dtype, mask
                            )
                            bit0_vector = T.simd.vdup(T.cast(bit0_mask, bit_dtype), bit_dtype, mask)
                            bit1_vector = T.simd.vdup(T.cast(bit1_mask, bit_dtype), bit_dtype, mask)
                            bit2_vector = T.simd.vdup(T.cast(bit2_mask, bit_dtype), bit_dtype, mask)
                            bit3_vector = T.simd.vdup(T.cast(bit3_mask, bit_dtype), bit_dtype, mask)
                            zero_vector = T.simd.vdup(T.cast(0, bit_dtype), bit_dtype, mask)
                            weight0_vector = T.simd.vdup(T.cast(1, bit_dtype), bit_dtype, mask)
                            weight1_vector = T.simd.vdup(T.cast(2, bit_dtype), bit_dtype, mask)
                            weight2_vector = T.simd.vdup(T.cast(4, bit_dtype), bit_dtype, mask)
                            weight3_vector = T.simd.vdup(T.cast(8, bit_dtype), bit_dtype, mask)
                            padding_vector = T.simd.vdup(
                                T.cast(byte_padding, bit_dtype), bit_dtype, mask
                            )
                            excluded_vector = T.simd.vdup(
                                T.cast(excluded_word, bit_dtype), bit_dtype, mask
                            )
                            if wide:
                                bit4_vector = T.simd.vdup(
                                    T.cast(bit4_mask, bit_dtype), bit_dtype, mask
                                )
                                bit5_vector = T.simd.vdup(
                                    T.cast(bit5_mask, bit_dtype), bit_dtype, mask
                                )
                                bit6_vector = T.simd.vdup(
                                    T.cast(bit6_mask, bit_dtype), bit_dtype, mask
                                )
                                bit7_vector = T.simd.vdup(
                                    T.cast(bit7_mask, bit_dtype), bit_dtype, mask
                                )
                                weight4_vector = T.simd.vdup(T.cast(16, bit_dtype), bit_dtype, mask)
                                weight5_vector = T.simd.vdup(T.cast(32, bit_dtype), bit_dtype, mask)
                                weight6_vector = T.simd.vdup(T.cast(64, bit_dtype), bit_dtype, mask)
                                weight7_vector = T.simd.vdup(
                                    T.cast(128, bit_dtype), bit_dtype, mask
                                )
                                reverse_vector = T.simd.vdup(
                                    T.cast(255, bit_dtype), bit_dtype, mask
                                )

                            if rank_value < vocab:
                                for vector_index in T.serial(chunk // vector_lanes):
                                    if (piece < last_piece) | (vector_index < tail // vector_lanes):
                                        original = T.reinterpret(
                                            T.simd.vld(
                                                values[vector_index * vector_lanes], dist="NORM"
                                            ),
                                            bit_vector_dtype,
                                        )
                                        magnitude = T.simd.vand(original, magnitude_vector, mask)
                                        positive_key = T.simd.vxor(original, sign_vector, mask)
                                        negative_key = T.simd.vxor(original, word_vector, mask)
                                        signed_key = T.simd.vsel(
                                            negative_key,
                                            positive_key,
                                            T.simd.vcmps(
                                                original, T.cast(sign_bit, bit_dtype), mask, op="ge"
                                            ),
                                        )
                                        zero_key = T.simd.vsel(
                                            sign_vector,
                                            signed_key,
                                            T.simd.vcmps(
                                                magnitude, T.cast(0, bit_dtype), mask, op="eq"
                                            ),
                                        )
                                        key = T.simd.vsel(
                                            word_vector,
                                            zero_key,
                                            T.simd.vcmps(
                                                magnitude,
                                                T.cast(infinity_magnitude, bit_dtype),
                                                mask,
                                                op="gt",
                                            ),
                                        )
                                        key_prefix = T.simd.vand(key, higher_vector, mask)
                                        selected_bit0 = T.simd.vand(key, bit0_vector, mask)
                                        digit0 = T.simd.vsel(
                                            zero_vector,
                                            weight0_vector,
                                            T.simd.vcmps(
                                                selected_bit0, T.cast(0, bit_dtype), mask, op="eq"
                                            ),
                                        )
                                        selected_bit1 = T.simd.vand(key, bit1_vector, mask)
                                        digit1 = T.simd.vsel(
                                            zero_vector,
                                            weight1_vector,
                                            T.simd.vcmps(
                                                selected_bit1, T.cast(0, bit_dtype), mask, op="eq"
                                            ),
                                        )
                                        selected_bit2 = T.simd.vand(key, bit2_vector, mask)
                                        digit2 = T.simd.vsel(
                                            zero_vector,
                                            weight2_vector,
                                            T.simd.vcmps(
                                                selected_bit2, T.cast(0, bit_dtype), mask, op="eq"
                                            ),
                                        )
                                        selected_bit3 = T.simd.vand(key, bit3_vector, mask)
                                        digit3 = T.simd.vsel(
                                            zero_vector,
                                            weight3_vector,
                                            T.simd.vcmps(
                                                selected_bit3, T.cast(0, bit_dtype), mask, op="eq"
                                            ),
                                        )
                                        digit01 = T.simd.vor(digit0, digit1, mask)
                                        digit23 = T.simd.vor(digit2, digit3, mask)
                                        low_digit = T.simd.vor(digit01, digit23, mask)
                                        if wide:
                                            selected_bit4 = T.simd.vand(key, bit4_vector, mask)
                                            digit4 = T.simd.vsel(
                                                zero_vector,
                                                weight4_vector,
                                                T.simd.vcmps(
                                                    selected_bit4,
                                                    T.cast(0, bit_dtype),
                                                    mask,
                                                    op="eq",
                                                ),
                                            )
                                            selected_bit5 = T.simd.vand(key, bit5_vector, mask)
                                            digit5 = T.simd.vsel(
                                                zero_vector,
                                                weight5_vector,
                                                T.simd.vcmps(
                                                    selected_bit5,
                                                    T.cast(0, bit_dtype),
                                                    mask,
                                                    op="eq",
                                                ),
                                            )
                                            selected_bit6 = T.simd.vand(key, bit6_vector, mask)
                                            digit6 = T.simd.vsel(
                                                zero_vector,
                                                weight6_vector,
                                                T.simd.vcmps(
                                                    selected_bit6,
                                                    T.cast(0, bit_dtype),
                                                    mask,
                                                    op="eq",
                                                ),
                                            )
                                            selected_bit7 = T.simd.vand(key, bit7_vector, mask)
                                            digit7 = T.simd.vsel(
                                                zero_vector,
                                                weight7_vector,
                                                T.simd.vcmps(
                                                    selected_bit7,
                                                    T.cast(0, bit_dtype),
                                                    mask,
                                                    op="eq",
                                                ),
                                            )
                                            digit45 = T.simd.vor(digit4, digit5, mask)
                                            digit67 = T.simd.vor(digit6, digit7, mask)
                                            high_digit = T.simd.vor(digit45, digit67, mask)
                                            full_digit = T.simd.vor(low_digit, high_digit, mask)
                                            descending_digit = T.simd.vxor(
                                                full_digit, reverse_vector, mask
                                            )
                                            padded_digit = T.simd.vor(
                                                descending_digit, padding_vector, mask
                                            )
                                        else:
                                            padded_digit = T.simd.vor(
                                                low_digit, padding_vector, mask
                                            )
                                        encoded = T.simd.vsel(
                                            padded_digit,
                                            excluded_vector,
                                            T.simd.vcmps(
                                                key_prefix, selected_prefix, mask, op="eq"
                                            ),
                                        )
                                        digit_bytes = T.reinterpret(encoded, "uint8x256")
                                        if wide:
                                            T.simd.chistv2(frequency0, digit_bytes, mask8, bin=0)
                                            T.simd.chistv2(frequency1, digit_bytes, mask8, bin=1)
                                        else:
                                            T.simd.dhistv2(frequency0, digit_bytes, mask8, bin=0)
                        else:
                            for histogram_round in T.serial(histogram_rounds):
                                digit_vector = T.simd.vld(
                                    digits[histogram_round * 256], dist="NORM"
                                )
                                if wide:
                                    T.simd.chistv2(frequency0, digit_vector, mask8, bin=0)
                                    T.simd.chistv2(frequency1, digit_vector, mask8, bin=1)
                                else:
                                    T.simd.dhistv2(frequency0, digit_vector, mask8, bin=0)
                        T.simd.vsts(frequencies[0], frequency0, mask16, dist="NORM_B16")
                        if wide:
                            T.simd.vsts(frequencies[128], frequency1, mask16, dist="NORM_B16")

                    with T.SimtVF(threads=threads):
                        for export_group in T.serial(export_groups):
                            for export_lane in T.Parallel(export_lanes):
                                export_digit = export_group * export_lanes + export_lane
                                Counts[row, piece, export_digit] = T.cast(
                                    frequencies[export_digit], "int32"
                                )
                else:
                    with T.SimdVF():
                        mask = T.simd.pset(bit_width)
                        threshold_key = T.reinterpret(prefix_ub[0], "uint32")
                        rank_value = rank_ub[0]
                        masked = T.simd.vdup(
                            T.cast(negative_infinity_bits, bit_dtype), bit_dtype, mask
                        )
                        filtered = T.simd.alloc_var(bit_dtype)
                        for vector_index in T.serial(chunk // vector_lanes):
                            if (piece < last_piece) | (vector_index < tail // vector_lanes):
                                original = T.reinterpret(
                                    T.simd.vld(values[vector_index * vector_lanes], dist="NORM"),
                                    bit_vector_dtype,
                                )
                                filtered = original
                                if rank_value < vocab:
                                    if threshold_key >= T.uint32(sign_bit):
                                        positive_cutoff = T.cast(
                                            threshold_key - T.uint32(sign_bit), bit_dtype
                                        )
                                        filtered = T.simd.vsel(
                                            original,
                                            masked,
                                            T.simd.vcmps(original, positive_cutoff, mask, op="ge"),
                                        )
                                        filtered = T.simd.vsel(
                                            filtered,
                                            masked,
                                            T.simd.vcmps(
                                                original,
                                                T.cast(infinity_magnitude, bit_dtype),
                                                mask,
                                                op="le",
                                            ),
                                        )
                                        if threshold_key == T.uint32(sign_bit):
                                            filtered = T.simd.vsel(
                                                original,
                                                filtered,
                                                T.simd.vcmps(
                                                    original,
                                                    T.cast(sign_bit, bit_dtype),
                                                    mask,
                                                    op="eq",
                                                ),
                                            )
                                    else:
                                        negative_cutoff = T.cast(
                                            T.uint32(word_mask) - threshold_key, bit_dtype
                                        )
                                        filtered = T.simd.vsel(
                                            original,
                                            masked,
                                            T.simd.vcmps(
                                                original,
                                                T.cast(infinity_magnitude, bit_dtype),
                                                mask,
                                                op="le",
                                            ),
                                        )
                                        filtered = T.simd.vsel(
                                            original,
                                            filtered,
                                            T.simd.vcmps(
                                                original, T.cast(sign_bit, bit_dtype), mask, op="ge"
                                            ),
                                        )
                                        filtered = T.simd.vsel(
                                            filtered,
                                            masked,
                                            T.simd.vcmps(original, negative_cutoff, mask, op="le"),
                                        )
                                T.simd.vsts(
                                    output_values[vector_index * vector_lanes],
                                    T.reinterpret(filtered, value_vector_dtype),
                                    mask,
                                    dist=store_dist,
                                )
                    if piece < last_piece:
                        T.copy(output_values, Y[row, piece * chunk : piece * chunk + chunk])
                    else:
                        T.copy(output_values[0:tail], Y[row, last_piece * chunk : vocab])

        return radix_stage

    def make_merge(shift):
        higher_mask = word_mask ^ ((1 << (shift + digit_bits)) - 1)

        @T.prim_func
        def merge_prefix(
            Counts: T.Tensor((batch, pieces, radix), "int32"),
            PreviousPrefix: T.Tensor((batch,), "int32"),
            PreviousRank: T.Tensor((batch,), "int32"),
            NextPrefix: T.Tensor((batch,), "int32"),
            NextRank: T.Tensor((batch,), "int32"),
        ):
            with T.Kernel(batch) as row:  # noqa: SIM117
                with T.SimtVF(threads=threads):
                    prefix = T.alloc_fragment((1,), "uint32")
                    residual = T.alloc_fragment((1,), "int32")
                    prefix[0] = T.reinterpret(PreviousPrefix[row], "uint32") & T.uint32(higher_mask)
                    residual[0] = PreviousRank[row]
                    if wide:
                        cumulative = T.alloc_fragment((radix,), "int32")
                        choice = T.alloc_reducer((1,), "int32", op="min")
                        choice_result = T.alloc_fragment((1,), "int32")
                        before = T.alloc_reducer((1,), "int32", op="sum")
                        before_result = T.alloc_fragment((1,), "int32")
                        for digit in T.Parallel(radix):
                            cumulative[digit] = 0
                        for piece in T.serial(pieces):
                            for digit in T.Parallel(radix):
                                cumulative[digit] = cumulative[digit] + Counts[row, piece, digit]
                        T.reducer_init(choice)
                        for digit in T.Parallel(radix):
                            T.reducer_update(
                                choice[0],
                                T.if_then_else(
                                    cumulative[digit] >= PreviousRank[row], digit, radix - 1
                                ),
                            )
                        T.finalize_reducer(choice, choice_result)
                        T.reducer_init(before)
                        for digit in T.Parallel(radix):
                            if digit + 1 == choice_result[0]:
                                T.reducer_update(before[0], cumulative[digit])
                        T.finalize_reducer(before, before_result)
                        prefix[0] = prefix[0] | (
                            T.cast(radix - 1 - choice_result[0], "uint32") << shift
                        )
                        residual[0] = residual[0] - before_result[0]
                    else:
                        found = T.alloc_fragment((1,), "int32")
                        count = T.alloc_reducer((1,), "int32", op="sum")
                        count_result = T.alloc_fragment((1,), "int32")
                        found[0] = 0
                        for order in T.serial(radix):
                            digit = radix - 1 - order
                            T.reducer_init(count)
                            for merge_round in T.serial(merge_rounds):
                                for lane in T.Parallel(threads):
                                    piece = merge_round * threads + lane
                                    if piece < pieces:
                                        T.reducer_update(count[0], Counts[row, piece, digit])
                            T.finalize_reducer(count, count_result)
                            if found[0] == 0:
                                if residual[0] > count_result[0]:
                                    residual[0] = residual[0] - count_result[0]
                                else:
                                    prefix[0] = prefix[0] | (T.cast(digit, "uint32") << shift)
                                    found[0] = 1
                    for lane in T.Parallel(1):
                        NextPrefix[row] = T.reinterpret(prefix[lane], "int32")
                        NextRank[row] = residual[lane]

        return merge_prefix

    @T.prim_func
    def apply_mask(
        X: T.Tensor((batch, vocab), dtype),
        K: T.Tensor((batch,), "int32"),
        Threshold: T.Tensor((batch,), "int32"),
        Y: T.Tensor((batch, vocab), dtype),
    ):
        with T.Kernel(output_programs) as block:  # noqa: SIM117
            with T.SimtVF(threads=threads):
                for item in T.Parallel(output_tile // pack):
                    for member in T.serial(pack):
                        flat = block * output_tile + item * pack + member
                        if flat < total_elements:
                            row = flat // vocab
                            col = flat % vocab
                            if K[row] >= vocab:
                                Y[row, col] = X[row, col]
                            else:
                                bits = T.cast(T.reinterpret(X[row, col], bit_dtype), "uint32")
                                magnitude = bits & T.uint32(magnitude_mask)
                                key = T.if_then_else(
                                    magnitude == T.uint32(0),
                                    T.uint32(sign_bit),
                                    T.if_then_else(
                                        (bits & T.uint32(sign_bit)) != T.uint32(0),
                                        bits ^ T.uint32(word_mask),
                                        bits ^ T.uint32(sign_bit),
                                    ),
                                )
                                threshold_key = T.reinterpret(Threshold[row], "uint32")
                                if (magnitude <= T.uint32(infinity_magnitude)) & (
                                    key >= threshold_key
                                ):
                                    Y[row, col] = X[row, col]
                                else:
                                    Y[row, col] = T.cast(float("-inf"), dtype)

    shifts = list(range(bit_width - digit_bits, -1, -digit_bits))
    kernels = []
    stages = []
    mask_kernel = None
    if fused_path:
        previous_shift = -1
        for shift in shifts:
            kernels.append(
                tilelang.compile(make_stage(shift, previous_shift), target="ascend", out_idx=[])
            )
            previous_shift = shift
        kernels.append(
            tilelang.compile(make_stage(-1, previous_shift), target="ascend", out_idx=[])
        )
    else:
        for shift in shifts:
            histogram = tilelang.compile(make_stage(shift, -1), target="ascend", out_idx=[])
            merge = tilelang.compile(make_merge(shift), target="ascend", out_idx=[])
            stages.append((histogram, merge))
        if aligned_rows:
            mask_kernel = tilelang.compile(make_stage(-1, -1), target="ascend", out_idx=[])
        else:
            mask_kernel = tilelang.compile(apply_mask, target="ascend", out_idx=[])

    def run(logits, k):
        counts_slots = [
            torch.empty((batch, pieces, radix), dtype=torch.int32, device=logits.device),
            torch.empty((batch, pieces, radix), dtype=torch.int32, device=logits.device),
        ]
        prefix_slots = [
            torch.empty((batch,), dtype=torch.int32, device=logits.device),
            torch.empty((batch,), dtype=torch.int32, device=logits.device),
        ]
        rank_slots = [
            torch.empty((batch,), dtype=torch.int32, device=logits.device),
            torch.empty((batch,), dtype=torch.int32, device=logits.device),
        ]
        output = torch.empty_like(logits)
        previous_counts = counts_slots[1]
        prefix = k
        residual = k
        if fused_path:
            for index, kernel in enumerate(kernels):
                slot = index % 2
                kernel(
                    logits,
                    k,
                    previous_counts,
                    prefix,
                    residual,
                    counts_slots[slot],
                    prefix_slots[slot],
                    rank_slots[slot],
                    output,
                )
                previous_counts = counts_slots[slot]
                prefix = prefix_slots[slot]
                residual = rank_slots[slot]
        else:
            for index, (histogram, merge) in enumerate(stages):
                slot = index % 2
                histogram(
                    logits,
                    k,
                    previous_counts,
                    prefix,
                    residual,
                    counts_slots[slot],
                    prefix_slots[slot],
                    rank_slots[slot],
                    output,
                )
                merge(counts_slots[slot], prefix, residual, prefix_slots[slot], rank_slots[slot])
                previous_counts = counts_slots[slot]
                prefix = prefix_slots[slot]
                residual = rank_slots[slot]
            if aligned_rows:
                unused_slot = len(stages) % 2
                mask_kernel(
                    logits,
                    k,
                    previous_counts,
                    prefix,
                    residual,
                    counts_slots[unused_slot],
                    prefix_slots[unused_slot],
                    rank_slots[unused_slot],
                    output,
                )
            else:
                mask_kernel(logits, k, prefix, output)
        return (output,)

    return run


class AscendTopKMaskKernel(Kernel, TopKMaskFwdInterface):
    """Prior controller-selected TopKMask implementation, adapted to the repo ABI."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: SamplingCall):
        return call, lambda: cls(call)

    def __init__(self, call: SamplingCall):
        super().__init__()
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("TopKMask supports float16, bfloat16 and float32")
        self.call = call
        self.dtype = call.dtype
        self.program = build(
            {
                "shape": {"logits": [call.batch, call.vocab]},
                "dtype": {"logits": str(call.dtype).split(".")[-1]},
            }
        )

    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._require_devices(logits=logits, k=k)
        if logits.shape != (self.call.batch, self.call.vocab):
            raise ValueError("logits shape differs from the compiled call")
        if logits.dtype != self.dtype or k.dtype != torch.int32:
            raise TypeError("input dtype differs from the compiled call")
        return self.program(logits.contiguous(), k.contiguous())[0]


register_kernel_type("TopKMaskFwdOp", "ascend", AscendTopKMaskKernel)
