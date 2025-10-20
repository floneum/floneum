use crate::{
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
    shift_right_scale,
    util::{maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type},
};
use std::fmt::Write;

pub(crate) const Q4K_SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk each thread will process at a time
const SUBGROUP_COUNT: u32 = 2;

const MASK1: u32 = 0b0011111100111111;
const MASK2: u32 = 0b0000111100001111;
const MASK3: u32 = 0b1100000011000000;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L5311
#[allow(clippy::too_many_arguments)]
pub(crate) fn q4k_sgemv(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    _m_size: &str,
    k_size: &str,
) {
    let dtype = op.input_datatype;
    let global_id = kernel.global_id();
    let workgroup_index = kernel.workgroup_index();
    let subgroup_index = kernel.subgroup_index();
    let subgroup_local_index = kernel.subgroup_local_index();
    let elements_per_block = op.elements_per_block();

    // Handle batch dimensions
    writeln!(kernel, "let batch_idx = {global_id}.z;").unwrap();
    // Handle M dimension - each workgroup handles one M value
    writeln!(kernel, "let m_idx = {global_id}.y;").unwrap();

    // Find the reduce size in blocks rounded up
    writeln!(
        kernel,
        "let k_block_size = {k_size} / {elements_per_block};"
    )
    .unwrap();

    // In index of the single element in the vector we are multiplying against
    writeln!(kernel, "let workgroup_offset = {workgroup_index}.x;").unwrap();
    writeln!(
        kernel,
        "let row = (workgroup_offset * {SUBGROUP_COUNT} + {subgroup_index}) * {Q4K_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} >> 3;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} & 7;").unwrap();
    writeln!(kernel, "let half_subgroup_id = thread_local_id >> 2;").unwrap();
    writeln!(kernel, "let half_subgroup_local_id = thread_local_id & 3;").unwrap();

    writeln!(kernel, "let block_offset = row * k_block_size;").unwrap();
    writeln!(kernel, "var vector_offset = thread_id * {elements_per_block} + half_subgroup_id * 64 + half_subgroup_local_id * 8;").unwrap();

    let sum_storage_type = maybe_vec_storage_type(Q4K_SGEMV_CHUNK_SIZE, dtype);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    writeln!(kernel, "var cached_a_low_values = array<{dtype}, 16>();",).unwrap();
    writeln!(kernel, "var cached_a_high_values = array<{dtype}, 16>();",).unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_id; i < k_block_size; i += 4) {{"
    )
    .unwrap();
    {
        // Keep track of the sum of each scale chunk of the vector for the offset calculation later
        writeln!(kernel, "var vector_sum = vec4<{dtype}>();").unwrap();

        // First load the values of a into the cache
        for j in 0..8 {
            // Load all 4 values using strided indexing
            write!(kernel, "let a_val_{j}_0 = {input_a}[").unwrap();
            input_a.strided_index(
                kernel,
                vec![
                    "batch_idx".to_string(),
                    "m_idx".to_string(),
                    format!("vector_offset + {j} + 0"),
                ],
            );
            writeln!(kernel, "];").unwrap();

            write!(kernel, "let a_val_{j}_1 = {input_a}[").unwrap();
            input_a.strided_index(
                kernel,
                vec![
                    "batch_idx".to_string(),
                    "m_idx".to_string(),
                    format!("vector_offset + {j} + 32"),
                ],
            );
            writeln!(kernel, "];").unwrap();

            write!(kernel, "let a_val_{j}_2 = {input_a}[").unwrap();
            input_a.strided_index(
                kernel,
                vec![
                    "batch_idx".to_string(),
                    "m_idx".to_string(),
                    format!("vector_offset + {j} + 128"),
                ],
            );
            writeln!(kernel, "];").unwrap();

            write!(kernel, "let a_val_{j}_3 = {input_a}[").unwrap();
            input_a.strided_index(
                kernel,
                vec![
                    "batch_idx".to_string(),
                    "m_idx".to_string(),
                    format!("vector_offset + {j} + 160"),
                ],
            );
            writeln!(kernel, "];").unwrap();

            writeln!(
                kernel,
                "let vec_{j} = vec4(a_val_{j}_0, a_val_{j}_1, a_val_{j}_2, a_val_{j}_3);"
            )
            .unwrap();
            writeln!(kernel, "vector_sum += vec_{j};").unwrap();
            writeln!(kernel, "cached_a_low_values[{j} + 0] = vec_{j}.x;").unwrap();
            writeln!(kernel, "cached_a_low_values[{j} + 8] = vec_{j}.y;").unwrap();
            writeln!(kernel, "cached_a_high_values[{j} + 0] = vec_{j}.z;").unwrap();
            writeln!(kernel, "cached_a_high_values[{j} + 8] = vec_{j}.w;").unwrap();
        }

        // Find the block value offsets
        writeln!(kernel, "let scale_offset = half_subgroup_id;").unwrap();
        writeln!(
            kernel,
            "let data_offset = half_subgroup_id * 8u + half_subgroup_local_id * 2u;"
        )
        .unwrap();

        writeln!(kernel, "var local_block_offset = block_offset + i;").unwrap();

        for offset in 0..Q4K_SGEMV_CHUNK_SIZE {
            writeln!(kernel, "{{").unwrap();
            // Fetch and unpack the two sets of values from the cache
            writeln!(kernel, "let first_values_offset = data_offset;").unwrap();
            writeln!(kernel, "let second_values_offset = data_offset + 16u;").unwrap();

            // Keep track of the sum of each chunk
            writeln!(kernel, "var first_sums = vec4<{dtype}>();").unwrap();
            writeln!(kernel, "var second_sums = vec4<{dtype}>();").unwrap();

            // Perform the dot product of the values and scales
            for j in 0..2 {
                for (sum, cache, values) in [
                    ("first_sums", "cached_a_low_values", "first_values_offset"),
                    (
                        "second_sums",
                        "cached_a_high_values",
                        "second_values_offset",
                    ),
                ] {
                    // Note: We add the values with a mask **without** shifting them
                    // this means the sums in the first_sums and second_sums
                    // will be scaled by different values. We correct this below
                    // by multiplying by the floating point values that correspond to the
                    // bit shifts.
                    writeln!(
                        kernel,
                        "let value_u32_{values}_{j} = {input_b}[local_block_offset].data[{values} + {j}];"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "let first_four_values_{values}_{j} = vec4({cache}[{j}*4 + 0], {cache}[{j}*4 + 1], {cache}[{j}*4 + 2], {cache}[{j}*4 + 3]);"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "let second_four_values_{values}_{j} = vec4({cache}[{j}*4 + 8], {cache}[{j}*4 + 9], {cache}[{j}*4 + 10], {cache}[{j}*4 + 11]);"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "{sum} += vec4<{dtype}>(first_four_values_{values}_{j}.x * {dtype}(value_u32_{values}_{j} & 0x000F), first_four_values_{values}_{j}.y * {dtype}(value_u32_{values}_{j} & 0x0F00), second_four_values_{values}_{j}.x * {dtype}(value_u32_{values}_{j} & 0x00F0), second_four_values_{values}_{j}.y * {dtype}(value_u32_{values}_{j} & 0xF000));"
                    )
                    .unwrap();
                    let shift_right_16 = shift_right_scale(16);
                    writeln!(
                        kernel,
                        "{sum} += vec4<{dtype}>(first_four_values_{values}_{j}.z * {dtype}(value_u32_{values}_{j} & 0x000F0000), first_four_values_{values}_{j}.w * {dtype}(value_u32_{values}_{j} & 0x0F000000), second_four_values_{values}_{j}.z * {dtype}(value_u32_{values}_{j} & 0x00F00000), second_four_values_{values}_{j}.w * {dtype}(value_u32_{values}_{j} & 0xF0000000)) * {shift_right_16};"
                    )
                    .unwrap();
                }
            }

            // Load the block scale and min
            writeln!(
                kernel,
                "let block_scale = {input_b}[local_block_offset].scale;"
            )
            .unwrap();
            writeln!(kernel, "let block_min = {input_b}[local_block_offset].min;").unwrap();
            // Load 8 scales into a cache
            writeln!(
                kernel,
                "let first_32_scale_bits = {input_b}[local_block_offset].scales[0] >> (16 * scale_offset);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let second_32_scale_bits = {input_b}[local_block_offset].scales[1] >> (16 * scale_offset);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let third_32_scale_bits = {input_b}[local_block_offset].scales[2] >> (16 * scale_offset);"
            )
            .unwrap();
            // Extract the scales from the bits into cached_scales
            writeln!(
                kernel,
                "let first_two_scales = first_32_scale_bits & {MASK1};"
            )
            .unwrap();
            writeln!(
                kernel,
                "let second_two_scales = second_32_scale_bits & {MASK1};"
            )
            .unwrap();

            writeln!(kernel, "let third_two_scales = ((third_32_scale_bits >> 0) & {MASK2}) | ((first_32_scale_bits & {MASK3}) >> 2);").unwrap();
            writeln!(kernel, "let fourth_two_scales = ((third_32_scale_bits >> 4) & {MASK2}) | ((second_32_scale_bits & {MASK3}) >> 2);").unwrap();

            writeln!(
                kernel,
                "let odd_scales_unpacked = vec4<{dtype}>(unpack4xU8(first_two_scales | (third_two_scales << 16)));"
            )
            .unwrap();
            writeln!(
                kernel,
                "let even_scales_unpacked = vec4<{dtype}>(unpack4xU8(second_two_scales | (fourth_two_scales << 16)));"
            )
            .unwrap();

            // Add the sums to the total sum
            let indexed_sum = maybe_vec_storage_index(Q4K_SGEMV_CHUNK_SIZE, "sum", offset);
            // *_sums[0] needs to be shifted by 0 bits
            // *_sums[1] needs to be shifted by 8 bits
            // *_sums[2] needs to be shifted by 4 bits
            // *_sums[3] needs to be shifted by 12 bits
            let shift_right_8 = shift_right_scale(8);
            let shift_right_4 = shift_right_scale(4);
            writeln!(
                kernel,
                "let small_shift_sums = vec4(first_sums[0], first_sums[2], second_sums[0], second_sums[2]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let large_shift_sums = vec4(first_sums[1], first_sums[3], second_sums[1], second_sums[3]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let shift_4 = vec4(1, {shift_right_4}, 1, {shift_right_4});"
            )
            .unwrap();
            writeln!(
                kernel,
                r#"{indexed_sum} += {dtype}(block_scale) * dot((small_shift_sums + {shift_right_8} * large_shift_sums) * odd_scales_unpacked, shift_4) -
                                                            {dtype}(block_min) * dot(vector_sum, even_scales_unpacked);"#
            )
            .unwrap();
            // Move forward the block offset by one row
            writeln!(kernel, "local_block_offset += k_block_size;").unwrap();
            writeln!(kernel, "}}").unwrap();
        }

        // move forward the vector offset
        writeln!(kernel, "vector_offset += 4 * {elements_per_block};").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q4K_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    for offset in 0..Q4K_SGEMV_CHUNK_SIZE {
        // If this is not the first simd thread in the workgroup, we can return early
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        {
            // Write the output to the output tensor if this is the first thread in the workgroup
            write!(kernel, "{output}[").unwrap();
            let index = format!("row + {offset}");
            let output_indices = vec!["batch_idx".to_string(), "m_idx".to_string(), index];
            output.strided_index(kernel, output_indices);
            let indexed = maybe_vec_storage_index(Q4K_SGEMV_CHUNK_SIZE, "sum", offset);
            writeln!(kernel, "] = {indexed};").unwrap();
        }
        writeln!(kernel, "}}").unwrap();
    }
}
