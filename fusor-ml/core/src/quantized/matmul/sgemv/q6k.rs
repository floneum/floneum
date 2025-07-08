use crate::{
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L5564
pub(crate) fn q6k_sgemv(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
) {
    let dtype = op.input_datatype;
    let workgroup_index = generic_kernel.workgroup_index();
    let subgroup_index = generic_kernel.subgroup_index();
    let subgroup_local_index = generic_kernel.subgroup_local_index();
    let elements_per_block = op.elements_per_block();

    let mut kernel = String::new();

    // Find the reduce size in blocks rounded up
    writeln!(
        &mut kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // In index of the single element in the vector we are multiplying against
    writeln!(&mut kernel, "let workgroup_offset = {workgroup_index}.x;").unwrap();
    writeln!(
        &mut kernel,
        "let row = 2 * workgroup_offset + {subgroup_index};"
    )
    .unwrap();

    writeln!(&mut kernel, "let block_offset = row * k_block_size;").unwrap();

    writeln!(&mut kernel, "let thread_id = {subgroup_local_index} / 2;").unwrap();
    writeln!(
        &mut kernel,
        "let thread_local_id = {subgroup_local_index} % 2;"
    )
    .unwrap();
    writeln!(&mut kernel, "let half_subgroup_id = thread_id / 8;").unwrap();
    writeln!(&mut kernel, "let half_subgroup_local_id = thread_id % 8;").unwrap();
    const CHUNKS_PER_STRIPE: usize = 4;
    writeln!(
        &mut kernel,
        "let stripe_offset = half_subgroup_local_id * {CHUNKS_PER_STRIPE};"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let scale_index_offset = half_subgroup_id * 2;"
    )
    .unwrap();

    writeln!(&mut kernel, "let scale_pair_offset = stripe_offset / 16;").unwrap();

    // Every half subgroup handles 128 normal values
    writeln!(
        &mut kernel,
        "let y_offset = 128 * half_subgroup_id + stripe_offset;"
    )
    .unwrap();
    // Every half subgroup handles 64 lower values (half bytes)
    writeln!(
        &mut kernel,
        "let q_offset_l = 16 * half_subgroup_id + stripe_offset / 4;"
    )
    .unwrap();
    // Every half subgroup handles 32 upper values (quarter bytes)
    writeln!(
        &mut kernel,
        "let q_offset_h = 8 * half_subgroup_id + stripe_offset / 4;"
    )
    .unwrap();

    writeln!(&mut kernel, "var sum = 0.0;").unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        &mut kernel,
        "for (var i = thread_local_id; i < k_block_size; i += 2) {{"
    )
    .unwrap();
    {
        writeln!(&mut kernel, "let local_block_offset = i + block_offset;").unwrap();
        writeln!(&mut kernel, "let low_offset_1 = q_offset_l;").unwrap();
        writeln!(
            &mut kernel,
            "let low_bytes_1 = unpack4xU8({input_b}[local_block_offset].data_low_bits[low_offset_1]);"
        )
        .unwrap();
        writeln!(&mut kernel, "let low_offset_2 = q_offset_l + 8;").unwrap();
        writeln!(&mut kernel, "let low_bytes_2 = unpack4xU8({input_b}[local_block_offset].data_low_bits[low_offset_2]);").unwrap();
        writeln!(&mut kernel, "let high_offset = q_offset_h;").unwrap();
        writeln!(
            &mut kernel,
            "let high_bytes = unpack4xU8({input_b}[local_block_offset].data_high_bits[high_offset]);"
        )
        .unwrap();
        writeln!(&mut kernel, "let scale_offset = scale_index_offset;").unwrap();
        writeln!(
            &mut kernel,
            "let scale_chunk_1 = unpack4xI8({input_b}[local_block_offset].scales[scale_offset]);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let scale_chunk_2 = unpack4xI8({input_b}[local_block_offset].scales[scale_offset + 1]);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let scales = vec4({dtype}(scale_chunk_1[scale_pair_offset]), {dtype}(scale_chunk_1[2 + scale_pair_offset]), {dtype}(scale_chunk_2[scale_pair_offset]), {dtype}(scale_chunk_2[2 + scale_pair_offset]));"
        )
        .unwrap();

        writeln!(
            &mut kernel,
            "let vector_offset = i * {elements_per_block} + y_offset;"
        )
        .unwrap();

        writeln!(
            &mut kernel,
            "let scale = {dtype}({input_b}[local_block_offset].scale);"
        )
        .unwrap();

        writeln!(&mut kernel, "var sums = vec4f();").unwrap();
        writeln!(&mut kernel, "for (var j = 0u; j < 4u; j += 1u) {{").unwrap();
        {
            let first_four_bytes = 0b00001111u8;
            let first_two_bytes = 0b00000011u8;
            let second_two_bytes = 0b00001100u8;
            let third_two_bytes = 0b00110000u8;
            let fourth_two_bytes = 0b11000000u8;
            writeln!(&mut kernel, "sums[0] += {input_a}[j + vector_offset +  0] * {dtype}(i32((low_bytes_1[j] & {first_four_bytes}) | ((high_bytes[j] & {first_two_bytes})  << 4)) - 32);").unwrap();
            writeln!(&mut kernel, "sums[1] += {input_a}[j + vector_offset + 32] * {dtype}(i32((low_bytes_2[j] & {first_four_bytes}) | ((high_bytes[j] & {second_two_bytes}) << 2)) - 32);").unwrap();
            writeln!(&mut kernel, "sums[2] += {input_a}[j + vector_offset + 64] * {dtype}(i32((low_bytes_1[j]                 >> 4) | ((high_bytes[j] & {third_two_bytes})  << 0)) - 32);").unwrap();
            writeln!(&mut kernel, "sums[3] += {input_a}[j + vector_offset + 96] * {dtype}(i32((low_bytes_2[j]                 >> 4) | ((high_bytes[j] & {fourth_two_bytes}) >> 2)) - 32);").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();
        writeln!(&mut kernel, "sum += scale * dot(sums, scales);").unwrap();
    }
    writeln!(&mut kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(&mut kernel, "sum = subgroupAdd(sum);").unwrap();

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(&mut kernel, "if {subgroup_local_index} != 0u {{ return; }}").unwrap();

    // Write the output to the output tensor if this is the first thread in the workgroup
    write!(&mut kernel, "{output}[").unwrap();
    output.strided_index(&mut kernel, ["0".to_string(), "row".to_string()]);
    writeln!(&mut kernel, "] = sum;",).unwrap();

    generic_kernel.push_body(&kernel);
}
