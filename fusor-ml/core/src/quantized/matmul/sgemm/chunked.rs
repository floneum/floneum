use crate::{
    DataTypeEnum, dequantize_mat4x4_block, dequantize_mat4x4_block_count,
    mir::{
        globals::{KernelGlobalSpace, MatrixType},
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

pub(crate) const SGEMM_MATRIX_SIZE: u32 = 16; // This is the size of the chunk we will dot at a time
pub(crate) const SGEMM_SUBGROUP_THERADS_PER_BLOCK: u32 = 2;

/// How many 16 wide blocks do we load from input_a and input_b into the cache every step
pub(crate) const SGEMM_INPUT_K_CHUNKS: u32 = 2;
pub(crate) const SGEMM_INPUT_K_ELEMENTS: u32 = SGEMM_INPUT_K_CHUNKS * SGEMM_MATRIX_SIZE;
/// How deep is the cache for input_a
pub(crate) const SGEMM_INPUT_M_ELEMENTS: u32 = 16;
/// How deep is the cache for input_b
pub(crate) const SGEMM_INPUT_N_ELEMENTS: u32 = 16;

#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_sgemm(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _: &str,
    _: &str,
    k_size: &str,
) {
    let workgroup_id = kernel.workgroup_index();
    let global_id = kernel.global_id();
    let workgroup_local_index = kernel.workgroup_local_index();
    let elements_per_block = op.elements_per_block();
    let dtype = op.input_datatype;

    let sub_chunks = dequantize_mat4x4_block_count(op.matrix.datatype);

    assert_eq!(elements_per_block % SGEMM_INPUT_K_ELEMENTS, 0);
    assert_eq!(sub_chunks % SGEMM_INPUT_K_CHUNKS as usize, 0);
    assert_eq!(SGEMM_INPUT_K_CHUNKS % SGEMM_SUBGROUP_THERADS_PER_BLOCK, 0);
    assert_eq!(SGEMM_INPUT_M_ELEMENTS % 4, 0);
    assert_eq!(SGEMM_INPUT_N_ELEMENTS % 4, 0);
    assert_eq!(SGEMM_INPUT_K_ELEMENTS % 4, 0);

    let cache_a_size = (SGEMM_INPUT_M_ELEMENTS / 4) * (SGEMM_INPUT_K_ELEMENTS / 4);
    let cache_a = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        MatrixType::new(["4".into(), "4".into()], dtype),
        cache_a_size.to_string(),
    );
    let cache_b_size = (SGEMM_INPUT_K_ELEMENTS / 4) * (SGEMM_INPUT_N_ELEMENTS / 4);
    let cache_b = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        MatrixType::new(["4".into(), "4".into()], dtype),
        cache_b_size.to_string(),
    );

    // Find the block index this workgroup is responsible for
    writeln!(kernel, "let x = {workgroup_id}.x;").unwrap();
    writeln!(kernel, "let y = {workgroup_id}.y;").unwrap();

    // Handle batch dimensions
    writeln!(kernel, "var block_batch = {workgroup_id}.z;").unwrap();

    // Decompose the batch index for higher-dimensional tensors
    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch = block_batch / {shape};").unwrap();
    }

    // Each thread is responsible for a 4x4 sub-block of the workgroup's block
    writeln!(kernel, "var acc = mat4x4<{dtype}>();").unwrap();

    // We subdivide the x dimension into pairs of threads in the same subgroup that will
    // collaboratively load one quantized block into shared memory
    writeln!(
        kernel,
        "let pair_index = {workgroup_local_index} / {SGEMM_SUBGROUP_THERADS_PER_BLOCK};"
    )
    .unwrap();
    writeln!(
        kernel,
        "let pair_local_index = {workgroup_local_index} % {SGEMM_SUBGROUP_THERADS_PER_BLOCK};"
    )
    .unwrap();

    // How many blocks do we have to process in the k dim
    writeln!(kernel, "let k_chunk_size = {k_size} / {SGEMM_MATRIX_SIZE};").unwrap();
    writeln!(
        kernel,
        "let k_block_size = {k_size} / {elements_per_block};"
    )
    .unwrap();

    // This threads b_input offset in blocks
    writeln!(
        kernel,
        "let b_block_offset = (y * {SGEMM_INPUT_N_ELEMENTS} + pair_index) * k_block_size;"
    )
    .unwrap();

    let chunks_per_block = elements_per_block / SGEMM_MATRIX_SIZE;

    // Calculate one block sized group
    writeln!(
        kernel,
        "for (var k_start = 0u; k_start < k_chunk_size; k_start += {SGEMM_SUBGROUP_THERADS_PER_BLOCK}u) {{"
    )
    .unwrap();
    {
        // This is seperated from k_start above to avoid divergance checks within each subgroup
        writeln!(kernel, "let k = k_start + pair_local_index;").unwrap();
        writeln!(
            kernel,
            "for (var i = 0u; i < {}u; i += 1u) {{",
            SGEMM_SUBGROUP_THERADS_PER_BLOCK
        )
        .unwrap();
        {
            // Load the b block into the cache
            writeln!(kernel, "let b_block_index = k / {chunks_per_block};").unwrap();
            writeln!(kernel, "let b_index_within_block = k % {chunks_per_block};").unwrap();
            let y_stride = SGEMM_INPUT_K_ELEMENTS / 4;
            let block_half = (SGEMM_INPUT_N_ELEMENTS / 4) / 2;
            dequantize_mat4x4_block(
                kernel,
                op.matrix.datatype,
                "b_index_within_block",
                format!(
                    "{input_b}[b_block_index + b_block_offset + i * {} * k_block_size]",
                    SGEMM_INPUT_N_ELEMENTS / 2
                ),
                DataTypeEnum::F32,
                |data, kernel| {
                    writeln!(kernel, "let b_values = {data};").unwrap();
                    writeln!(kernel, "for (var index = 0u; index < 4u; index += 1u) {{").unwrap();
                    writeln!(
                        kernel,
                        "{cache_b}[pair_local_index * 4 + ((pair_index / 4) + i * {block_half}) * {y_stride} + index][pair_index % 4] = b_values[index];"
                    )
                    .unwrap();
                    writeln!(kernel, "}}").unwrap();
                },
            );

            // Load the a block into the cache
            writeln!(
                kernel,
                "for (var index = 0u; index < {SGEMM_MATRIX_SIZE}u; index += 1u) {{"
            )
            .unwrap();
            {
                let y_stride = SGEMM_INPUT_K_ELEMENTS / 4;
                let block_half = (SGEMM_INPUT_M_ELEMENTS / 4) / 2;
                let mut indices = vec![];
                // Add batch indices first
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("block_batch_{dim}"));
                }
                // Then add M and K indices
                indices.push(format!(
                    "x * {SGEMM_INPUT_M_ELEMENTS} + pair_index + i * {}",
                    SGEMM_INPUT_M_ELEMENTS / 2
                ));
                indices.push(format!("k*{SGEMM_MATRIX_SIZE} + index"));
                writeln!(
                    kernel,
                    "let chunk_index = pair_local_index * 4 + ((pair_index / 4) + i * {block_half})*{y_stride} + index / 4;"
                )
                .unwrap();
                writeln!(kernel, "let row_index = pair_index % 4;").unwrap();
                writeln!(kernel, "let col_index = index % 4;").unwrap();
                input_a
                    .check_bounds(
                        kernel,
                        indices.iter().cloned(),
                        |kernel: &mut GenericKernel| {
                            write!(
                                kernel,
                                "{cache_a}[chunk_index][row_index][col_index] = {input_a}["
                            )?;
                            input_a.strided_index(kernel, indices.iter().cloned());
                            write!(kernel, "];")?;
                            std::fmt::Result::Ok(())
                        },
                    )
                    .unwrap();
                writeln!(kernel, "else {{").unwrap();
                {
                    writeln!(
                        kernel,
                        "{cache_a}[chunk_index][row_index][col_index] = 0.0;"
                    )
                    .unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Make sure the caches are ready
        writeln!(kernel, "workgroupBarrier();").unwrap();

        // Now that the items are in cache, do the matrix multiplication
        writeln!(
            kernel,
            "let n_workgroup_offset = {} * ({workgroup_local_index} / {});",
            SGEMM_INPUT_K_ELEMENTS / 4,
            SGEMM_INPUT_N_ELEMENTS / 4
        )
        .unwrap();
        writeln!(
            kernel,
            "let m_workgroup_offset = {} * ({workgroup_local_index} % {});",
            SGEMM_INPUT_K_ELEMENTS / 4,
            SGEMM_INPUT_M_ELEMENTS / 4
        )
        .unwrap();
        writeln!(
            kernel,
            "for (var index = 0u; index < {}u; index += 1u) {{",
            SGEMM_INPUT_K_ELEMENTS / 4
        )
        .unwrap();
        {
            // Load a 4x4 from cache_a
            writeln!(
                kernel,
                "let a_values = {cache_a}[index + m_workgroup_offset];"
            )
            .unwrap();
            // Load a 4x4 from cache_b
            writeln!(
                kernel,
                "let b_values = {cache_b}[index + n_workgroup_offset];"
            )
            .unwrap();
            writeln!(kernel, "acc = acc + transpose(a_values) * b_values;").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Make sure all threads are done reading from the cache before we overwrite it
        writeln!(kernel, "workgroupBarrier();").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    write_acc_back(kernel, output, global_id).unwrap();
}

fn write_acc_back(
    kernel: &mut GenericKernel,
    output: &TensorInput,
    global_id: &str,
) -> std::fmt::Result {
    writeln!(
        kernel,
        "for (var y_offset = 0u; y_offset < 4u; y_offset += 1u) {{"
    )?;
    writeln!(
        kernel,
        "for (var x_offset = 0u; x_offset < 4u; x_offset += 1u) {{"
    )?;
    // Then write the result

    let mut output_indices = vec![];
    // Add batch indices first
    for dim in (0..output.rank()).rev().skip(2) {
        output_indices.push(format!("block_batch_{dim}"));
    }
    // Then add M and N indices
    output_indices.push(format!("{global_id}.x * 4u + x_offset"));
    output_indices.push(format!("{global_id}.y * 4u + y_offset"));
    output.check_bounds(
        kernel,
        output_indices.iter().cloned(),
        |kernel: &mut GenericKernel| {
            write!(kernel, "let output_index = ")?;
            output.strided_index(kernel, output_indices.iter().cloned());
            writeln!(kernel, ";")?;
            writeln!(kernel, "{output}[output_index] = acc[y_offset][x_offset];")?;
            writeln!(kernel, "}}")?;
            writeln!(kernel, "}}")?;
            Ok(())
        },
    )?;
    Ok(())
}
