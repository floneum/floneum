use crate::{
    DataTypeEnum, dequantize_mat4x4_block, dequantize_mat4x4_block_count,
    mir::{
        globals::{KernelGlobalSpace, MatrixType},
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

/// Size of the chunk we will dot at a time
const MATRIX_ELEMENTS: u32 = 16;

/// Configuration for chunked SGEMM algorithm
#[derive(Debug, Clone, Copy)]
pub struct ChunkedSgemmConfig {
    /// Number of subgroup threads per block
    pub subgroup_threads_per_block: u32,
    /// How many matrix_size-wide blocks we load from input_a and input_b into the cache every step
    pub input_k_chunks: u32,
    /// How deep is the cache for input_a (M dimension, must be divisible by 4)
    pub input_m_elements: u32,
    /// How deep is the cache for input_b (N dimension, must be divisible by 4)
    pub input_n_elements: u32,
}

impl ChunkedSgemmConfig {
    /// Default configuration
    pub const fn default() -> Self {
        Self {
            subgroup_threads_per_block: 2,
            input_k_chunks: 4,
            input_m_elements: 4 * 16,
            input_n_elements: 4 * 16,
        }
    }

    /// Compute the total K elements loaded per step
    pub const fn input_k_elements(&self) -> u32 {
        self.input_k_chunks * MATRIX_ELEMENTS
    }

    /// Validate configuration parameters
    pub fn validate(&self, elements_per_block: u32, _: usize) -> Result<(), String> {
        let input_k_elements = self.input_k_elements();

        // Check that input_k_elements is a multiple of matrix_size
        if !input_k_elements.is_multiple_of(MATRIX_ELEMENTS) {
            return Err(format!(
                "input_k_elements ({}) must be divisible by matrix_size ({})",
                input_k_elements, MATRIX_ELEMENTS
            ));
        }

        // Check that matrix_size divides elements_per_block evenly
        if !elements_per_block.is_multiple_of(MATRIX_ELEMENTS) {
            return Err(format!(
                "elements_per_block ({}) must be divisible by matrix_size ({})",
                elements_per_block, MATRIX_ELEMENTS
            ));
        }

        if !self
            .input_k_chunks
            .is_multiple_of(self.subgroup_threads_per_block)
        {
            return Err(format!(
                "input_k_chunks ({}) must be divisible by subgroup_threads_per_block ({})",
                self.input_k_chunks, self.subgroup_threads_per_block
            ));
        }

        if !self.input_m_elements.is_multiple_of(4) {
            return Err(format!(
                "input_m_elements ({}) must be divisible by 4",
                self.input_m_elements
            ));
        }

        if !self.input_n_elements.is_multiple_of(4) {
            return Err(format!(
                "input_n_elements ({}) must be divisible by 4",
                self.input_n_elements
            ));
        }

        if !input_k_elements.is_multiple_of(4) {
            return Err(format!(
                "input_k_elements ({}) must be divisible by 4",
                input_k_elements
            ));
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_sgemm_with_config(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    k_size: &str,
    config: ChunkedSgemmConfig,
) {
    let workgroup_id = kernel.workgroup_index();
    let global_id = kernel.global_id();
    let workgroup_local_index = kernel.workgroup_local_index();
    let elements_per_block = op.elements_per_block();
    let dtype = op.input_datatype;

    let sub_chunks = dequantize_mat4x4_block_count(op.matrix.datatype);

    // Validate configuration
    config.validate(elements_per_block, sub_chunks).unwrap();

    let sgemm_subgroup_threads_per_block = config.subgroup_threads_per_block;
    let sgemm_input_k_chunks = config.input_k_chunks;
    let sgemm_input_k_elements = config.input_k_elements();
    let sgemm_input_m_elements = config.input_m_elements;
    let sgemm_input_n_elements = config.input_n_elements;

    let cache_a_size = (sgemm_input_k_elements / 4) * (sgemm_input_m_elements / 4);
    let cache_a = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        MatrixType::new(["4".into(), "4".into()], dtype),
        cache_a_size.to_string(),
    );
    let cache_b_size = (sgemm_input_n_elements / 4) * (sgemm_input_k_elements / 4);
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
        "let pair_index = {workgroup_local_index} / {sgemm_subgroup_threads_per_block};"
    )
    .unwrap();
    writeln!(
        kernel,
        "let pair_local_index = {workgroup_local_index} % {sgemm_subgroup_threads_per_block};"
    )
    .unwrap();
    let subgroup_threads_per_sgemm_input_k_chunks =
        sgemm_input_k_chunks / sgemm_subgroup_threads_per_block;
    writeln!(
        kernel,
        "let pair_index_row = pair_local_index + {sgemm_subgroup_threads_per_block} * (pair_index % {subgroup_threads_per_sgemm_input_k_chunks});"
    )
    .unwrap();
    writeln!(
        kernel,
        "let pair_index_col = pair_index / {subgroup_threads_per_sgemm_input_k_chunks};"
    )
    .unwrap();

    // How many blocks do we have to process in the k dim
    writeln!(kernel, "let k_chunk_size = {k_size} / {MATRIX_ELEMENTS};").unwrap();
    writeln!(
        kernel,
        "let k_block_size = {k_size} / {elements_per_block};"
    )
    .unwrap();

    // This threads b_input offset in blocks
    writeln!(
        kernel,
        "let b_block_offset = (y * {sgemm_input_n_elements} + pair_index_col) * k_block_size;"
    )
    .unwrap();

    let chunks_per_block = elements_per_block / MATRIX_ELEMENTS;

    // Calculate one block sized group
    writeln!(
        kernel,
        "for (var k_start = 0u; k_start < k_chunk_size; k_start += {sgemm_input_k_chunks}u) {{"
    )
    .unwrap();
    {
        // This is seperated from k_start above to avoid divergance checks within each subgroup
        writeln!(kernel, "let k = k_start + pair_index_row;").unwrap();
        {
            // Load the b block into the cache
            writeln!(kernel, "let b_block_index = k / {chunks_per_block};").unwrap();
            writeln!(kernel, "let b_index_within_block = k % {chunks_per_block};").unwrap();
            let y_stride = sgemm_input_k_elements / 4;
            dequantize_mat4x4_block(
                kernel,
                op.matrix.datatype,
                "b_index_within_block",
                format!("{input_b}[b_block_index + b_block_offset]"),
                DataTypeEnum::F32,
                |data, kernel| {
                    writeln!(kernel, "let b_values = {data};").unwrap();
                    writeln!(kernel, "for (var index = 0u; index < 4u; index += 1u) {{").unwrap();
                    writeln!(
                        kernel,
                        "{cache_b}[(pair_index_col / 4) * {y_stride} + pair_index_row + {sgemm_input_k_chunks} * index][pair_index_col % 4] = b_values[index];"
                    )
                    .unwrap();
                    writeln!(kernel, "}}").unwrap();
                },
            );

            // Load the a block into the cache
            writeln!(
                kernel,
                "for (var index = 0u; index < {MATRIX_ELEMENTS}u; index += 1u) {{"
            )
            .unwrap();
            {
                let y_stride = sgemm_input_m_elements / 4;
                let mut indices = vec![];
                // Add batch indices first
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("block_batch_{dim}"));
                }
                // Then add M and K indices
                indices.push(format!("x * {sgemm_input_m_elements} + pair_index_col"));
                indices.push(format!("k*{MATRIX_ELEMENTS} + index"));
                writeln!(
                    kernel,
                    "let chunk_index = (pair_index_row * 4 + index / 4) * {y_stride} + pair_index_col / 4;"
                )
                .unwrap();
                writeln!(kernel, "let col_index = pair_index_col % 4;").unwrap();
                writeln!(kernel, "let row_index = index % 4;").unwrap();
                input_a
                    .check_bounds(
                        kernel,
                        indices.iter().cloned(),
                        |kernel: &mut GenericKernel| {
                            write!(
                                kernel,
                                "{cache_a}[chunk_index][col_index][row_index] = {input_a}["
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
                        "{cache_a}[chunk_index][col_index][row_index] = 0.0;"
                    )
                    .unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }

        // Make sure the caches are ready
        writeln!(kernel, "workgroupBarrier();").unwrap();

        // Now that the items are in cache, do the matrix multiplication
        writeln!(
            kernel,
            "let n_workgroup_index = {workgroup_local_index} / {};",
            sgemm_input_n_elements / 4
        )
        .unwrap();
        writeln!(
            kernel,
            "let m_workgroup_index = {workgroup_local_index} % {};",
            sgemm_input_m_elements / 4
        )
        .unwrap();
        writeln!(
            kernel,
            "for (var index = 0u; index < {}u; index += 1u) {{",
            sgemm_input_k_elements / 4
        )
        .unwrap();
        {
            // Load a 4x4 from cache_a
            writeln!(
                kernel,
                "let a_values = {cache_a}[index * {} + m_workgroup_index];",
                sgemm_input_m_elements / 4
            )
            .unwrap();
            // Load a 4x4 from cache_b with coalesced layout
            writeln!(
                kernel,
                "let b_cache_offset = (index / 4u) + {}u * (index % 4u);",
                sgemm_input_k_chunks
            )
            .unwrap();
            writeln!(
                kernel,
                "let b_values = {cache_b}[n_workgroup_index * {} + b_cache_offset];",
                sgemm_input_k_elements / 4
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
            std::fmt::Result::Ok(())
        },
    )?;
    Ok(())
}
