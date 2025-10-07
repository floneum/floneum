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

/// Configuration for chunked SGEMM algorithm
#[derive(Debug, Clone, Copy)]
pub struct ChunkedSgemmConfig {
    /// Size of the chunk we will dot at a time (must be divisible by 4)
    pub matrix_size: u32,
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
            matrix_size: 16,
            subgroup_threads_per_block: 2,
            input_k_chunks: 2,
            input_m_elements: 16,
            input_n_elements: 16,
        }
    }

    /// Compute the total K elements loaded per step
    pub const fn input_k_elements(&self) -> u32 {
        self.input_k_chunks * self.matrix_size
    }

    /// Validate configuration parameters
    pub fn validate(&self, elements_per_block: u32, sub_chunks: usize) -> Result<(), String> {
        let input_k_elements = self.input_k_elements();

        if elements_per_block % input_k_elements != 0 {
            return Err(format!(
                "elements_per_block ({}) must be divisible by input_k_elements ({})",
                elements_per_block, input_k_elements
            ));
        }

        if sub_chunks % self.input_k_chunks as usize != 0 {
            return Err(format!(
                "sub_chunks ({}) must be divisible by input_k_chunks ({})",
                sub_chunks, self.input_k_chunks
            ));
        }

        if self.input_k_chunks % self.subgroup_threads_per_block != 0 {
            return Err(format!(
                "input_k_chunks ({}) must be divisible by subgroup_threads_per_block ({})",
                self.input_k_chunks, self.subgroup_threads_per_block
            ));
        }

        if self.input_m_elements % 4 != 0 {
            return Err(format!(
                "input_m_elements ({}) must be divisible by 4",
                self.input_m_elements
            ));
        }

        if self.input_n_elements % 4 != 0 {
            return Err(format!(
                "input_n_elements ({}) must be divisible by 4",
                self.input_n_elements
            ));
        }

        if input_k_elements % 4 != 0 {
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

    let sgemm_matrix_size = config.matrix_size;
    let sgemm_subgroup_threads_per_block = config.subgroup_threads_per_block;
    let sgemm_input_k_elements = config.input_k_elements();
    let sgemm_input_m_elements = config.input_m_elements;
    let sgemm_input_n_elements = config.input_n_elements;

    let cache_a_size = (sgemm_input_m_elements / 4) * (sgemm_input_k_elements / 4);
    let cache_a = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        MatrixType::new(["4".into(), "4".into()], dtype),
        cache_a_size.to_string(),
    );
    let cache_b_size = (sgemm_input_k_elements / 4) * (sgemm_input_n_elements / 4);
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

    // How many blocks do we have to process in the k dim
    writeln!(kernel, "let k_chunk_size = {k_size} / {sgemm_matrix_size};").unwrap();
    writeln!(
        kernel,
        "let k_block_size = {k_size} / {elements_per_block};"
    )
    .unwrap();

    // This threads b_input offset in blocks
    writeln!(
        kernel,
        "let b_block_offset = (y * {sgemm_input_n_elements} + pair_index) * k_block_size;"
    )
    .unwrap();

    let chunks_per_block = elements_per_block / sgemm_matrix_size;

    // Calculate one block sized group
    writeln!(
        kernel,
        "for (var k_start = 0u; k_start < k_chunk_size; k_start += {sgemm_subgroup_threads_per_block}u) {{"
    )
    .unwrap();
    {
        // This is seperated from k_start above to avoid divergance checks within each subgroup
        writeln!(kernel, "let k = k_start + pair_local_index;").unwrap();
        writeln!(
            kernel,
            "for (var i = 0u; i < {}u; i += 1u) {{",
            sgemm_subgroup_threads_per_block
        )
        .unwrap();
        {
            // Load the b block into the cache
            writeln!(kernel, "let b_block_index = k / {chunks_per_block};").unwrap();
            writeln!(kernel, "let b_index_within_block = k % {chunks_per_block};").unwrap();
            let y_stride = sgemm_input_k_elements / 4;
            let block_half = (sgemm_input_n_elements / 4) / 2;
            dequantize_mat4x4_block(
                kernel,
                op.matrix.datatype,
                "b_index_within_block",
                format!(
                    "{input_b}[b_block_index + b_block_offset + i * {} * k_block_size]",
                    sgemm_input_n_elements / 2
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
                "for (var index = 0u; index < {sgemm_matrix_size}u; index += 1u) {{"
            )
            .unwrap();
            {
                let y_stride = sgemm_input_k_elements / 4;
                let block_half = (sgemm_input_m_elements / 4) / 2;
                let mut indices = vec![];
                // Add batch indices first
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("block_batch_{dim}"));
                }
                // Then add M and K indices
                indices.push(format!(
                    "x * {sgemm_input_m_elements} + pair_index + i * {}",
                    sgemm_input_m_elements / 2
                ));
                indices.push(format!("k*{sgemm_matrix_size} + index"));
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
            sgemm_input_k_elements / 4,
            sgemm_input_n_elements / 4
        )
        .unwrap();
        writeln!(
            kernel,
            "let m_workgroup_offset = {} * ({workgroup_local_index} % {});",
            sgemm_input_k_elements / 4,
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
            std::fmt::Result::Ok(())
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = ChunkedSgemmConfig::default();
        // For Q4_K, elements_per_block = 256, sub_chunks = 8
        assert!(config.validate(256, 8).is_ok());
    }

    #[test]
    fn test_alternative_config_small_cache() {
        // Smaller cache sizes for memory-constrained environments
        let config = ChunkedSgemmConfig {
            matrix_size: 16,
            subgroup_threads_per_block: 2,
            input_k_chunks: 2,
            input_m_elements: 8,
            input_n_elements: 8,
        };
        assert!(config.validate(256, 8).is_ok());
    }

    #[test]
    fn test_alternative_config_larger_k_chunks() {
        // More K chunks for better parallelism
        let config = ChunkedSgemmConfig {
            matrix_size: 16,
            subgroup_threads_per_block: 4,
            input_k_chunks: 4,
            input_m_elements: 16,
            input_n_elements: 16,
        };
        assert!(config.validate(256, 8).is_ok());
    }

    #[test]
    fn test_config_validation_fails_divisibility() {
        let config = ChunkedSgemmConfig {
            matrix_size: 16,
            subgroup_threads_per_block: 2,
            input_k_chunks: 3, // Not divisible by subgroup_threads_per_block
            input_m_elements: 16,
            input_n_elements: 16,
        };
        assert!(config.validate(256, 8).is_err());
    }

    #[test]
    fn test_config_validation_fails_not_divisible_by_4() {
        let config = ChunkedSgemmConfig {
            matrix_size: 16,
            subgroup_threads_per_block: 2,
            input_k_chunks: 2,
            input_m_elements: 15, // Not divisible by 4
            input_n_elements: 16,
        };
        assert!(config.validate(256, 8).is_err());
    }
}
