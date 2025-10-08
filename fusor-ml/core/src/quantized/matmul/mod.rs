use crate::{
    DataType, DataTypeEnum, Device, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{inputs::MirValue, kernel::GenericKernel, operation::Operation},
};

use super::QMatrix;

mod sgemm;
mod sgemv;

pub use sgemm::{ChunkedSgemmConfig, GeneralSgemmConfig};

#[derive(Debug)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: AnyComputeKey,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) chunked_config: Option<ChunkedSgemmConfig>,
    pub(crate) general_config: Option<GeneralSgemmConfig>,
}

impl QMatMulOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: AnyComputeKey,
        matrix: QMatrix,
    ) -> Self {
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = matrix.shape[0];
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
            chunked_config: None,
            general_config: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_chunked_config(mut self, config: ChunkedSgemmConfig) -> Self {
        self.chunked_config = Some(config);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_general_config(mut self, config: GeneralSgemmConfig) -> Self {
        self.general_config = Some(config);
        self
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn sgemv(&self) -> bool {
        let m_dim_idx = self.in_shape.len() - 2;
        self.in_shape[m_dim_idx] == 1
    }

    fn m_size(&self) -> u32 {
        let m_dim_idx = self.in_shape.len() - 2;
        self.in_shape[m_dim_idx] as u32
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }

    #[cfg(test)]
    pub(crate) fn q_mat_mul_with_chunked_config(
        &self,
        other: &QMatrix,
        config: ChunkedSgemmConfig,
    ) -> Self {
        self.add_q_mat_mul_with_chunked_config(other, config)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn q_mat_mul_with_general_config(
        &self,
        other: &QMatrix,
        config: GeneralSgemmConfig,
    ) -> Self {
        self.add_q_mat_mul_with_general_config(other, config)
    }
}

#[cfg(test)]
async fn setup_smol_lm_matrix(
    name: &str,
) -> (crate::Device, QMatrix, candle_core::quantized::QMatMul) {
    use kalosm_model_types::FileSource;
    let source = FileSource::HuggingFace {
        model_id: "unsloth/SmolLM2-135M-Instruct-GGUF".to_string(),
        revision: "main".to_string(),
        file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".to_string(),
    };

    setup_smol_lm_matrix_with_source(name, source).await
}

#[cfg(test)]
async fn setup_smol_lm_matrix_with_source(
    name: &str,
    source: kalosm_model_types::FileSource,
) -> (crate::Device, QMatrix, candle_core::quantized::QMatMul) {
    use crate::Device;
    use fusor_gguf::GgufMetadata;
    use kalosm_common::Cache;

    let device = Device::new().await.unwrap();

    let cache = Cache::default();
    let path = cache.get(&source, |_| {}).await.unwrap();
    let bytes = tokio::fs::read(&path).await.unwrap();

    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let candle_q_matrix_metadata = candle_metadata.tensor_infos.get(name).unwrap();
    let candle_q_tensor = candle_q_matrix_metadata
        .read(
            &mut reader,
            candle_metadata.tensor_data_offset,
            &candle_core::Device::Cpu,
        )
        .unwrap();
    let candle_q_matrix = candle_core::quantized::QMatMul::from_qtensor(candle_q_tensor).unwrap();

    let q_matrix_metadata = metadata.tensor_infos.get(name).unwrap();

    let q_matrix = QMatrix::read(
        &device,
        q_matrix_metadata,
        &mut reader,
        metadata.tensor_data_offset,
    )
    .unwrap();

    (device, q_matrix, candle_q_matrix)
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    for _ in 0..25 {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..576)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);
        println!("tensor: {tensor:?}");

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, 576, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 576, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 576, 576]);

        for batch in 0..batch {
            for x in 0..576 {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_transposed() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    for _ in 0..25 {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..576)
            .map(|_| {
                (0..576)
                    .map(|_| (0..batch).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);
        println!("tensor: {tensor:?}");

        let result = tensor.transpose(0, 2).q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[576, 576, batch])
        .unwrap();
        let candle_result = candle_q_matrix
            .forward(&candle_b.transpose(0, 2).unwrap().contiguous().unwrap())
            .unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 576, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 576, 576]);

        for batch in 0..batch {
            for x in 0..576 {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_sgemv() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("token_embd.weight").await;

    for _ in 0..25 {
        let size = 576;
        let embed_dim = 49152;
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..1)
                    .map(|_| (0..size).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, 1, size])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 1, embed_dim]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 1, embed_dim]);

        for batch in 0..batch {
            for x in 0..1 {
                for y in 0..embed_dim {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_gemv_transposed() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    for _ in 0..25 {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..576)
            .map(|_| {
                (0..1)
                    .map(|_| (0..batch).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);
        println!("tensor: {tensor:?}");

        let result = tensor.transpose(0, 2).q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[576, 1, batch])
        .unwrap();
        let candle_result = candle_q_matrix
            .forward(&candle_b.transpose(0, 2).unwrap().contiguous().unwrap())
            .unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 1, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 1, 576]);

        for batch in 0..batch {
            for x in 0..1 {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q8_0() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("token_embd.weight").await;

    // Always test the edge cases
    let mut widths = vec![1, 256];
    // Then test a bunch of other random widths
    widths.extend((2..25).map(|_| rand::random_range(1..=64)));

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 2) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 49152]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 49152]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..49152 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q5_0_gemv() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.ffn_gate.weight").await;

    for _ in 0..25 {
        let width = 1;
        let height = 1536;
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, height]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, height]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..height {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q4_0_gemv() {
    use crate::Tensor;
    use candle_core::Module;
    use kalosm_model_types::FileSource;

    let source = FileSource::HuggingFace {
        model_id: "bartowski/SmolLM2-135M-Instruct-GGUF".to_string(),
        revision: "main".to_string(),
        file: "SmolLM2-135M-Instruct-Q4_0.gguf".to_string(),
    };
    let (device, q_matrix, candle_q_matrix) =
        setup_smol_lm_matrix_with_source("blk.0.ffn_gate.weight", source).await;

    for _ in 0..25 {
        let width = 1;
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..1)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 1536]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 1536]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..1536 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q6k() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.ffn_down.weight").await;

    // Always test the edge cases
    let mut widths = vec![1, 256];
    // Then test a bunch of other random widths
    widths.extend((2..25).map(|_| rand::random_range(1..=64)));

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..1536).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 1536])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 576]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("width: {width}");
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_q6k_with_different_configs() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.ffn_down.weight").await;

    // Test multiple configs
    let configs = vec![
        // Small cache
        ChunkedSgemmConfig {
            subgroup_threads_per_block: 2,
            input_k_chunks: 2,
            input_m_elements: 8,
            input_n_elements: 8,
        },
        // Larger K chunks
        ChunkedSgemmConfig {
            subgroup_threads_per_block: 4,
            input_k_chunks: 4,
            input_m_elements: 16,
            input_n_elements: 16,
        },
        // Default
        ChunkedSgemmConfig::default(),
    ];

    for config in configs {
        let width = 32;
        let batch = 2;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..1536).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul_with_chunked_config(&q_matrix, config);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 1536])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 576]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("config: {config:?}");
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q4k() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.3.ffn_down.weight").await;

    // Always test the edge cases
    let mut widths = vec![1, 256];
    // Then test a bunch of other random widths
    widths.extend((2..25).map(|_| rand::random_range(1..=64)));

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..1536).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 1536])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 576]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 3. {
                        println!("width: {width}");
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

impl Operation for QMatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        if self.sgemv() {
            sgemv::workgroup_shape_constraints(&self.matrix, device)
        } else {
            sgemm::workgroup_shape_constraints(&self.matrix, device)
        }
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let n = self.n_size();
        let m = self.m_size();
        // Calculate batch size for dimensions beyond the last two (M, K)
        let batch_size: u32 = self
            .in_shape
            .iter()
            .rev()
            .skip(2)
            .map(|x| *x as u32)
            .product();

        if self.sgemv() {
            sgemv::dispatch_size(&self.matrix, n, m, batch_size)
        } else {
            sgemm::dispatch_size(workgroup_shape, &self.matrix, n, m, batch_size)
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_result(self.input).unwrap();
        let q_matrix = self.matrix.clone();
        let device = input.device();
        let output_tensor = TensorData::new_for_shape(device, &self.out_shape, input.datatype());
        vec![input.into(), q_matrix.into(), output_tensor.into()]
    }

    // Related files/PRs in llama.cpp for reference:
    // https://github.com/ggml-org/llama.cpp/pull/2290
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L4561
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L5881
    // based on https://siboehm.com/articles/22/CUDA-MMM
    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        generic_kernel: &mut GenericKernel,
    ) {
        let datatype = self.input_datatype;
        let rank = self.in_shape.len() as u32;
        let matrix_rank = self.matrix.shape.len() as u32;

        let input_a = generic_kernel.add_tensor_input(rank, false, datatype);
        let input_b = generic_kernel.add_q_matrix_input(matrix_rank, self.matrix.datatype);
        let output = generic_kernel.add_tensor_input(rank, true, datatype);

        // For batched operations, we need to get the correct dimension indices
        let k_size = input_a.shape_binding(rank - 1).to_string(); // Last dimension is K
        let m_size = input_a.shape_binding(rank - 2).to_string(); // Second-to-last dimension is M
        let n_size = input_b.shape_binding(0).to_string();

        // Check if this is a sgemv or sgemm operation
        let algo = if self.sgemv() {
            sgemv::sgemv
        } else {
            sgemm::sgemm
        };

        algo(
            self,
            generic_kernel,
            workgroup_shape,
            &input_a,
            &input_b,
            &output,
            &n_size,
            &m_size,
            &k_size,
            graph,
        );
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap();
        output_tensor.clone().into()
    }

    fn name(&self) -> String {
        format!(
            "q_mat_mul_{}_{}_{}_{}",
            self.input_datatype,
            self.in_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.matrix.datatype,
            self.matrix
                .shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}
