use crate::{
    DataType, DataTypeEnum, Device, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{inputs::MirValue, kernel::GenericKernel, operation::Operation},
    quantized::matmul::sgemv::SGEMV_CHUNK_SIZE,
};

use super::QMatrix;

mod sgemm;
mod sgemv;

#[derive(Debug)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: AnyComputeKey,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
}

impl QMatMulOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: AnyComputeKey,
        matrix: QMatrix,
    ) -> Self {
        let out_shape = vec![input_shape[0], matrix.shape[0]];
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
        }
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn sgemv(&self) -> bool {
        self.in_shape[0] == 1
    }

    fn k_size(&self) -> u32 {
        self.in_shape[1] as u32
    }

    fn m_size(&self) -> u32 {
        self.in_shape[0] as u32
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }
}

impl<T: DataType> Tensor<2, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }
}

#[cfg(test)]
async fn setup_smol_lm_matrix(
    name: &str,
) -> (crate::Device, QMatrix, candle_core::quantized::QMatMul) {
    use crate::Device;
    use fusor_gguf::GgufMetadata;

    let device = Device::new().await.unwrap();

    static BYTES: tokio::sync::OnceCell<Vec<u8>> = tokio::sync::OnceCell::const_new();

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = BYTES
        .get_or_init(|| async move {
            reqwest::get(url)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap()
                .into()
        })
        .await;
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

    for _ in 0..25 {
        let random_data: Vec<Vec<f32>> = (0..576)
            .map(|_| (0..576).map(|_| rand::random()).collect())
            .collect();
        let tensor = Tensor::<2, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data.iter().flat_map(|x| x.iter().copied()),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[576, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[576, 576]);
        let candle_result = candle_result.to_vec2::<f32>().unwrap();

        assert_eq!(fusor_shape, &[576, 576]);

        for x in 0..576 {
            for y in 0..576 {
                let expected = candle_result[x][y];
                let actual = result[[x, y]];
                if (expected - actual).abs() > 3. {
                    println!("Expected: {:?}", candle_result);
                    println!("Actual: {:?}", result);
                    panic!("expected: {}, actual: {}", expected, actual);
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
        let random_data: Vec<Vec<f32>> = (0..1)
            .map(|_| (0..size).map(|_| rand::random()).collect())
            .collect();
        let tensor = Tensor::<2, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data.iter().flat_map(|x| x.iter().copied()),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[1, size])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[1, embed_dim]);
        let candle_result = candle_result.to_vec2::<f32>().unwrap();

        assert_eq!(fusor_shape, &[1, embed_dim]);

        for x in 0..1 {
            for y in 0..embed_dim {
                let expected = candle_result[x][y];
                let actual = result[[x, y]];
                if (expected - actual).abs() > 3. {
                    println!("Expected: {:?}", candle_result);
                    println!("Actual: {:?}", result);
                    panic!("expected: {}, actual: {}", expected, actual);
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
        let random_data: Vec<Vec<f32>> = (0..width)
            .map(|_| (0..576).map(|_| rand::random()).collect())
            .collect();
        let tensor = Tensor::<2, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data.iter().flat_map(|x| x.iter().copied()),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[width, 49152]);
        let candle_result = candle_result.to_vec2::<f32>().unwrap();

        assert_eq!(fusor_shape, &[width, 49152]);

        for x in 0..width {
            for y in 0..49152 {
                let expected = candle_result[x][y];
                let actual = result[[x, y]];
                if (expected - actual).abs() > 3. {
                    println!("Expected: {:?}", candle_result);
                    println!("Actual: {:?}", result);
                    panic!("expected: {}, actual: {}", expected, actual);
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
        let random_data: Vec<Vec<f32>> = (0..width)
            .map(|_| (0..1536).map(|_| rand::random()).collect())
            .collect();
        let tensor = Tensor::<2, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data.iter().flat_map(|x| x.iter().copied()),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[width, 1536])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[width, 576]);
        let candle_result = candle_result.to_vec2::<f32>().unwrap();

        assert_eq!(fusor_shape, &[width, 576]);

        for x in 0..width {
            for y in 0..576 {
                let expected = candle_result[x][y];
                let actual = result[[x, y]];
                if (expected - actual).abs() > 3. {
                    println!("width: {}", width);
                    println!("Expected: {:?}", candle_result);
                    println!("Actual: {:?}", result);
                    panic!("expected: {}, actual: {}", expected, actual);
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
        let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
        if self.sgemv() {
            let limits = device.wgpu_device().limits();
            constraints.add_constraint(
                0,
                crate::mir::workgroup_shape::Constraint::less_than(
                    limits.max_compute_workgroup_size_x + 1,
                ),
            );
            constraints.add_constraint(
                0,
                crate::mir::workgroup_shape::Constraint::equals(limits.min_subgroup_size.max(16)),
            );
        } else {
            constraints.add_constraint(0, crate::mir::workgroup_shape::Constraint::Equals(1));
        }
        constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
        constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let n = self.n_size();
        let m = self.m_size();
        if self.sgemv() {
            [(n as u32).div_ceil(SGEMV_CHUNK_SIZE), 1, 1]
        } else {
            [
                (n as u32).div_ceil(workgroup_shape.x()),
                (m as u32).div_ceil(workgroup_shape.y()),
                1,
            ]
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_result(self.input).unwrap();
        let q_matrix = self.matrix.clone();
        let device = input.device();
        let a_shape = input.layout().shape();
        let b_shape = &self.matrix.shape;
        let output_tensor =
            TensorData::new_for_shape(device, &[a_shape[0], b_shape[0]], input.datatype());
        vec![input.into(), q_matrix.into(), output_tensor.into()]
    }

    // Related files/PRs in llama.cpp for reference:
    // https://github.com/ggml-org/llama.cpp/pull/2290
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L4561
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L5881
    // based on https://siboehm.com/articles/22/CUDA-MMM
    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        generic_kernel: &mut GenericKernel,
    ) {
        let datatype = self.input_datatype;
        let rank = self.matrix.shape.len() as u32;

        let input_a = generic_kernel.add_tensor_input(rank, false, datatype);
        let input_b = generic_kernel.add_q_matrix_input(rank, self.matrix.datatype);
        let output = generic_kernel.add_tensor_input(rank, true, datatype);

        let k_size = input_a.shape_binding(1);
        let m_size = input_a.shape_binding(0);
        let n_size = input_b.shape_binding(0);

        // Check if this is a sgemv or sgemm operation
        let algo = if self.sgemv() {
            sgemv::sgemv
        } else {
            sgemm::sgemm
        };

        algo(
            self,
            generic_kernel,
            &workgroup_shape,
            &input_a,
            &input_b,
            &output,
            &n_size,
            &m_size,
            &k_size,
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
