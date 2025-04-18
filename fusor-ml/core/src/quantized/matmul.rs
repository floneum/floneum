use crate::{
    DataType, DataTypeEnum, Device, PerformanceQueries, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    kernel::{GenericKernel, KernelInputValue},
    padded_tensor_size,
};
use std::{fmt::Write, sync::OnceLock};
use wgpu::CommandEncoder;

use super::{QMatrix, dequantize_block};

#[derive(Debug)]
pub(crate) struct QMatMulOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) matrix: QMatrix,
    pub(crate) out_shape: Box<[usize]>,
}

impl QMatMulOperation {
    pub(crate) fn new(input_shape: &[usize], input: AnyComputeKey, matrix: QMatrix) -> Self {
        let out_shape = vec![input_shape[0], matrix.shape[0]];
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input,
            matrix,
            out_shape,
        }
    }
}

impl<T: DataType> Tensor<2, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul() {
    use crate::Device;
    use crate::Tensor;
    use candle_core::Module;
    use fusor_gguf::GgufMetadata;

    let device = Device::new().await.unwrap();

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let candle_q_matrix_metadata = candle_metadata
        .tensor_infos
        .get("blk.0.attn_q.weight")
        .unwrap();
    let candle_q_tensor = candle_q_matrix_metadata
        .read(
            &mut reader,
            candle_metadata.tensor_data_offset,
            &candle_core::Device::Cpu,
        )
        .unwrap();
    let candle_q_matrix = candle_core::quantized::QMatMul::from_qtensor(candle_q_tensor).unwrap();

    let q_matrix_metadata = metadata.tensor_infos.get("blk.0.attn_q.weight").unwrap();

    let q_matrix = QMatrix::read(
        &device,
        q_matrix_metadata,
        &mut reader,
        metadata.tensor_data_offset,
    )
    .unwrap();

    for _ in 0..10 {
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
async fn test_fuzz_q_mat_mul_q8_0() {
    use crate::Device;
    use crate::Tensor;
    use candle_core::Module;
    use fusor_gguf::GgufMetadata;

    let device = Device::new().await.unwrap();

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let candle_q_matrix_metadata = candle_metadata
        .tensor_infos
        .get("token_embd.weight")
        .unwrap();
    let candle_q_tensor = candle_q_matrix_metadata
        .read(
            &mut reader,
            candle_metadata.tensor_data_offset,
            &candle_core::Device::Cpu,
        )
        .unwrap();
    let candle_q_matrix = candle_core::quantized::QMatMul::from_qtensor(candle_q_tensor).unwrap();

    let q_matrix_metadata = metadata.tensor_infos.get("token_embd.weight").unwrap();

    let q_matrix = QMatrix::read(
        &device,
        q_matrix_metadata,
        &mut reader,
        metadata.tensor_data_offset,
    )
    .unwrap();

    for _ in 0..10 {
        let random_data: Vec<Vec<f32>> = (0..1)
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
        .reshape(&[1, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[1, 49152]);
        let candle_result = candle_result.to_vec2::<f32>().unwrap();

        assert_eq!(fusor_shape, &[1, 49152]);

        for x in 0..1 {
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

pub(crate) struct UntypedQMatMul {
    kernel: OnceLock<GenericKernel>,
    input_datatype: DataTypeEnum,
    matrix: QMatrix,
}

impl UntypedQMatMul {
    pub(crate) const fn new(datatype: DataTypeEnum, matrix: QMatrix) -> Self {
        Self {
            kernel: OnceLock::new(),
            input_datatype: datatype,
            matrix,
        }
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn work_group_dispatch(&self, a_shape: &[usize]) -> [u32; 3] {
        let m = a_shape[0];
        let n = self.matrix.shape[0];
        [
            (n as u32).div_ceil(self.work_group_size()[0]),
            (m as u32).div_ceil(self.work_group_size()[1]),
            1,
        ]
    }

    fn work_group_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }

    // Related files/PRs in llama.cpp for reference:
    // https://github.com/ggml-org/llama.cpp/pull/2290
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L4561
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L5881
    fn compile(&self) -> &GenericKernel {
        self.kernel.get_or_init(|| {
            // based on https://siboehm.com/articles/22/CUDA-MMM
            let mut generic_kernel = GenericKernel::new();

            generic_kernel.set_workgroup_size(self.work_group_size());

            let mut kernel = String::new();

            let datatype = self.input_datatype;
            let rank = self.matrix.shape.len() as u32;

            let input_a = generic_kernel.add_tensor_input(rank, false, datatype);
            let input_b = generic_kernel.add_q_matrix_input(rank, self.matrix.datatype);
            let output = generic_kernel.add_tensor_input(rank, true, datatype);

            let global_id = generic_kernel.global_id();
            let elements_per_block = self.elements_per_block();

            let k_size = input_a.shape_binding(1);
            let m_size = input_a.shape_binding(0);
            let n_size = input_b.shape_binding(0);

            writeln!(&mut kernel, "let x = {global_id}.x;").unwrap();
            writeln!(&mut kernel, "let y = {global_id}.y;").unwrap();

            writeln!(&mut kernel, "var acc = 0.0;").unwrap();

            // Calculate one block sized group
            writeln!(&mut kernel, "if x < {n_size} && y < {m_size} {{").unwrap();

            writeln!(
                &mut kernel,
                "for (var k = 0u; k < {k_size} / {elements_per_block}; k += 1u) {{"
            )
            .unwrap();

            writeln!(
                &mut kernel,
                "let chunk = {input_b}[k + x * {k_size} / {elements_per_block}];"
            )
            .unwrap();

            dequantize_block(
                &mut kernel,
                self.matrix.datatype,
                "chunk".to_string(),
                DataTypeEnum::F32,
                |i, data, code| {
                    write!(
                        code,
                        "acc = fma({input_a}["
                    ).unwrap();
                    input_a.strided_index(code, ["y".to_string(), format!("k * {elements_per_block} + {i}")]);
                    write!(code, "], {data}, acc);").unwrap();
                },
            );

            writeln!(&mut kernel, "}}").unwrap();

            writeln!(&mut kernel, "}}").unwrap();

            // Then write the result
            writeln!(&mut kernel, "if x < {n_size} && y < {m_size} {{").unwrap();
            write!(&mut kernel, "let output_index = ").unwrap();
            output.strided_index(&mut kernel, ["y".to_string(), "x".to_string()]);
            writeln!(&mut kernel, ";").unwrap();
            writeln!(&mut kernel, "{output}[output_index] = acc;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();

            generic_kernel.set_body(kernel);

            generic_kernel
        })
    }

    pub fn run_with_query(
        &self,
        input: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let device = input.device();
        let a_shape = input.layout().shape();
        let b_shape = &self.matrix.shape;
        let output_buf = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_tensor_size(
                (a_shape[0] * b_shape[0] * input.datatype().element_size()) as u64,
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor = TensorData::new_from_buffer(
            device,
            output_buf,
            &[a_shape[0], b_shape[0]],
            input.datatype(),
        );
        self.run_with_query_and_out_tensor(device, input, query, &output_tensor, command_encoder);
        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        device: &Device,
        input: &TensorData,
        query: Option<&PerformanceQueries>,
        output_tensor: &TensorData,
        command_encoder: &mut CommandEncoder,
    ) {
        let matrix_shape = &self.matrix.shape;
        assert_eq!(input.layout().shape()[1], matrix_shape[1]);
        let module = self.compile();

        let a_shape = input.layout().shape();
        let b_shape = matrix_shape;
        assert_eq!(*output_tensor.layout().shape(), [a_shape[0], b_shape[0]]);

        let workgroup_dispatch_size = self.work_group_dispatch(a_shape);

        module.run_with_query(
            device,
            [
                KernelInputValue::from(input.clone()),
                self.matrix.clone().into(),
                output_tensor.clone().into(),
            ],
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}
