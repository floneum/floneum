use crate::{
    DataType, DataTypeEnum, Device, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{inputs::MirValue, kernel::GenericKernel, operation::Operation},
};
use std::fmt::Write;

use super::{QMatrix, dequantize_block};

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

impl QMatMulOperation {
    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }
}

impl Operation for QMatMulOperation {
    // fn work_group_dispatch(&self, a_shape: &[usize]) -> [u32; 3] {

    // }

    // fn work_group_size(&self) -> [u32; 3] {
    //     [16, 16, 1]
    // }

    fn workgroup_shape_constraints(
        &self,
        _: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
        constraints.add_constraint(0, crate::mir::workgroup_shape::Constraint::Equals(16));
        constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(16));
        constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let input = inputs[0].as_tensor().unwrap();
        let a_shape = input.layout().shape();
        let m = a_shape[0];
        let n = self.matrix.shape[0];
        [
            (n as u32).div_ceil(workgroup_shape.x()),
            (m as u32).div_ceil(workgroup_shape.y()),
            1,
        ]
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
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
        generic_kernel: &mut GenericKernel,
    ) -> MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap().clone();
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
                write!(code, "acc = fma({input_a}[").unwrap();
                input_a.strided_index(
                    code,
                    ["y".to_string(), format!("k * {elements_per_block} + {i}")],
                );
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

        generic_kernel.push_body(kernel);

        output_tensor.into()
    }
}
