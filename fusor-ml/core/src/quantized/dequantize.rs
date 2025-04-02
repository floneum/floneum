use crate::{
    DataType, DataTypeEnum, Device, LazyTensorData, PerformanceQueries, Tensor, TensorData,
    TensorInfo, UntypedElementWiseKernel,
    compute_graph::ComputeGraph,
    kernel::{GenericKernel, KernelInputValue},
    padded_tensor_size,
};
use std::{fmt::Write, sync::OnceLock};
use wgpu::CommandEncoder;

use super::{QMatrix, dequantize_block};

pub(crate) struct DequantizeOperation {
    pub(crate) matrix: QMatrix,
    pub(crate) datatype: DataTypeEnum,
}

impl DequantizeOperation {
    pub(crate) fn new(matrix: QMatrix, datatype: DataTypeEnum) -> Self {
        DequantizeOperation { matrix, datatype }
    }
}

impl QMatrix {
    pub fn dequantize<const R: usize, T: DataType>(&self) -> Tensor<R, T> {
        assert_eq!(self.shape.len(), R, 
            "Dequantize: expected {}D tensor, got {}D tensor. Shape: {:?}",
            R,
            self.shape.len(),
            self.shape
        );

        let device = self.device.clone();
        let graph = ComputeGraph::new(device.clone());
        let key = graph.dequantize(self.clone(), T::WGSL_TYPE);

        let data = LazyTensorData::from_parts(
            device,
            graph,
            TensorInfo::new(self.shape().into(), T::WGSL_TYPE),
            key.into(),
        );

        Tensor::from_parts(data)
    }
}

pub(crate) struct UntypedDequantize {
    kernel: OnceLock<GenericKernel>,
    post_dequantize: UntypedElementWiseKernel,
    datatype: DataTypeEnum,
    matrix: QMatrix,
}

impl UntypedDequantize {
    pub(crate) fn new(datatype: DataTypeEnum, matrix: QMatrix) -> Self {
        Self {
            kernel: OnceLock::new(),
            post_dequantize: UntypedElementWiseKernel::empty(datatype),
            datatype,
            matrix,
        }
    }

    pub(crate) fn set_post_element_wise(&mut self, post_dequantize: UntypedElementWiseKernel) {
        self.post_dequantize = post_dequantize;
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn work_group_dispatch(&self) -> [u32; 3] {
        std::array::from_fn(|i| match i.cmp(&(self.matrix.shape.len() - 1)) {
            std::cmp::Ordering::Less => {
                let n = self.matrix.shape[i];
                (n as u32).div_ceil(self.work_group_size()[i])
            }
            std::cmp::Ordering::Equal => {
                let n = self.matrix.shape[i];
                (n as u32).div_ceil(self.work_group_size()[i] * self.elements_per_block())
            }
            std::cmp::Ordering::Greater => 1,
        })
    }

    fn work_group_size(&self) -> [u32; 3] {
        [1, 1, 1]
    }

    fn compile(&self) -> &GenericKernel {
        self.kernel.get_or_init(|| {
            let mut generic_kernel = GenericKernel::new();

            generic_kernel.set_workgroup_size(self.work_group_size());

            let mut kernel = String::new();

            let datatype = self.datatype;
            let rank = self.matrix.shape.len() as u32;

            let input = generic_kernel.add_q_matrix_input(rank, self.matrix.datatype);
            let output = generic_kernel.add_tensor_input(rank, true, datatype);

            let post_element_wise = self.post_dequantize.add_functions(&mut generic_kernel);
            let process_output = |input: &str| {
                post_element_wise
                    .iter()
                    .fold(input.to_string(), |acc, f| f.call(vec![acc]))
            };

            let global_id = generic_kernel.global_id();
            let elements_per_block = self.elements_per_block();

            for (dim, axis) in ["x", "y", "z"]
                .iter()
                .enumerate()
                .take(self.matrix.shape.len())
            {
                writeln!(&mut kernel, "let index_{dim} = {global_id}.{axis};").unwrap();
            }

            write!(&mut kernel, "let chunk_index = ").unwrap();
            input.strided_index(&mut kernel, (0..).map(|i| format!("index_{i}")));
            writeln!(&mut kernel, ";").unwrap();
            writeln!(&mut kernel, "let chunk = {input}[chunk_index];").unwrap();

            dequantize_block(
                &mut kernel,
                self.matrix.datatype,
                "chunk".to_string(),
                datatype,
                |i, data, kernel| {
                    let indexes: Box<[_]> = (0..rank)
                        .map(|dim| {
                            let base = format!("index_{dim}");
                            if dim == rank - 1 {
                                format!("{base} * {elements_per_block} + {i}")
                            } else {
                                base
                            }
                        })
                        .collect();
                    output.check_bounds(kernel, indexes.clone(), |kernel| {
                        write!(kernel, "let output_index = ").unwrap();
                        output.strided_index(kernel, indexes);
                        writeln!(kernel, ";").unwrap();

                        writeln!(
                            kernel,
                            "{output}[output_index] = {};",
                            process_output(&data)
                        )
                        .unwrap();
                    });
                },
            );

            generic_kernel.set_body(kernel);

            generic_kernel
        })
    }

    pub fn run_with_query(
        &self,
        device: &crate::Device,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let shape = &self.matrix.shape;
        let output_buf = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_tensor_size(
                (shape.iter().product::<usize>() * self.datatype.element_size()) as u64,
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor = TensorData::new_from_buffer(device, output_buf, &shape, self.datatype);
        self.run_with_query_and_out_tensor(device, query, &output_tensor, command_encoder);
        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        device: &Device,
        query: Option<&PerformanceQueries>,
        output_tensor: &TensorData,
        command_encoder: &mut CommandEncoder,
    ) {
        let matrix_shape = &self.matrix.shape;
        assert_eq!(&**matrix_shape, output_tensor.layout().shape());
        let module = self.compile();

        let workgroup_dispatch_size = self.work_group_dispatch();

        module.run_with_query(
            device,
            [
                KernelInputValue::from(self.matrix.clone()),
                output_tensor.clone().into(),
            ],
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_dequantize_smol_lm() {
    use crate::Device;
    use fusor_gguf::GgufMetadata;
    use num_traits::float::Float;

    let device = Device::new().await.unwrap();

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();

    for (name, candle_q_matrix_metadata) in candle_metadata.tensor_infos {
        let tensor = metadata.tensor_infos.get(&*name).unwrap();
        println!("{}: {:?}", name, tensor);

        let candle_q_tensor = candle_q_matrix_metadata
            .read(
                &mut reader,
                candle_metadata.tensor_data_offset,
                &candle_core::Device::Cpu,
            )
            .unwrap();
        let candle_result = candle_q_tensor
            .dequantize_f16(&candle_core::Device::Cpu)
            .unwrap();

        let candle_result_doubled = (&candle_result * 2.0).unwrap();

        let q_matrix_metadata = metadata.tensor_infos.get(&*name).unwrap();

        let q_matrix = QMatrix::read(
            &device,
            q_matrix_metadata,
            &mut reader,
            metadata.tensor_data_offset,
        )
        .unwrap();
        assert_eq!(candle_result.shape().dims(), q_matrix.shape());

        match candle_result.rank() {
            1 => {
                let fusor_result = q_matrix.dequantize::<1, half::f16>();
                let candle_result = candle_result.to_vec1::<half::f16>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for i in 0..candle_result.len() {
                    let expected = candle_result[i];
                    let actual = result[[i]];
                    if (expected - actual).abs() > half::f16::from_f32(0.01) {
                        assert_eq!(
                            expected, actual,
                            "Mismatch at ({i}) - expected: {expected}, actual: {actual}"
                        );
                    }
                }

                let fusor_result = q_matrix.dequantize::<1, half::f16>() * half::f16::from_f32(2.0);
                let candle_result = candle_result_doubled.to_vec1::<half::f16>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for i in 0..candle_result.len() {
                    let expected = candle_result[i];
                    let actual = result[[i]];
                    if (expected - actual).abs() > half::f16::from_f32(0.01) {
                        assert_eq!(
                            expected, actual,
                            "Mismatch at ({i}) - expected: {expected}, actual: {actual}"
                        );
                    }
                }
            }
            2 => {
                let fusor_result = q_matrix.dequantize::<2, half::f16>();
                let candle_result = candle_result.to_vec2::<half::f16>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for x in 0..candle_result.len() {
                    for y in 0..candle_result[0].len() {
                        let expected = candle_result[x][y];
                        let actual = result[[x, y]];
                        if (expected - actual).abs() > half::f16::from_f32(0.01) {
                            assert_eq!(
                                expected, actual,
                                "Mismatch at ({x}, {y}) - expected: {expected}, actual: {actual}"
                            );
                        }
                    }
                }

                let fusor_result = q_matrix.dequantize::<2, half::f16>() * half::f16::from_f32(2.0);
                let candle_result = candle_result_doubled.to_vec2::<half::f16>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for x in 0..candle_result.len() {
                    for y in 0..candle_result[0].len() {
                        let expected = candle_result[x][y];
                        let actual = result[[x, y]];
                        if (expected - actual).abs() > half::f16::from_f32(0.01) {
                            assert_eq!(
                                expected, actual,
                                "Mismatch at ({x}, {y}) - expected: {expected}, actual: {actual}"
                            );
                        }
                    }
                }
            }
            _ => todo!(),
        }
    }
}
