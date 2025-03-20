use std::{
    fmt::{Display, Write},
    sync::{Arc, OnceLock},
};

use fusor_gguf::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock, GgufReadError,
    GgufTensorMetadata,
};
use wgpu::{CommandEncoder, util::DeviceExt};

use crate::{
    DataType, DataTypeEnum, Device, PerformanceQueries, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    kernel::{GenericKernel, KernelInputValue},
    padded_tensor_size,
    quantized_types_wgsl::{
        write_q4_0_type, write_q4_k_type, write_q5_0_type, write_q6_k_type, write_q8_0_type,
    },
};

pub struct QMatMulOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) matrix: QMatrix,
}

impl QMatMulOperation {
    pub fn new(input: AnyComputeKey, matrix: QMatrix) -> Self {
        QMatMulOperation { input, matrix }
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul() {
    use crate::Device;
    use crate::Tensor;
    use fusor_gguf::GgufMetadata;

    let device = Device::new().await.unwrap();

    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

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
    let candle_q_matrix = candle_q_matrix_metadata
        .read(
            &mut reader,
            candle_metadata.tensor_data_offset,
            &candle_core::Device::Cpu,
        )
        .unwrap();
    let dequantized = candle_q_matrix
        .dequantize(&candle_core::Device::Cpu)
        .unwrap();
    let as_vec2 = dequantized.to_vec2::<f32>().unwrap();
    let mut ndarray_a = ndarray::Array2::zeros((576, 576));
    for i in 0..576 {
        for j in 0..576 {
            ndarray_a[[i, j]] = as_vec2[i][j];
        }
    }

    let q_matrix_metadata = metadata.tensor_infos.get("blk.0.attn_q.weight").unwrap();

    let q_matrix = QMatrix::read(
        device.wgpu_device(),
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

        let mut ndarray_b = ndarray::Array2::zeros((576, 576));
        for i in 0..576 {
            for j in 0..576 {
                ndarray_b[[i, j]] = random_data[i][j];
            }
        }
        let nd_array_result = ndarray_b.dot(&ndarray_a);

        assert_eq!(nd_array_result.shape(), &[576, 576]);
        assert_eq!(fusor_shape, &[576, 576]);

        for x in 0..576 {
            for y in 0..576 {
                let expected = nd_array_result[[x, y]];
                let actual = result[[x, y]];
                if (expected - actual).abs() > 3. {
                    println!("Expected: {:?}", nd_array_result);
                    println!("Actual: {:?}", result);
                    panic!("expected: {}, actual: {}", expected, actual);
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_q_mat_mul() {
    use candle_core::Module;

    use crate::Device;
    use crate::Tensor;

    let device = Device::new().await.unwrap();

    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

    let mut q_data = [[0f32; 256]; 256];
    q_data[0][0] = 1.;
    q_data[0][1] = 2.;
    q_data[0][2] = 3.;
    q_data[1][0] = 3.;
    q_data[1][1] = 2.;
    q_data[1][2] = 1.;
    q_data[2][0] = 1.;
    q_data[2][1] = 5.;
    q_data[2][2] = 3.;

    let tensor = candle_core::Tensor::new(&q_data, &candle_core::Device::Cpu).unwrap();
    let quantized =
        candle_core::quantized::QTensor::quantize(&tensor, candle_core::quantized::GgmlDType::Q4K)
            .unwrap();
    let q_data = quantized.data().unwrap();
    let q_matrix = QMatrix::from_parts(
        device.wgpu_device(),
        &q_data,
        quantized.shape().dims().into(),
        GgmlType::Q4K,
    )
    .unwrap();
    let candle_q_matrix = candle_core::quantized::QMatMul::from_qtensor(quantized).unwrap();

    let mut tensor_data = vec![vec![0f32; 256]; 256];
    tensor_data[0][0] = 4.;
    tensor_data[0][1] = 5.;
    tensor_data[0][2] = 6.;
    tensor_data[1][0] = 6.;
    tensor_data[1][1] = 5.;
    tensor_data[1][2] = 21.;
    tensor_data[2][0] = 4.;
    tensor_data[2][1] = 6.;
    tensor_data[2][2] = 5.;

    let candle_input = candle_core::Tensor::from_iter(
        tensor_data.iter().flat_map(|x| x.iter().copied()),
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .reshape(&[256, 256])
    .unwrap();
    println!(
        "candle_input: {:?}",
        candle_input
            .narrow(0, 0, 3)
            .unwrap()
            .narrow(1, 0, 3)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    );
    let candle_output = candle_q_matrix.forward(&candle_input).unwrap();
    let candle_output = candle_output
        .narrow(0, 0, 3)
        .unwrap()
        .narrow(1, 0, 3)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    println!("candle_output: {:?}", candle_output);

    let input = Tensor::<2, f32>::new(&device, &tensor_data);

    let result = input.q_mat_mul(&q_matrix);
    let result = result.slice([0..3, 0..3]).as_slice().await.unwrap();
    println!("Result: {:?}", result);

    let expected = [[25f32, 48., 35.], [42., 127., 86.], [27., 45., 33.]];
    assert!((result[[0, 0]] - expected[0][0]).abs() < 3.);
    assert!((result[[0, 1]] - expected[0][1]).abs() < 3.);
    assert!((result[[0, 2]] - expected[0][2]).abs() < 3.);
    assert!((result[[1, 0]] - expected[1][0]).abs() < 3.);
    assert!((result[[1, 1]] - expected[1][1]).abs() < 3.);
    assert!((result[[1, 2]] - expected[1][2]).abs() < 3.);
    assert!((result[[2, 0]] - expected[2][0]).abs() < 3.);
    assert!((result[[2, 1]] - expected[2][1]).abs() < 3.);
    assert!((result[[2, 2]] - expected[2][2]).abs() < 3.);
}

#[derive(Clone)]
pub struct QMatrix {
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType,
}

impl QMatrix {
    pub(crate) fn read<R: std::io::Read + std::io::Seek>(
        device: &wgpu::Device,
        metadata: &GgufTensorMetadata,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Self, GgufReadError> {
        let bytes = metadata.read_tensor_bytes(reader, tensor_data_offset)?;
        let shape = metadata.shape.iter().map(|x| *x as usize).collect();
        let ty = metadata.ty;
        QMatrix::from_parts(device, &bytes, shape, ty)
    }

    pub(crate) fn from_parts(
        device: &wgpu::Device,
        bytes: &[u8],
        shape: Box<[usize]>,
        ty: GgmlType,
    ) -> Result<Self, GgufReadError> {
        let bytes: Box<[u8]> = match ty {
            GgmlType::Q4_0 => bytemuck::cast_slice::<_, BlockQ4_0>(&bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q5_0 => bytemuck::cast_slice::<_, BlockQ5_0>(&bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q8_0 => bytemuck::cast_slice::<_, BlockQ8_0>(&bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q4K => bytemuck::cast_slice::<_, BlockQ4K>(&bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q6K => bytemuck::cast_slice::<_, BlockQ6K>(&bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            _ => todo!(),
        };
        let buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }),
        );

        Ok(QMatrix {
            shape,
            buffer,
            datatype: ty,
        })
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub(crate) struct UntypedQMatMul {
    #[allow(unused)]
    sparse_kernel: OnceLock<GenericKernel>,
    first_dim_dense_kernel: OnceLock<GenericKernel>,
    input_datatype: DataTypeEnum,
    matrix: QMatrix,
}

impl UntypedQMatMul {
    pub(crate) const fn new(datatype: DataTypeEnum, matrix: QMatrix) -> Self {
        Self {
            sparse_kernel: OnceLock::new(),
            first_dim_dense_kernel: OnceLock::new(),
            input_datatype: datatype,
            matrix,
        }
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn work_group_dispatch(&self, a_shape: &[usize]) -> [u32; 3] {
        let n = self.matrix.shape[0];
        let m = a_shape[1];
        [
            (n as u32).div_ceil(self.elements_per_block() * self.work_group_size()[0]),
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
        self.first_dim_dense_kernel.get_or_init(|| {
            // based on https://siboehm.com/articles/22/CUDA-MMM
            let mut generic_kernel = GenericKernel::new();

            generic_kernel.set_workgroup_size(self.work_group_size());

            let mut kernel = String::new();

            let datatype = self.input_datatype;

            let input_a = generic_kernel.add_tensor_input(2, false, datatype);
            let input_b = generic_kernel.add_q_matrix_input(2, self.matrix.datatype);
            let output = generic_kernel.add_tensor_input(2, true, datatype);

            let global_id = generic_kernel.global_id();
            let elements_per_block = self.elements_per_block();

            let k_size = input_a.shape_binding(0);
            let m_size = input_a.shape_binding(1);
            let n_size = input_b.shape_binding(0);

            writeln!(&mut kernel, "let x = {global_id}.x;").unwrap();
            writeln!(&mut kernel, "let y = {global_id}.y;").unwrap();

            writeln!(
                &mut kernel,
                "var acc = array<{datatype}, {elements_per_block}>();"
            )
            .unwrap();

            // Calculate one block sized group
            writeln!(
                &mut kernel,
                "if x * {elements_per_block} < {n_size} && y < {m_size} {{"
            )
            .unwrap();

            writeln!(&mut kernel, "for (var k = 0u; k < {k_size}; k += 1u) {{").unwrap();

            writeln!(&mut kernel, "let a = {input_a}[y * {k_size} + k];").unwrap();
            writeln!(
                &mut kernel,
                "let chunk = {input_b}[(k * {n_size}) / {elements_per_block} + x];"
            )
            .unwrap();

            dequantize_block(
                &mut kernel,
                self.matrix.datatype,
                "chunk".to_string(),
                DataTypeEnum::F32,
                |i, data, code| {
                    writeln!(code, "acc[{i}] += a * {data};").unwrap();
                },
            );

            writeln!(&mut kernel, "}}").unwrap();
            // writeln!(
            //     &mut kernel,
            //     "let chunk = {input_b}[(y * {n_size}) / {elements_per_block} + x];"
            // )
            // .unwrap();

            // dequantize_block(
            //     &mut kernel,
            //     self.matrix.datatype,
            //     "chunk".to_string(),
            //     DataTypeEnum::F32,
            //     |i, data, code| {
            //         writeln!(code, "acc[{i}] += {data};").unwrap();
            //     },
            // );

            writeln!(&mut kernel, "}}").unwrap();

            // Then write the result
            writeln!(
                &mut kernel,
                "for (var x_offset = 0u; x_offset < {elements_per_block}; x_offset += 1u) {{"
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "let x_output_index = x * {elements_per_block} + x_offset;"
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "if x_output_index < {n_size} && y < {m_size} {{"
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "let output_index = x_output_index + y * {n_size};"
            )
            .unwrap();
            writeln!(&mut kernel, "{output}[output_index] = acc[x_offset];").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            println!("{}", kernel);

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
                (a_shape[0] * b_shape[1] * input.datatype().element_size()) as u64,
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor = TensorData::new_from_buffer(
            device,
            output_buf,
            &[a_shape[0], b_shape[1]],
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
        assert_eq!(input.layout().shape()[1], matrix_shape[0]);
        let module = self.compile();

        let a_shape = input.layout().shape();
        let b_shape = matrix_shape;
        assert_eq!(*output_tensor.layout().shape(), [a_shape[0], b_shape[1]]);

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

fn dequantize_block(
    kernel: &mut String,
    ty: GgmlType,
    chunk: String,
    datatype: DataTypeEnum,
    process_element: impl FnMut(String, String, &mut String),
) {
    let out = match ty {
        GgmlType::Q4_0 => BlockQ4_0::dequantize_block(chunk, datatype, process_element),
        GgmlType::Q5_0 => BlockQ5_0::dequantize_block(chunk, datatype, process_element),
        GgmlType::Q8_0 => BlockQ8_0::dequantize_block(chunk, datatype, process_element),
        GgmlType::Q4K => BlockQ4K::dequantize_block(chunk, datatype, process_element),
        GgmlType::Q6K => BlockQ6K::dequantize_block(chunk, datatype, process_element),
        _ => todo!(),
    };
    *kernel += &out;
}

trait WgslQuantizedType: GgufBlock {
    const GGML_TYPE: GgmlType;

    fn dequantize_block(
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnMut(String, String, &mut String),
    ) -> String;

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result;
}

const fn center_of_bit_space(bits: u8) -> u8 {
    1 << (bits - 1)
}

const fn shift_right_scale(shift_bits: u8) -> f32 {
    1.0 / (1 << shift_bits) as f32
}

impl WgslQuantizedType for BlockQ4_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q4_0;

    fn dequantize_block(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut String),
    ) -> String {
        const CENTER: u8 = center_of_bit_space(4);
        const RIGHT_SHIFT_4_SCALE: f32 = shift_right_scale(4);

        let half_block_size = BlockQ4_0::BLOCK_SIZE as u8 / 2;
        let weights_size_u32 = BlockQ4_0::WEIGHTS_SIZE as u8 / 4;
        let mut code = String::new();
        writeln!(
            &mut code,
            "const SHIFT_SCALES = vec2({datatype}(1.0), {datatype}({RIGHT_SHIFT_4_SCALE}));"
        )
        .unwrap();
        writeln!(&mut code, "const CENTER = vec2({datatype}({CENTER}));").unwrap();

        writeln!(&mut code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(&mut code, "var output_index = 0;").unwrap();
        writeln!(&mut code, "for (var i = 0; i < {weights_size_u32}; i++) {{").unwrap();
        writeln!(&mut code, "let weight_chunk = {chunk}.data[i];").unwrap();
        writeln!(
            &mut code,
            "let weight_chunk_bytes = unpack4xU8(weight_chunk);"
        )
        .unwrap();
        for offset in 0..4 {
            writeln!(
                &mut code,
                "let byte{offset} = weight_chunk_bytes[{}];",
                offset
            )
            .unwrap();
            writeln!(
                &mut code,
                "let data{offset} = vec2(byte{offset} & 0x0F, byte{offset} & 0xF0);"
            )
            .unwrap();
            writeln!(
            &mut code,
            "let data_float{offset} = ((vec2<{datatype}>(data{offset}) * SHIFT_SCALES) - CENTER) * scale;"
        )
        .unwrap();
            process_element(
                "output_index".to_string(),
                format!("data_float{offset}.x"),
                &mut code,
            );
            process_element(
                format!("output_index + {half_block_size}"),
                format!("data_float{offset}.y"),
                &mut code,
            );
            writeln!(&mut code, "output_index += 1;").unwrap();
        }
        writeln!(&mut code, "}}").unwrap();

        code
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q4_0_type(f)
    }
}

impl WgslQuantizedType for BlockQ5_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q5_0;

    fn dequantize_block(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut String),
    ) -> String {
        const CENTER: u8 = center_of_bit_space(5);
        const FIFTH_BIT: u8 = 0x10;

        let half_block_size = BlockQ5_0::BLOCK_SIZE as u8 / 2;
        let low_weights_size_u32 = BlockQ5_0::WEIGHTS_LOW_BITS_SIZE as u8 / 4;
        let mut code = String::new();
        writeln!(&mut code, "const CENTER = vec2({datatype}({CENTER}));").unwrap();

        writeln!(&mut code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(&mut code, "var output_index = 0;").unwrap();
        writeln!(&mut code, "let high_bits = {chunk}.data_high_bits[0];").unwrap();
        writeln!(
            &mut code,
            "for (var i = 0u; i < {low_weights_size_u32}; i++) {{"
        )
        .unwrap();
        writeln!(
            &mut code,
            "let low_weight_chunk = {chunk}.data_low_bits[i];"
        )
        .unwrap();
        writeln!(
            &mut code,
            "let low_weight_chunk_bytes = unpack4xU8(low_weight_chunk);"
        )
        .unwrap();
        for offset in 0..4 {
            writeln!(
                &mut code,
                "let byte{offset} = low_weight_chunk_bytes[{}];",
                offset
            )
            .unwrap();
            writeln!(
                &mut code,
                "let data{offset} = vec2((byte{offset} & 0x0F) | ((high_bits >> (i*4 + {offset})) << 4) & {FIFTH_BIT}, (byte{offset} >> 4) | (high_bits >> (i*4 + {offset} + 12)) & {FIFTH_BIT});"
            )
            .unwrap();
            writeln!(
                &mut code,
                "let data_float{offset} = (vec2<{datatype}>(data{offset}) - CENTER) * scale;"
            )
            .unwrap();
            process_element(
                "output_index".to_string(),
                format!("data_float{offset}.x"),
                &mut code,
            );
            process_element(
                format!("output_index + {half_block_size}"),
                format!("data_float{offset}.y"),
                &mut code,
            );
            writeln!(&mut code, "output_index += 1;").unwrap();
        }
        writeln!(&mut code, "}}").unwrap();

        code
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q5_0_type(f)
    }
}

impl WgslQuantizedType for BlockQ8_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q8_0;

    fn dequantize_block(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut String),
    ) -> String {
        let weights_size_u32 = BlockQ8_0::WEIGHTS_SIZE as u8 / 4;
        let mut code = String::new();

        writeln!(&mut code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(&mut code, "var output_index = 0;").unwrap();
        writeln!(
            &mut code,
            "for (var i = 0u; i < {weights_size_u32}; i++) {{"
        )
        .unwrap();
        writeln!(&mut code, "let weight_chunk = {chunk}.data[i];").unwrap();
        writeln!(
            &mut code,
            "let weight_chunk_bytes = unpack4xI8(weight_chunk);"
        )
        .unwrap();
        for offset in 0..4 {
            writeln!(
                &mut code,
                "let data{offset} = weight_chunk_bytes[{}];",
                offset
            )
            .unwrap();
            writeln!(
                &mut code,
                "let data_float{offset} = {datatype}(data{offset}) * scale;"
            )
            .unwrap();
            process_element(
                "output_index".to_string(),
                format!("data_float{offset}"),
                &mut code,
            );
            writeln!(&mut code, "output_index += 1;").unwrap();
        }
        writeln!(&mut code, "}}").unwrap();

        code
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q8_0_type(f)
    }
}

impl WgslQuantizedType for BlockQ4K {
    const GGML_TYPE: GgmlType = GgmlType::Q4K;

    fn dequantize_block(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut String),
    ) -> String {
        const SIX_BITS_MASK: u32 = 0b0011_1111_0011_1111_0011_1111_0011_1111;
        const MSB_TWO_BITS_MASK: u32 = 0b1100_0000_1100_0000_1100_0000_1100_0000;
        const LOW_FOUR_BITS: u32 = 0b0000_1111_0000_1111_0000_1111_0000_1111;
        const HIGH_FOUR_BITS: u32 = 0b1111_0000_1111_0000_1111_0000_1111_0000;
        const MSB_SCALES_MASK: u32 = LOW_FOUR_BITS;
        const MSB_OFFSET_MASK: u32 = HIGH_FOUR_BITS;

        let mut code = String::new();

        writeln!(
            &mut code,
            "let super_block_scale = {datatype}({chunk}.scale);"
        )
        .unwrap();
        writeln!(&mut code, "let super_block_min = {datatype}({chunk}.min);").unwrap();

        writeln!(&mut code, "let first_four_bytes = {chunk}.scales[0];").unwrap();
        writeln!(&mut code, "let middle_four_bytes = {chunk}.scales[1];").unwrap();
        writeln!(&mut code, "let last_four_bytes = {chunk}.scales[2];").unwrap();

        // Extracts this bit pattern. The first 6 bits of the first
        // 4 bytes are the scales. The first 6 bits of the second 4
        // bytes are the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // __000000|__111111|__222222|__333333|__000000|__111111
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // __222222|__333333|________|________|________|________
        writeln!(
            &mut code,
            "let first_scales = first_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();
        writeln!(
            &mut code,
            "let first_offsets = middle_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();

        // Extracts this bit pattern. The first 2 bits of the first
        // 4 bytes are the scales. The first 2 bits of the second 4
        // bytes are the offsets.
        // The first 4 bits of the last 4 bytes are the lower 4 bits
        // of the scales. The second 4 bits of the last 4 bytes are
        // the lower 4 bits of the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // 44______|55______|66______|77______|44______|55______
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // 66______|77______|44444444|55555555|66666666|77777777
        writeln!(
            &mut code,
            "let msb_scales = (first_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
        )
        .unwrap();
        writeln!(
            &mut code,
            "let lsb_scales = last_four_bytes & {MSB_SCALES_MASK};"
        )
        .unwrap();
        writeln!(&mut code, "let second_scales = msb_scales | lsb_scales;").unwrap();
        writeln!(
            &mut code,
            "let msb_offsets = (middle_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
        )
        .unwrap();
        writeln!(
            &mut code,
            "let lsb_offsets = (last_four_bytes & {MSB_OFFSET_MASK}) >> 4;"
        )
        .unwrap();
        writeln!(&mut code, "let second_offsets = msb_offsets | lsb_offsets;").unwrap();

        writeln!(code, "var weight_index = 0u;").unwrap();
        writeln!(code, "var output_index = 0u;").unwrap();

        let mut run_chunk = |scales: &str, offsets: &str| {
            writeln!(code, "{{").unwrap();
            writeln!(
                code,
                "let scales = vec4<{datatype}>(unpack4xU8({scales})) * super_block_scale;"
            )
            .unwrap();
            writeln!(code, "let low_scales = scales.xz;").unwrap();
            writeln!(
                code,
                "let high_scales = scales.yw * {};",
                shift_right_scale(4)
            )
            .unwrap();
            writeln!(
                code,
                "let offsets = vec4<{datatype}>(unpack4xU8({offsets})) * super_block_min;"
            )
            .unwrap();
            writeln!(code, "let low_offsets = offsets.xz;").unwrap();
            writeln!(code, "let high_offsets = offsets.yw;").unwrap();

            for suffix in ["x", "y"] {
                writeln!(code, "for (var i = 0u; i < 32u; i += 4) {{").unwrap();
                writeln!(code, "let weight_chunk = {chunk}.data[weight_index];").unwrap();
                writeln!(code, "let weight_chunk_low = vec4<{datatype}>(unpack4xU8(weight_chunk & {LOW_FOUR_BITS}));").unwrap();
                writeln!(code, "let weight_chunk_high = vec4<{datatype}>(unpack4xU8(weight_chunk & {HIGH_FOUR_BITS}));").unwrap();
                writeln!(code, "let low_result = weight_chunk_low * low_scales.{suffix} - low_offsets.{suffix};").unwrap();
                writeln!(code, "let high_result = weight_chunk_high * high_scales.{suffix} - high_offsets.{suffix};").unwrap();
                for i in 0..4 {
                    process_element(
                        format!("output_index + i + {i}u"),
                        format!("low_result[{i}]"),
                        &mut code,
                    );
                    process_element(
                        format!("output_index + i + {i}u + 32u"),
                        format!("high_result[{i}]"),
                        &mut code,
                    );
                }
                writeln!(code, "weight_index += 1;").unwrap();
                writeln!(code, "}}").unwrap();
                writeln!(code, "output_index += 64;").unwrap();
            }
            writeln!(code, "}}").unwrap();
        };

        run_chunk("first_scales", "first_offsets");
        run_chunk("second_scales", "second_offsets");

        code
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q4_k_type(f)
    }
}

impl WgslQuantizedType for BlockQ6K {
    const GGML_TYPE: GgmlType = GgmlType::Q6K;

    fn dequantize_block(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut String),
    ) -> String {
        const CENTER_SIX_BIT: i8 = center_of_bit_space(6) as i8;
        const TWO_BITS: u8 = 0b11;
        const FOUR_BITS: u8 = 0b1111;

        // This implementation is very unoptimized because of the byte u32 indexing
        fn index_signed_bytes(u32_array: impl Display, byte_index: impl Display) -> String {
            format!("unpack4xI8({u32_array}[({byte_index}) / 4u])[({byte_index}) % 4]")
        }

        fn index_bytes(u32_array: impl Display, byte_index: impl Display) -> String {
            format!("unpack4xU8({u32_array}[({byte_index}) / 4u])[({byte_index}) % 4]")
        }

        let mut code = String::new();

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();

        writeln!(
            code,
            "for (var chunk_index = 0u; chunk_index < 2u; chunk_index += 1u) {{",
        )
        .unwrap();
        writeln!(code, "let output_index = chunk_index * 128u;").unwrap();
        writeln!(code, "let lower_index = chunk_index * 64u;").unwrap();
        writeln!(code, "let high_index = chunk_index * 32u;").unwrap();
        writeln!(code, "let scale_index = chunk_index * 8u;").unwrap();
        writeln!(
            code,
            "for (var high_byte_index = 0u; high_byte_index < 32u; high_byte_index += 1u) {{"
        )
        .unwrap();
        writeln!(
            code,
            "let scale_index = scale_index + high_byte_index / 16u;"
        )
        .unwrap();
        writeln!(
            code,
            "let high_byte = {};",
            index_bytes(
                format!("{chunk}.data_high_bits"),
                "high_index + high_byte_index"
            )
        )
        .unwrap();
        writeln!(
            code,
            "let first_low_byte = {};",
            index_bytes(
                format!("{chunk}.data_low_bits"),
                "lower_index + high_byte_index"
            )
        )
        .unwrap();
        writeln!(
            code,
            "let second_low_byte = {};",
            index_bytes(
                format!("{chunk}.data_low_bits"),
                "lower_index + high_byte_index + 32"
            )
        )
        .unwrap();

        writeln!(code, "let first_two_bits = high_byte & {TWO_BITS};").unwrap();
        writeln!(
            code,
            "let first_high_nibble = first_low_byte & {FOUR_BITS};"
        )
        .unwrap();
        writeln!(code, "let first_merged = {datatype}((first_two_bits << 4) | first_high_nibble) - {datatype}({CENTER_SIX_BIT});").unwrap();
        process_element(
            format!("output_index + high_byte_index"),
            format!(
                "scale * {datatype}({}) * first_merged",
                index_signed_bytes(format!("{chunk}.scales"), "scale_index")
            ),
            &mut code,
        );

        writeln!(code, "let second_two_bits = (high_byte >> 2) & {TWO_BITS};").unwrap();
        writeln!(
            code,
            "let second_high_nibble = second_low_byte & {FOUR_BITS};"
        )
        .unwrap();
        writeln!(code, "let second_merged = {datatype}((second_two_bits << 4) | second_high_nibble) - {datatype}({CENTER_SIX_BIT});").unwrap();
        process_element(
            format!("output_index + high_byte_index + 32"),
            format!(
                "scale * {datatype}({}) * second_merged",
                index_signed_bytes(format!("{chunk}.scales"), "scale_index + 2")
            ),
            &mut code,
        );

        writeln!(code, "let third_two_bits = (high_byte >> 4) & {TWO_BITS};").unwrap();
        writeln!(code, "let third_high_nibble = first_low_byte >> 4;").unwrap();
        writeln!(code, "let third_merged = {datatype}((third_two_bits << 4) | third_high_nibble) - {datatype}({CENTER_SIX_BIT});").unwrap();
        process_element(
            format!("output_index + high_byte_index + 64"),
            format!(
                "scale * {datatype}({}) * third_merged",
                index_signed_bytes(format!("{chunk}.scales"), "scale_index + 4")
            ),
            &mut code,
        );

        writeln!(code, "let fourth_two_bits = (high_byte >> 6) & {TWO_BITS};").unwrap();
        writeln!(code, "let fourth_high_nibble = second_low_byte >> 4;").unwrap();
        writeln!(code, "let fourth_merged = {datatype}((fourth_two_bits << 4) | fourth_high_nibble) - {datatype}({CENTER_SIX_BIT});").unwrap();
        process_element(
            format!("output_index + high_byte_index + 96"),
            format!(
                "scale * {datatype}({}) * fourth_merged",
                index_signed_bytes(format!("{chunk}.scales"), "scale_index + 6")
            ),
            &mut code,
        );
        writeln!(code, "}}").unwrap();

        writeln!(code, "}}").unwrap();

        code
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q6_k_type(f)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_de_quantize_4_0_block() {
    fuzz_de_quantize::<BlockQ4_0>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_de_quantize_5_0_block() {
    fuzz_de_quantize::<BlockQ5_0>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_de_quantize_8_0_block() {
    fuzz_de_quantize::<BlockQ8_0>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_de_quantize_4_k_block() {
    fuzz_de_quantize::<BlockQ4K>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_de_quantize_6_k_block() {
    fuzz_de_quantize::<BlockQ6K>().await;
}

#[cfg(test)]
async fn fuzz_de_quantize<B: WgslQuantizedType + PartialEq + std::fmt::Debug>()
where
    rand::distr::StandardUniform:
        rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
{
    use pretty_assertions::assert_eq;
    use wgpu::util::DownloadBuffer;

    println!("testing f32...");
    test_de_quantize_block_inner::<B, f32>().await;

    async fn test_de_quantize_block_inner<
        B: WgslQuantizedType + PartialEq + std::fmt::Debug,
        T: DataType,
    >()
    where
        rand::distr::StandardUniform:
            rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
    {
        let dtype = T::WGSL_TYPE;
        let device = crate::Device::new().await.unwrap();
        let kernel_body = B::dequantize_block("block".to_string(), dtype, |i, data, code| {
            writeln!(code, "output[{i}] = {data};",).unwrap();
        });
        let mut kernel = String::new();
        writeln!(&mut kernel, "enable f16;").unwrap();
        B::write_type(&mut kernel).unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(0) var<storage, read> block: {};",
            B::GGML_TYPE
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(1) var<storage, read_write> output: array<{dtype}, {}>;",
            B::BLOCK_SIZE
        )
        .unwrap();
        writeln!(&mut kernel, "@compute @workgroup_size(1, 1, 1)").unwrap();
        writeln!(&mut kernel, "fn main() {{").unwrap();
        writeln!(&mut kernel, "{}", kernel_body).unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        let bind_group_layout =
            device
                .wgpu_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let compute_pipeline_layout =
            device
                .wgpu_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let module = device.create_shader_module(kernel);
        let pipeline =
            device
                .wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&compute_pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
        for _ in 0..200 {
            let block_bytes: B::AsBytes = rand::random();
            let block = bytemuck::pod_read_unaligned::<B>(block_bytes.as_ref());
            if !block.finite() {
                continue;
            }
            let block_wgsl = block.into_wgsl_bytes();
            assert_eq!(block, B::from_wgsl_bytes(block_wgsl));
            let output =
                device
                    .wgpu_device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&vec![T::zero(); B::BLOCK_SIZE]),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
            let bind_group = device
                .wgpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &device.wgpu_device().create_buffer_init(
                                    &wgpu::util::BufferInitDescriptor {
                                        label: None,
                                        contents: block_wgsl.as_ref(),
                                        usage: wgpu::BufferUsages::STORAGE
                                            | wgpu::BufferUsages::COPY_SRC,
                                    },
                                ),
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &output,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                });

            let mut encoder = device
                .wgpu_device()
                .create_command_encoder(&Default::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = [1, 1, 1];
                cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z);
            }
            device.wgpu_queue().submit(Some(encoder.finish()));

            let (sender, receiver) = futures_channel::oneshot::channel();
            DownloadBuffer::read_buffer(
                device.wgpu_device(),
                device.wgpu_queue(),
                &output.slice(..),
                move |result| {
                    _ = sender.send(result);
                },
            );
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
            let output = receiver
                .await
                .map_err(|_| wgpu::BufferAsyncError)
                .unwrap()
                .unwrap();

            let ouptut_as_floats = bytemuck::cast_slice::<_, T>(&*output);
            let expected_result = block
                .dequantize()
                .as_ref()
                .into_iter()
                .map(|x| T::from_f32(*x))
                .collect::<Vec<_>>();
            assert_eq!(ouptut_as_floats, expected_result);
        }
    }
}
