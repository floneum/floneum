use std::{
    fmt::Write,
    sync::{Arc, OnceLock},
};

use fusor_gguf::{BlockQ4_0, GgmlType, GgufReadError, GgufTensorMetadata};
use wgpu::{CommandEncoder, util::DeviceExt};

use crate::{
    DataType, DataTypeEnum, Device, PerformanceQueries, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    kernel::{GenericKernel, KernelGlobalSpace, KernelInputValue},
    padded_tensor_size,
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

    fn thread_block_m_size(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn work_group_block_m_size(&self) -> u32 {
        self.thread_block_m_size() * 2
    }

    fn work_group_block_n_size(&self) -> u32 {
        8
    }

    fn work_group_block_k_size(&self) -> u32 {
        8
    }

    fn work_group_size_element(&self) -> u32 {
        (self.work_group_block_n_size() * self.work_group_block_m_size())
            / self.thread_block_m_size()
    }

    fn work_group_size(&self) -> [u32; 3] {
        [
            self.work_group_size_element(),
            self.work_group_size_element(),
            1,
        ]
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
            let work_group_block_k_size = self.work_group_block_k_size();
            let work_group_block_n_size = self.work_group_block_n_size();
            let work_group_block_m_size = self.work_group_block_m_size();
            let thread_block_m_size = self.thread_block_m_size();

            let input_a = generic_kernel.add_tensor_input(2, false, datatype);
            let input_b = generic_kernel.add_tensor_input(2, false, datatype);
            let output = generic_kernel.add_tensor_input(2, true, datatype);

            let cache_a = generic_kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                datatype,
                (work_group_block_k_size * work_group_block_k_size).to_string(),
            );
            let cache_b = generic_kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                datatype,
                (work_group_block_n_size * work_group_block_n_size).to_string(),
            );

            let workgroup_index = generic_kernel.workgroup_index();
            let workgroup_local_index = generic_kernel.workgroup_local_index();

            let m_size = input_a.shape_binding(0);
            let n_size = input_b.shape_binding(1);
            let k_size = input_a.shape_binding(1);

            // The index of the workgroup block destination
            writeln!(&mut kernel, "let block_row = {workgroup_index}.y;").unwrap();
            writeln!(&mut kernel, "let block_col = {workgroup_index}.x;").unwrap();
            // The index of the thread within the workgroup block destination
            writeln!(&mut kernel, "let thread_col = {workgroup_local_index} % {work_group_block_n_size};").unwrap();
            writeln!(&mut kernel, "let thread_row = {workgroup_local_index} / {work_group_block_n_size};").unwrap();

            // The index of the thread within the workgroup block source
            writeln!(&mut kernel, "let a_thread_col = {workgroup_local_index} % {work_group_block_k_size};").unwrap();
            writeln!(&mut kernel, "let a_thread_row = {workgroup_local_index} / {work_group_block_k_size};").unwrap();
            writeln!(&mut kernel, "let b_thread_col = {workgroup_local_index} % {work_group_block_n_size};").unwrap();
            writeln!(&mut kernel, "let b_thread_row = {workgroup_local_index} / {work_group_block_n_size};").unwrap();
            writeln!(&mut kernel, "var results: array<{datatype}, {thread_block_m_size}>;").unwrap();
            writeln!(&mut kernel, "let a_row = {k_size} * (a_thread_row + block_row * {work_group_block_m_size});").unwrap();
            writeln!(&mut kernel, "var a_col = a_thread_col;").unwrap();
            writeln!(&mut kernel, "var b_row = {n_size} * b_thread_row;").unwrap(); 
            writeln!(&mut kernel, "let b_col = b_thread_col + block_col * {work_group_block_n_size};").unwrap();

            // Loop over each workgroup block in the K dimension
            writeln!(&mut kernel, "for (var block_index = 0u; block_index < {k_size}; block_index += {work_group_block_k_size}) {{").unwrap();

            // Load the workgroup block data into memory
            writeln!(&mut kernel, "let a_index = a_row + a_col;").unwrap();
            writeln!(&mut kernel, "{cache_a}[a_thread_row * {work_group_block_k_size} + a_thread_col] = {input_a}[a_index];").unwrap();
            writeln!(&mut kernel, "let b_index = b_row + b_col;").unwrap();
            writeln!(&mut kernel, "{cache_b}[b_thread_row * {work_group_block_n_size} + b_thread_col] = {input_b}[b_index];").unwrap();

            writeln!(&mut kernel, "workgroupBarrier();").unwrap();

            writeln!(&mut kernel, "a_col += {work_group_block_k_size};").unwrap();
            writeln!(&mut kernel, "b_row += {work_group_block_k_size} * {n_size};").unwrap();

            writeln!(&mut kernel, "for (var dot_index = 0u; dot_index < {work_group_block_k_size}; dot_index += 1u) {{").unwrap();
            writeln!(&mut kernel, "let tmp = {cache_b}[dot_index * {work_group_block_n_size} + thread_col];").unwrap();
            writeln!(&mut kernel, "for (var result_index = 0u; result_index < {thread_block_m_size}; result_index += 1u) {{").unwrap();
            writeln!(&mut kernel, "results[result_index] += {cache_a}[(thread_row * {thread_block_m_size} + result_index) * {work_group_block_k_size} + dot_index] * tmp;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "}}").unwrap();

            writeln!(&mut kernel, "workgroupBarrier();").unwrap();

            writeln!(&mut kernel, "}}").unwrap();

            // Write out the results
            writeln!(&mut kernel, "let start_output_row = thread_row * {thread_block_m_size} + block_row * {work_group_block_m_size};").unwrap();
            writeln!(&mut kernel, "let start_output_col = thread_col + block_col * {work_group_block_n_size};").unwrap();
            writeln!(&mut kernel, "for (var result_index = 0u; result_index < {thread_block_m_size}; result_index += 1u) {{").unwrap();
            writeln!(&mut kernel, "let output_row = start_output_row + result_index;").unwrap();
            writeln!(&mut kernel, "let output_col = start_output_col;").unwrap();
            writeln!(&mut kernel, "let output_index = output_row * {n_size} + output_col;").unwrap();
            writeln!(&mut kernel, "{output}[output_index] = results[result_index];").unwrap();
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

        let work_group_size = self.work_group_size();
        let thread_block_m_size = self.thread_block_m_size();

        let workgroup_dispatch_size = [
            (a_shape[0] as u32).div_ceil(work_group_size[0] * thread_block_m_size),
            (b_shape[1] as u32).div_ceil(work_group_size[1]),
            1,
        ];

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

const fn center_of_bit_space(bits: u8) -> u8 {
    1 << (bits - 1)
}

const fn shift_right_scale(shift_bits: u8) -> f32 {
    1.0 / (1 << shift_bits) as f32
}

fn de_quantize_block_q4_0(
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

    println!("{}", code);
    code
}

#[cfg(test)]
#[tokio::test]
async fn test_de_quantize_4_0_block() {
    fuzz_de_quantize::<BlockQ4_0>().await;
}

#[cfg(test)]
async fn fuzz_de_quantize<B: fusor_gguf::GgufBlock + PartialEq + std::fmt::Debug>()
where
    rand::distr::StandardUniform: rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
{
    use crate::quantized_types_wgsl::write_q4_0_type;
    use pretty_assertions::assert_eq;
    use wgpu::util::DownloadBuffer;

    println!("testing f32...");
    test_de_quantize_4_0_block_inner::<B, f32>().await;
    println!("testing f16...");
    test_de_quantize_4_0_block_inner::<B, half::f16>().await;

    async fn test_de_quantize_4_0_block_inner<
        B: fusor_gguf::GgufBlock + PartialEq + std::fmt::Debug,
        T: DataType,
    >()
    where
        rand::distr::StandardUniform:
            rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
    {
        let dtype = T::WGSL_TYPE;
        let device = crate::Device::new().await.unwrap();
        let kernel_body = de_quantize_block_q4_0("block".to_string(), dtype, |i, data, code| {
            writeln!(code, "output[{i}] = {data};",).unwrap();
        });
        let mut kernel = String::new();
        writeln!(&mut kernel, "enable f16;").unwrap();
        write_q4_0_type(&mut kernel).unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(0) var<storage, read> block: {};",
            GgmlType::Q4_0
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
        for _ in 0..100 {
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
                        contents: bytemuck::cast_slice(&[T::zero(); BlockQ4_0::BLOCK_SIZE]),
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
