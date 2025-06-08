use enumset::{EnumSet, EnumSetType};
use fusor_gguf::GgmlType;
use std::fmt::{Debug, Write};
use std::sync::OnceLock;
use wgpu::{BindGroupLayout, CommandEncoder, PipelineCompilationOptions, util::DeviceExt};

use crate::mir::inputs::{KernelInputValue, QBufferInput, QInfoInput, TensorBufferInput, TensorInfoInput};
use crate::quantized_types_wgsl::{
    write_q4_0_type, write_q4_k_type, write_q5_0_type, write_q6_k_type, write_q8_0_type,
};
use crate::{DataTypeEnum, Device};

use super::function::Function;
use super::globals::{ArrayType, KernelGlobal, KernelGlobalSpace, KernelGlobalType};
use super::inputs::{
    FloatInput, IntegerInput, KernelInput, KernelInputType, QMatrixInput,
    TensorInput,
};
use super::workgroup_shape::WorkgroupShape;

#[derive(EnumSetType, Debug)]
pub(crate) enum EnabledBuiltins {
    GlobalId,
    SubgroupSize,
    WorkgroupIndex,
    WorkgroupLocalIndex,
    SubgroupIndex,
    SubgroupLocalIndex,
    SubgroupsPerWorkgroup,
}

#[derive(Debug)]
pub(crate) struct GenericKernel {
    workgroup_size: [u32; 3],
    max_binding: u32,
    registered_bindings: Vec<u32>,
    max_function_id: u32,
    max_global_id: u32,
    inputs: Vec<KernelInput>,
    functions: Vec<Function>,
    globals: Vec<KernelGlobal>,
    enabled_builtins: EnumSet<EnabledBuiltins>,
    quantized_type_definitions: EnumSet<GgmlType>,
    kernel: OnceLock<wgpu::ShaderModule>,
    body: String,
}

impl Default for GenericKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GenericKernel {
    pub(crate) fn new() -> Self {
        Self {
            workgroup_size: [1, 1, 1],
            inputs: Default::default(),
            max_binding: 0,
            registered_bindings: Vec::new(),
            functions: Default::default(),
            max_function_id: 0,
            globals: Default::default(),
            max_global_id: 0,
            enabled_builtins: Default::default(),
            quantized_type_definitions: Default::default(),
            kernel: OnceLock::new(),
            body: String::new(),
        }
    }

    pub(crate) fn pre_register_binding(&mut self, binding: u32) {
        self.registered_bindings.push(binding);
    }

    pub(crate) fn push_body(&mut self, body: &str) {
        self.body.push_str(body);
    }

    pub(crate) fn set_workgroup_size(&mut self, workgroup_size: impl Into<WorkgroupShape>) {
        let workgroup_size = workgroup_size.into().shape();
        assert!(
            workgroup_size.iter().product::<u32>() <= 256,
            "{workgroup_size:?} product must be <= 256"
        );
        self.workgroup_size = workgroup_size;
    }

    pub(crate) fn take_binding(&mut self, or: impl FnOnce(u32) -> KernelInput) -> u32 {
        let index = self.max_binding as usize;
        self.max_binding += 1;
        let binding = if let Some(binding) = self.registered_bindings.get(index) {
            *binding
        } else {
            self.registered_bindings.push(index as u32);
            index as u32
        };
        if binding >= self.inputs.len() as u32 {
            self.inputs.push(or(binding));
        }
        binding
    }

    pub(crate) fn add_function(
        &mut self,
        ty: impl ToString,
        function_body: impl ToString,
        inputs: impl IntoIterator<Item = (String, String)>,
    ) -> Function {
        let inputs = inputs.into_iter().collect();
        let id = self.max_function_id;
        self.max_function_id += 1;
        let function = Function::new(id, ty.to_string(), function_body.to_string(), inputs);
        self.functions.push(function.clone());
        function
    }

    pub(crate) fn add_tensor_input(
        &mut self,
        rank: u32,
        mutable: bool,
        datatype: DataTypeEnum,
    ) -> TensorInput {
        let tensor_binding = self.take_binding(|tensor_binding| KernelInput {
            ty: KernelInputType::TensorBuffer(TensorBufferInput {
                tensor_binding,
                mutable,
                datatype: datatype.clone(),
            }),
        });
        let info_binding = self.take_binding(|info_binding| KernelInput {
            ty: KernelInputType::TensorInfo(TensorInfoInput { info_binding, rank }),
        });

        let input = TensorInput {
            tensor_binding,
            info_binding,
            rank,
            mutable,
            datatype,
        };

        input
    }

    pub(crate) fn add_q_matrix_input(&mut self, rank: u32, datatype: GgmlType) -> QMatrixInput {
        let matrix_binding = self.take_binding(|matrix_binding| KernelInput {
            ty: KernelInputType::QBuffer(QBufferInput {
                matrix_binding,
                datatype: datatype.clone(),
            }),
        });
        let info_binding = self.take_binding(|info_binding| KernelInput {
            ty: KernelInputType::QInfo(QInfoInput { info_binding, rank }),
        });

        let input = QMatrixInput {
            matrix_binding,
            info_binding,
            datatype,
            rank,
        };

        self.quantized_type_definitions |= datatype;

        input
    }

    pub(crate) fn add_integer_input(&mut self) -> IntegerInput {
        let index = self.take_binding(|index| KernelInput {
            ty: KernelInputType::Integer(IntegerInput { index }),
        });

        let input = IntegerInput { index };

        input
    }

    pub(crate) fn add_float_input(&mut self) -> FloatInput {
        let index = self.take_binding(|index| KernelInput {
            ty: KernelInputType::Float(FloatInput { index }),
        });

        let input = FloatInput { index };

        input
    }

    pub(crate) fn add_global_array(
        &mut self,
        space: KernelGlobalSpace,
        array_type: DataTypeEnum,
        size: String,
    ) -> KernelGlobal {
        let index = self.max_global_id;
        self.max_global_id += 1;
        let global = KernelGlobal::new(
            index,
            space,
            KernelGlobalType::Array(ArrayType::new(size, array_type)),
        );
        self.globals.push(global.clone());
        global
    }

    pub(crate) fn subgroup_size(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::SubgroupSize;
        "subgroup_size".to_string()
    }

    pub(crate) fn global_id(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::GlobalId;
        "global_id".to_string()
    }

    pub(crate) fn subgroup_local_index(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::SubgroupLocalIndex;
        "subgroup_local_id".to_string()
    }

    pub(crate) fn subgroup_index(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::SubgroupIndex;
        "subgroup_id".to_string()
    }

    pub(crate) fn workgroup_local_index(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::WorkgroupLocalIndex;
        "workgroup_local_id".to_string()
    }

    pub(crate) fn subgroups_per_workgroup(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::SubgroupsPerWorkgroup;
        "subgroups_per_workgroup".to_string()
    }

    pub(crate) fn workgroup_index(&mut self) -> String {
        self.enabled_builtins |= EnabledBuiltins::WorkgroupIndex;
        "workgroup_index".to_string()
    }

    pub(crate) fn bind_group_layout(&self, device: &crate::Device) -> BindGroupLayout {
        let mut entries = Vec::new();
        for input in &self.inputs {
            match &input.ty {
                KernelInputType::QBuffer(matrix) => {
                    // Matrix bytes
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: matrix.matrix_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
                KernelInputType::QInfo(matrix) => {
                    // Matrix info
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: matrix.info_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
                KernelInputType::TensorBuffer(tensor_input) => {
                    // Tensor weight
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: tensor_input.tensor_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: !tensor_input.mutable,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
                KernelInputType::TensorInfo(tensor_input) => {
                    // Tensor info
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: tensor_input.info_binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
                KernelInputType::Integer(integer_input) => {
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: integer_input.index,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                }
                KernelInputType::Float(float_input) => {
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: float_input.index,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
            }
        }

        device
            .wgpu_device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
    }

    fn compute_pipeline(
        &self,
        device: &crate::Device,
        bind_group_layout: &BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let compute_pipeline_layout =
            device
                .wgpu_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[bind_group_layout],
                    push_constant_ranges: &[],
                });
        let module = self.kernel.get_or_init(|| {
            let mut kernel = String::new();
            self.kernel(&mut kernel).unwrap();
            device.create_shader_module(kernel)
        });
        device
            .wgpu_device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&compute_pipeline_layout),
                module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
            })
    }

    fn create_bind_group(
        &self,
        device: &crate::Device,
        bind_group_layout: &BindGroupLayout,
        inputs: Vec<KernelInputValue>,
    ) -> wgpu::BindGroup {
        assert_eq!(self.inputs.len(), inputs.len(), "Input count mismatch");

        let mut entries = Vec::new();
        let mut owned_entries = Vec::new();
        fn create_u32_iter_buffer(
            device: &crate::Device,
            data: impl IntoIterator<Item = u32>,
        ) -> wgpu::Buffer {
            device
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&data.into_iter().collect::<Vec<_>>()),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        }
        let create_u32_buffer = |device: &crate::Device, data: u32| {
            device
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&data),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        };
        let create_f32_buffer = |device: &crate::Device, data: f32| {
            device
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&data),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        };
        for (input, value) in self.inputs.iter().zip(inputs.iter()) {
            match (&input.ty, value) {
                (KernelInputType::QBuffer(matrix_input), KernelInputValue::QBuffer(matrix)) => {
                    // Tensor weight
                    entries.push(wgpu::BindGroupEntry {
                        binding: matrix_input.matrix_binding,
                        resource: matrix.as_entire_binding(),
                    });
                }
                (KernelInputType::QInfo(matrix_input), KernelInputValue::QInfo(matrix)) => {
                    // Tensor info
                    owned_entries.push((
                        matrix_input.info_binding,
                        create_u32_iter_buffer(device, matrix.iter().map(|x| *x as u32)),
                    ));
                }
                (KernelInputType::TensorBuffer(tensor_input), KernelInputValue::TensorBuffer(tensor)) => {
                    // Tensor weight
                    entries.push(wgpu::BindGroupEntry {
                        binding: tensor_input.tensor_binding,
                        resource: tensor.as_entire_binding(),
                    });
                }
                (KernelInputType::TensorInfo(tensor_input), KernelInputValue::TensorInfo(tensor)) => {
                    // Tensor info
                    owned_entries.push((
                        tensor_input.info_binding,
                        create_u32_iter_buffer(
                            device,
                            std::iter::once(tensor.layout().offset() as u32).chain(
                                (0..tensor_input.rank).flat_map(|i| {
                                    [
                                        tensor.layout().strides()[i as usize] as u32,
                                        tensor.layout().shape()[i as usize] as u32,
                                    ]
                                }),
                            ),
                        ),
                    ));
                }
                (KernelInputType::Integer(integer_input), KernelInputValue::Integer(value)) => {
                    owned_entries.push((integer_input.index, create_u32_buffer(device, *value)));
                }
                (KernelInputType::Float(float_input), KernelInputValue::Float(value)) => {
                    owned_entries.push((float_input.index, create_f32_buffer(device, *value)));
                }
                _ => panic!("cannot bind {input:?} to {value:?}"),
            }
        }

        for (binding, resource) in &owned_entries {
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.as_entire_binding(),
            });
        }

        device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries: &entries,
            })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        inputs: Vec<KernelInputValue>,
        command_encoder: &mut CommandEncoder,
        workgroup_dispatch_size: [u32; 3],
    ) {
        let bind_group_layout = self.bind_group_layout(device);
        let bind_group = self.create_bind_group(device, &bind_group_layout, inputs);
        let pipeline = self.compute_pipeline(device, &bind_group_layout);

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = workgroup_dispatch_size;
            cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z);
        }
    }

    fn declare_quantized_types(&self, f: &mut String) -> std::fmt::Result {
        let q4_0 = GgmlType::Q4_0;
        if self.quantized_type_definitions.contains(q4_0) {
            write_q4_0_type(f)?;
        }

        let q5_0 = GgmlType::Q5_0;
        if self.quantized_type_definitions.contains(q5_0) {
            write_q5_0_type(f)?;
        }

        let q8_0 = GgmlType::Q8_0;
        if self.quantized_type_definitions.contains(q8_0) {
            write_q8_0_type(f)?;
        }

        let q4_k = GgmlType::Q4K;
        if self.quantized_type_definitions.contains(q4_k) {
            write_q4_k_type(f)?;
        }

        let q6_k = GgmlType::Q6K;
        if self.quantized_type_definitions.contains(q6_k) {
            write_q6_k_type(f)?;
        }

        Ok(())
    }

    fn kernel(&self, f: &mut String) -> std::fmt::Result {
        writeln!(f, "enable f16;")?;

        self.declare_quantized_types(f)?;

        for input in &self.inputs {
            write!(f, "{input}")?;
        }

        for global in &self.globals {
            write!(f, "{}", global.global_definition())?;
        }

        for function in &self.functions {
            write!(f, "{}", function.function_definition())?;
        }

        let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = self.workgroup_size;
        writeln!(
            f,
            "const BLOCKSIZE: u32 = {}u;",
            self.workgroup_size.iter().product::<u32>()
        )?;
        writeln!(
            f,
            "@compute @workgroup_size({workgroup_size_x}, {workgroup_size_y}, {workgroup_size_z})"
        )?;
        let mut built_ins = String::new();
        if self.enabled_builtins.contains(EnabledBuiltins::GlobalId)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupIndex)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupLocalIndex)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::WorkgroupLocalIndex)
        {
            built_ins.push_str("@builtin(global_invocation_id) global_id: vec3<u32>, ");
        }
        if self
            .enabled_builtins
            .contains(EnabledBuiltins::SubgroupSize)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupIndex)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupLocalIndex)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupsPerWorkgroup)
        {
            built_ins.push_str("@builtin(subgroup_size) subgroup_size: u32, ");
        }
        if self
            .enabled_builtins
            .contains(EnabledBuiltins::WorkgroupLocalIndex)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupIndex)
            | self
                .enabled_builtins
                .contains(EnabledBuiltins::SubgroupLocalIndex)
        {
            built_ins.push_str("@builtin(local_invocation_index) workgroup_local_id: u32, ");
        }
        if self
            .enabled_builtins
            .contains(EnabledBuiltins::WorkgroupIndex)
        {
            built_ins.push_str("@builtin(workgroup_id) workgroup_index: vec3<u32>, ");
        }
        for _ in 0..2 {
            built_ins.pop();
        }
        writeln!(f, "fn main({built_ins}) {{")?;
        if self
            .enabled_builtins
            .contains(EnabledBuiltins::SubgroupsPerWorkgroup)
        {
            writeln!(
                f,
                "let subgroups_per_workgroup = BLOCKSIZE / subgroup_size;"
            )?;
        }
        if self
            .enabled_builtins
            .contains(EnabledBuiltins::SubgroupIndex)
        {
            writeln!(f, "let subgroup_id = workgroup_local_id / subgroup_size;")?;
        }
        if self
            .enabled_builtins
            .contains(EnabledBuiltins::SubgroupLocalIndex)
        {
            writeln!(
                f,
                "let subgroup_local_id = workgroup_local_id % subgroup_size;"
            )?;
        }
        writeln!(f, "{}", self.body)?;
        writeln!(f, "}}")?;

        Ok(())
    }
}
