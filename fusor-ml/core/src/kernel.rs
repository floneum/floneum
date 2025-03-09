use enumset::{EnumSet, EnumSetType};
use std::fmt::{Debug, Write};
use std::{fmt::Display, sync::OnceLock};
use wgpu::{BindGroupLayout, CommandEncoder, PipelineCompilationOptions, util::DeviceExt};

use crate::{DataTypeEnum, Device, PerformanceQueries, TensorData};

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

pub(crate) struct GenericKernel {
    workgroup_size: [u32; 3],
    max_binding: u32,
    max_function_id: u32,
    max_global_id: u32,
    inputs: Vec<KernelInput>,
    functions: Vec<Function>,
    globals: Vec<KernelGlobal>,
    enabled_builtins: EnumSet<EnabledBuiltins>,
    kernel: OnceLock<wgpu::ShaderModule>,
    body: String,
}

impl GenericKernel {
    pub(crate) fn new() -> Self {
        Self {
            workgroup_size: [1, 1, 1],
            inputs: Default::default(),
            max_binding: 0,
            functions: Default::default(),
            max_function_id: 0,
            globals: Default::default(),
            max_global_id: 0,
            enabled_builtins: Default::default(),
            kernel: OnceLock::new(),
            body: String::new(),
        }
    }

    pub(crate) fn set_body(&mut self, body: String) {
        self.body = body;
    }

    pub(crate) fn set_workgroup_size(&mut self, workgroup_size: [u32; 3]) {
        assert!(
            workgroup_size.iter().product::<u32>() <= 256,
            "{workgroup_size:?} product must be <= 256"
        );
        self.workgroup_size = workgroup_size;
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
        let start_index = self.max_binding;
        self.max_binding += 2;

        let input = TensorInput {
            start_index,
            rank,
            mutable,
            datatype,
        };

        self.inputs.push(KernelInput {
            ty: KernelInputType::Tensor(input.clone()),
        });

        input
    }

    pub(crate) fn add_integer_input(&mut self) -> IntegerInput {
        let index = self.max_binding;
        self.max_binding += 1;

        let input = IntegerInput { index };

        self.inputs.push(KernelInput {
            ty: KernelInputType::Integer(input.clone()),
        });

        input
    }

    #[allow(dead_code)]
    pub(crate) fn add_float_input(&mut self) -> FloatInput {
        let index = self.max_binding;
        self.max_binding += 1;

        let input = FloatInput { index };

        self.inputs.push(KernelInput {
            ty: KernelInputType::Float(input.clone()),
        });

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
            KernelGlobalType::Array(ArrayType {
                size,
                datatype: array_type,
            }),
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
                KernelInputType::Tensor(tensor_input) => {
                    // Tensor weight
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: tensor_input.get_tensor_binding(),
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
                    // Tensor info
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: tensor_input.get_info_binding(),
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

    fn create_bind_group<'a>(
        &self,
        device: &crate::Device,
        bind_group_layout: &BindGroupLayout,
        tensors: impl IntoIterator<Item = impl Into<KernelInputValue>>,
    ) -> wgpu::BindGroup {
        let mut entries = Vec::new();
        let mut owned_entries = Vec::new();
        let tensors = tensors.into_iter().map(|x| x.into()).collect::<Vec<_>>();
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
        for (input, value) in self.inputs.iter().zip(tensors.iter()) {
            match (&input.ty, value) {
                (KernelInputType::Tensor(tensor_input), KernelInputValue::Tensor(tensor)) => {
                    // Tensor weight
                    entries.push(wgpu::BindGroupEntry {
                        binding: tensor_input.get_tensor_binding(),
                        resource: tensor.buffer().as_entire_binding(),
                    });
                    // Tensor info
                    owned_entries.push((
                        tensor_input.get_info_binding(),
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
                _ => todo!(),
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

    pub(crate) fn run_with_query<'a>(
        &self,
        device: &Device,
        tensors: impl IntoIterator<Item = impl Into<KernelInputValue>>,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
        workgroup_dispatch_size: [u32; 3],
    ) {
        let bind_group_layout = self.bind_group_layout(device);
        let bind_group = self.create_bind_group(device, &bind_group_layout, tensors);
        let pipeline = self.compute_pipeline(device, &bind_group_layout);

        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: query.map(|query| query.compute_timestamp_writes()),
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = workgroup_dispatch_size;
            cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z);
        }

        if let Some(query) = query {
            query.resolve(command_encoder);
        }
    }

    fn kernel(&self, f: &mut String) -> std::fmt::Result {
        writeln!(f, "enable f16;")?;

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

pub(crate) enum KernelInputValue {
    Tensor(TensorData),
    Integer(u32),
    Float(f32),
}

impl From<TensorData> for KernelInputValue {
    fn from(value: TensorData) -> Self {
        Self::Tensor(value)
    }
}

impl From<u32> for KernelInputValue {
    fn from(value: u32) -> Self {
        Self::Integer(value)
    }
}

impl From<f32> for KernelInputValue {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Function {
    id: u32,
    ty: String,
    body: String,
    inputs: Vec<(String, String)>,
}

impl Function {
    fn new(id: u32, ty: String, body: String, inputs: Vec<(String, String)>) -> Self {
        Self {
            id,
            ty,
            body,
            inputs,
        }
    }

    fn function_definition(&self) -> String {
        let name = self.function_name();
        let inputs = &self.inputs;
        let mut inputs_string = String::new();
        for (name, ty) in inputs {
            inputs_string.push_str(&format!("{name}: {ty}, "));
        }
        for _ in 0..2 {
            inputs_string.pop();
        }
        let body = &self.body;
        let ty = &self.ty;
        format!("fn {name}({inputs_string}) -> {ty} {{ {body} return output; }}")
    }

    fn function_name(&self) -> String {
        format!("f_{}", self.id)
    }

    pub(crate) fn call(&self, inputs: Vec<String>) -> String {
        format!("{}({})", self.function_name(), inputs.join(", "))
    }

    #[allow(dead_code)]
    pub(crate) fn call_inlined(&self, inputs: Vec<String>) -> String {
        let mut output = String::new();
        output.push_str("{\n");
        for (i_value, (i_name, ty)) in inputs.iter().zip(&self.inputs) {
            output.push_str(&format!("let {i_name}: {ty} = {i_value};\n"));
        }
        output.push_str("}\n");
        output
    }
}

#[derive(Clone)]
pub enum KernelGlobalSpace {
    Workgroup,
}

impl Display for KernelGlobalSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelGlobalSpace::Workgroup => write!(f, "workgroup"),
        }
    }
}

#[derive(Clone)]
pub struct KernelGlobal {
    id: u32,
    space: KernelGlobalSpace,
    ty: KernelGlobalType,
}

impl KernelGlobal {
    pub fn new(id: u32, space: KernelGlobalSpace, ty: KernelGlobalType) -> Self {
        Self { id, space, ty }
    }

    pub fn global_definition(&self) -> String {
        match &self.ty {
            KernelGlobalType::Array(array) => {
                let dtype = &array.datatype;
                let size = &array.size;
                let space = &self.space;
                format!("var<{space}> {self}: array<{dtype}, {size}>;\n")
            }
        }
    }
}

impl Display for KernelGlobal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g_{}", self.id)
    }
}

#[derive(Clone)]
pub enum KernelGlobalType {
    Array(ArrayType),
}

#[derive(Clone)]
pub struct ArrayType {
    size: String,
    datatype: DataTypeEnum,
}

struct KernelInput {
    ty: KernelInputType,
}

impl Display for KernelInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.ty {
            KernelInputType::Tensor(tensor) => {
                let start_index = tensor.start_index;
                let datatype = tensor.datatype;
                write!(f, "@group(0) @binding({start_index}) ")?;

                if tensor.mutable {
                    write!(f, "var<storage, read_write> ")?;
                } else {
                    write!(f, "var<storage, read> ")?;
                }

                writeln!(f, "i_{start_index}: array<{datatype}>;")?;

                writeln!(f, "struct Tensor{start_index}Info {{")?;
                writeln!(f, "    offset: u32,")?;
                for i in 0..tensor.rank {
                    writeln!(f, "    stride_{}: u32,", i)?;
                    writeln!(f, "    shape_{}: u32,", i)?;
                }
                writeln!(f, "}};")?;

                let info_index = tensor.get_info_binding();
                writeln!(
                    f,
                    "@group(0) @binding({info_index}) var<uniform> i_{info_index}: Tensor{start_index}Info;"
                )?;
            }
            KernelInputType::Integer(integer) => {
                let index = integer.index;
                write!(
                    f,
                    "@group(0) @binding({index}) var<uniform> i_{index}: u32;"
                )?
            }
            KernelInputType::Float(float) => write!(f, "var<uniform> i_{}: f32;", float.index)?,
        }

        Ok(())
    }
}

enum KernelInputType {
    Tensor(TensorInput),
    Integer(IntegerInput),
    Float(FloatInput),
}

#[derive(Clone)]
pub(crate) struct IntegerInput {
    index: u32,
}

impl Display for IntegerInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.index)
    }
}

#[derive(Clone)]
pub(crate) struct FloatInput {
    index: u32,
}

impl Display for FloatInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.index)
    }
}

#[derive(Clone)]
pub(crate) struct TensorInput {
    start_index: u32,
    rank: u32,
    mutable: bool,
    datatype: DataTypeEnum,
}

impl TensorInput {
    fn get_tensor_binding(&self) -> u32 {
        self.start_index
    }

    fn get_info_binding(&self) -> u32 {
        self.start_index + 1
    }

    fn info_binding(&self) -> String {
        format!("i_{}", self.get_info_binding())
    }

    pub(crate) fn offset_binding(&self) -> String {
        format!("{}.offset", self.info_binding())
    }

    pub(crate) fn stride_binding(&self, rank: u32) -> String {
        format!("{}.stride_{}", self.info_binding(), rank)
    }

    pub(crate) fn shape_binding(&self, rank: u32) -> String {
        format!("{}.shape_{}", self.info_binding(), rank)
    }

    pub(crate) fn check_bounds(
        &self,
        write: &mut String,
        indexes: impl IntoIterator<Item = String>,
        in_bounds: impl FnOnce(&mut String),
    ) {
        write!(write, "if true ").unwrap();
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = self.shape_binding(i as u32);
            write!(write, "&& {index} < {stride} ").unwrap();
        }
        write!(write, "{{").unwrap();
        in_bounds(write);
        write!(write, "}}").unwrap();
    }

    pub(crate) fn check_bounds_contiguous(
        &self,
        write: &mut String,
        contiguous_index: String,
        in_bounds: impl FnOnce(&mut String),
    ) {
        write!(write, "if {contiguous_index} < 1 ").unwrap();
        for i in 0..self.rank {
            let stride = self.shape_binding(i);
            write!(write, "* {stride} ").unwrap();
        }
        write!(write, "{{").unwrap();
        in_bounds(write);
        write!(write, "}}").unwrap();
    }

    pub(crate) fn strided_index(
        &self,
        write: &mut String,
        indexes: impl IntoIterator<Item = String>,
    ) {
        let offset = self.offset_binding();
        write!(write, "{offset} + ").unwrap();
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = self.stride_binding(i as u32);
            write!(write, "{index}*{stride} + ").unwrap();
        }
        for _ in 0..3 {
            write.pop();
        }
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }
}

impl Display for TensorInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.start_index)
    }
}
