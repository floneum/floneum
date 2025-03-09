use std::{
    fmt::{Display, Write},
    sync::OnceLock,
};

use wgpu::CommandEncoder;

use crate::{
    Layout, Tensor, UntypedElementWiseKernel,
    compute_graph::AnyComputeKey,
    kernel::{Function, GenericKernel, KernelGlobalSpace, KernelInputValue},
    query::PerformanceQueries,
    tensor::{DataType, DataTypeEnum, TensorData, padded_tensor_size},
};

#[derive(Clone)]
pub(crate) struct ReduceOperation {
    pub(crate) value: AnyComputeKey,
    pub(crate) function: ReduceFunction,
    pub(crate) axis: usize,
}

impl ReduceOperation {
    pub fn new(value: AnyComputeKey, function: ReduceFunction, axis: usize) -> Self {
        Self {
            value,
            function,
            axis,
        }
    }
}

pub(crate) struct UntypedReduceKernel {
    pre_element_wise: UntypedElementWiseKernel,
    reduce: ReduceFunction,
    post_element_wise: UntypedElementWiseKernel,
    kernel: OnceLock<GenericKernel>,
    datatype: DataTypeEnum,
}

impl UntypedReduceKernel {
    pub fn new(reduce: ReduceFunction, datatype: DataTypeEnum) -> Self {
        Self {
            pre_element_wise: UntypedElementWiseKernel::empty(datatype),
            reduce,
            post_element_wise: UntypedElementWiseKernel::empty(datatype),
            kernel: OnceLock::new(),
            datatype,
        }
    }

    pub fn set_post_element_wise(&mut self, kernel: UntypedElementWiseKernel) {
        self.post_element_wise = kernel;
    }

    pub fn set_pre_element_wise(&mut self, kernel: UntypedElementWiseKernel) {
        self.pre_element_wise = kernel;
    }

    pub fn add_pre_element_wise_functions(&self, kernel: &mut GenericKernel) -> Vec<Function> {
        self.pre_element_wise.add_functions(kernel)
    }

    pub fn add_post_element_wise_functions(&self, kernel: &mut GenericKernel) -> Vec<Function> {
        self.post_element_wise.add_functions(kernel)
    }

    pub fn add_function(&self, kernel: &mut GenericKernel) -> Function {
        kernel.add_function(
            self.reduce.datatype(),
            self.reduce.operation.clone(),
            [
                ("a".to_string(), self.reduce.datatype().to_string()),
                ("b".to_string(), self.reduce.datatype().to_string()),
            ],
        )
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
    }

    fn tiled_map(&self, blocksize: u32, input_rank: u32) -> GenericKernel {
        let dtype = self.reduce.datatype();
        let out_datatype = self.out_datatype();
        let mut kernel = GenericKernel::new();
        let output_rank = input_rank - 1;
        // Based on v7 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        // And the mlx implementation https://github.com/ml-explore/mlx/blob/b05bcfd27f5f1293401b74dce02e38c8fd7ef66a/mlx/backend/metal/kernels/arg_reduce.metal
        // We can't query the warp size in WGSL, but we can use subgroup operations
        // https://github.com/gpuweb/gpuweb/issues/4437 would unlock a better equivalent to warp synchronization
        // We also can't synchronize among workgroups without atomics. storageBarrier() is a barrier for
        // the storage memory only inside the workgroup.
        // This kernel just uses one workgroup per reduction unit like the MLX kernel
        let input_tensor = kernel.add_tensor_input(output_rank, false, self.datatype);
        let output_tensor = kernel.add_tensor_input(output_rank, true, out_datatype);
        let reduce_size = kernel.add_integer_input();
        let reduce_stride = kernel.add_integer_input();
        let local_data =
            kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());
        let reduce = self.add_function(&mut kernel);
        let pre_element_wise = self.add_pre_element_wise_functions(&mut kernel);
        let post_element_wise = self.add_post_element_wise_functions(&mut kernel);
        let workgroup_index = kernel.workgroup_index();
        let workgroup_local_index = kernel.workgroup_local_index();
        let subgroup_id = kernel.subgroup_index();
        let subgroup_local_id = kernel.subgroup_local_index();
        let subgroups_per_workgroup = kernel.subgroups_per_workgroup();
        let subgroup_size = kernel.subgroup_size();

        let mut kernel_body = String::new();
        // Each workgroup group works on a single column in the input tensor. This code calculates the
        // start offset of the input and output tensors for each thread group.
        writeln!(
            &mut kernel_body,
            "var workgroup_index_remainder = {workgroup_index}.x;"
        )
        .unwrap();
        for i in (0..output_rank).rev() {
            let out_shape_i = output_tensor.shape_binding(i);
            writeln!(
                &mut kernel_body,
                "let index_{i} = workgroup_index_remainder % {out_shape_i};",
            )
            .unwrap();
            writeln!(
                &mut kernel_body,
                "workgroup_index_remainder /= {out_shape_i};",
            )
            .unwrap();
        }
        writeln!(&mut kernel_body, "var in_start_offset = ",).unwrap();
        input_tensor.strided_index(&mut kernel_body, (0..).map(|i| format!("index_{i}")));
        writeln!(&mut kernel_body, ";").unwrap();
        writeln!(&mut kernel_body, "var out_start_offset = ",).unwrap();
        output_tensor.strided_index(&mut kernel_body, (0..).map(|i| format!("index_{i}")));
        writeln!(&mut kernel_body, ";").unwrap();
        writeln!(&mut kernel_body).unwrap();

        writeln!(
            &mut kernel_body,
            "var merged = {dtype}({});",
            self.reduce.initial_value
        )
        .unwrap();

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        writeln!(
            &mut kernel_body,
            "let bucket_size = {reduce_size} / {blocksize}u + u32(({reduce_size} % {blocksize}u) != 0u);"
        )
        .unwrap();
        // Then loop over this thread's portion of the column and merge the values
        writeln!(
            &mut kernel_body,
            "for (var index = 0u; index < bucket_size; index += 1u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let axis_index = {workgroup_local_index} * bucket_size + index;"
        )
        .unwrap();
        writeln!(&mut kernel_body, "if axis_index < {reduce_size} {{").unwrap();
        writeln!(
            &mut kernel_body,
            "let in_index = in_start_offset + axis_index * {reduce_stride};"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let data = {};",
            pre_element_wise
                .iter()
                .fold(format!("{input_tensor}[in_index]"), |acc, f| f
                    .call(vec![acc]))
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "merged = {};",
            reduce.call(vec!["data".to_string(), "merged".to_string()])
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body).unwrap();

        // Next merge within each subgroup with shuffle down
        writeln!(
            &mut kernel_body,
            "for (var offset = {subgroup_size} / 2u; offset > 0u; offset /= 2u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let neighbor = subgroupShuffleDown(merged, offset);"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "merged = {};",
            reduce.call(vec!["neighbor".to_string(), "merged".to_string()])
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body).unwrap();

        // Write the output to the workgroup memory if this is the first thread in the subgroup
        writeln!(&mut kernel_body, "if {subgroup_local_id} == 0u {{").unwrap();
        writeln!(&mut kernel_body, "{local_data}[{subgroup_id}] = merged;").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

        // Wait until all threads have written to the workgroup shared memory
        writeln!(&mut kernel_body, "workgroupBarrier();").unwrap();

        // Then if this is the first subgroup, do one final shuffle down reduction
        writeln!(&mut kernel_body, "if {subgroup_id} == 0u {{").unwrap();
        // Copy over the best value from each subgroup from the workgroup shared memory to the merged variable
        writeln!(
            &mut kernel_body,
            "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "merged = {local_data}[{subgroup_local_id}];"
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body, "else {{").unwrap();
        writeln!(
            &mut kernel_body,
            "merged = {dtype}({});\n",
            self.reduce.initial_value,
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(
            &mut kernel_body,
            "for (var offset = {subgroup_size} / 2u; offset > 0u; offset /= 2u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let neighbor = subgroupShuffleDown(merged, offset);"
        )
        .unwrap();
        writeln!(&mut kernel_body, "var data = neighbor;").unwrap();
        writeln!(
            &mut kernel_body,
            "merged = {};",
            reduce.call(vec!["neighbor".to_string(), "merged".to_string()])
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

        // Write the output to the output tensor if this is the first thread in the workgroup
        writeln!(&mut kernel_body, "if {workgroup_local_index} == 0u {{").unwrap();
        writeln!(
            &mut kernel_body,
            "let data = {};",
            post_element_wise
                .iter()
                .fold("merged".to_string(), |acc, f| f.call(vec![acc]))
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "{output_tensor}[out_start_offset] = data;"
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

        kernel.set_body(kernel_body);
        kernel.set_workgroup_size([blocksize, 1, 1]);

        kernel
    }

    pub fn run_with_query(
        &self,
        tensor: &TensorData,
        dim: usize,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let shape = tensor.layout().shape();
        let new_tensor_shape = shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect::<Vec<_>>();
        let output_type = self.out_datatype();
        let output_buf = tensor
            .device()
            .wgpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: padded_tensor_size(
                    (new_tensor_shape.iter().product::<usize>() * output_type.element_size())
                        as u64,
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        let output_tensor = TensorData::new_from_buffer(
            tensor.device(),
            output_buf,
            &new_tensor_shape,
            output_type,
        );

        self.run_with_query_and_out_tensor(tensor, dim, query, &output_tensor, command_encoder);

        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        tensor: &TensorData,
        dim: usize,
        query: Option<&PerformanceQueries>,
        output_tensor: &TensorData,
        command_encoder: &mut CommandEncoder,
    ) {
        // assert_eq!(
        //     *output_tensor.layout().shape(),
        //     [tensor
        //         .layout()
        //         .shape()
        //         .iter()
        //         .enumerate()
        //         .filter_map(|(i, x)| { (i != dim).then_some(*x as u32) })
        //         .product::<u32>() as usize]
        // );

        let limits = tensor.device().wgpu_device().limits();
        let max_blocksize = (tensor.layout().shape()[dim] as u32)
            .min(limits.max_compute_workgroup_size_x)
            .max(limits.min_subgroup_size)
            .max(32);
        let kernel = self
            .kernel
            .get_or_init(|| self.tiled_map(max_blocksize, tensor.layout().rank() as u32));

        let workgroup_size = output_tensor.layout().shape().iter().product::<usize>() as u32;
        let workgroup_dispatch_size = [workgroup_size, 1, 1];
        let trimmed_tensor_layout = Layout::from_parts(
            0,
            tensor
                .layout()
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != dim).then_some(*x))
                .collect(),
            tensor
                .layout()
                .strides()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != dim).then_some(*x))
                .collect(),
        );
        let trimmed_tensor = TensorData::new_from_parts(
            tensor.device(),
            tensor.buffer().clone(),
            trimmed_tensor_layout,
            tensor.datatype(),
        );
        kernel.run_with_query(
            tensor.device(),
            [
                KernelInputValue::Tensor(trimmed_tensor.clone()),
                KernelInputValue::Tensor(output_tensor.clone()),
                KernelInputValue::Integer(tensor.layout().shape()[dim] as u32),
                KernelInputValue::Integer(tensor.layout().strides()[dim] as u32),
            ],
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}

#[derive(Clone)]
pub struct ReduceFunction {
    name: Option<String>,
    operation: String,
    initial_value: String,
    datatype: DataTypeEnum,
}

impl ReduceFunction {
    fn new(operation: impl Display, initial_value: impl Display, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            operation: operation.to_string(),
            initial_value: initial_value.to_string(),
            datatype,
        }
    }

    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("reduce")
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

macro_rules! impl_reduce {
    ($R:expr, $T:ident, $f_untyped:ident, $f:ident, $($arg:ident: $arg_type:ty),*) => {
        impl<D: DataType> $T for Tensor<$R, D> {
            type Output = Tensor<{ $R - 1 }, D>;

            fn $f(&self $(, $arg: $arg_type)*) -> Self::Output {
                $f_untyped(self $(, $arg)*)
            }
        }
    };
}

pub trait Sum {
    type Output;

    fn sum(&self, dim: usize) -> Self::Output;
}

fn unchecked_sum<const R1: usize, const R2: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    dim: usize,
) -> Tensor<R2, D> {
    tensor.reduce(
        ReduceFunction::new("let output = a + b;".to_string(), "0.0", D::WGSL_TYPE)
            .with_name("sum"),
        dim,
    )
}

impl_reduce!(1, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(2, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(3, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(4, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(5, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(6, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(7, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(8, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(9, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(10, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(11, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(12, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(13, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(14, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(15, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(16, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(17, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(18, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(19, Sum, unchecked_sum, sum, dim: usize);
impl_reduce!(20, Sum, unchecked_sum, sum, dim: usize);

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 9.);
    assert_eq!(output[[1]], 12.);

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 3.);
    assert_eq!(output[[1]], 7.);
    assert_eq!(output[[2]], 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum_f16() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], half::f16::from_f32(9.));
    assert_eq!(output[[1]], half::f16::from_f32(12.));

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], half::f16::from_f32(3.));
    assert_eq!(output[[1]], half::f16::from_f32(7.));
    assert_eq!(output[[2]], half::f16::from_f32(11.));
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sliced_sum() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.slice([0..3, 0..1]);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 9.);

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 3.);
    assert_eq!(output[[2]], 5.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_add_then_sum_fused() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = (tensor.clone() + 1.).sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 3. + 9.);
    assert_eq!(output[[1]], 3. + 12.);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = (tensor + 1.).sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 2. + 3.);
    assert_eq!(output[[1]], 2. + 7.);
    assert_eq!(output[[2]], 2. + 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_sum_then_add_fused() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0) + 1.;

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1. + 9.);
    assert_eq!(output[[1]], 1. + 12.);

    let output = tensor.sum(1) + 1.;

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1. + 3.);
    assert_eq!(output[[1]], 1. + 7.);
    assert_eq!(output[[2]], 1. + 11.);
}

fn unchecked_max<const R1: usize, const R2: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    dim: usize,
) -> Tensor<R2, D> {
    tensor.reduce(
        ReduceFunction::new(
            "let output = max(a, b);".to_string(),
            "-3.40282e+38",
            D::WGSL_TYPE,
        )
        .with_name("max"),
        dim,
    )
}

pub trait Max {
    type Output;

    fn max(&self, dim: usize) -> Self::Output;
}

impl_reduce!(1, Max, unchecked_max, max, dim: usize);
impl_reduce!(2, Max, unchecked_max, max, dim: usize);
impl_reduce!(3, Max, unchecked_max, max, dim: usize);
impl_reduce!(4, Max, unchecked_max, max, dim: usize);
impl_reduce!(5, Max, unchecked_max, max, dim: usize);
impl_reduce!(6, Max, unchecked_max, max, dim: usize);
impl_reduce!(7, Max, unchecked_max, max, dim: usize);
impl_reduce!(8, Max, unchecked_max, max, dim: usize);
impl_reduce!(9, Max, unchecked_max, max, dim: usize);
impl_reduce!(10, Max, unchecked_max, max, dim: usize);
impl_reduce!(11, Max, unchecked_max, max, dim: usize);
impl_reduce!(12, Max, unchecked_max, max, dim: usize);
impl_reduce!(13, Max, unchecked_max, max, dim: usize);
impl_reduce!(14, Max, unchecked_max, max, dim: usize);
impl_reduce!(15, Max, unchecked_max, max, dim: usize);
impl_reduce!(16, Max, unchecked_max, max, dim: usize);
impl_reduce!(17, Max, unchecked_max, max, dim: usize);
impl_reduce!(18, Max, unchecked_max, max, dim: usize);
impl_reduce!(19, Max, unchecked_max, max, dim: usize);
impl_reduce!(20, Max, unchecked_max, max, dim: usize);

#[cfg(test)]
#[tokio::test]
async fn test_reduce_max() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.max(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 5.);
    assert_eq!(output[[1]], 6.);

    let output = tensor.max(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 4.);
    assert_eq!(output[[2]], 6.);
}

fn unchecked_min<const R1: usize, const R2: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    dim: usize,
) -> Tensor<R2, D> {
    tensor.reduce(
        ReduceFunction::new(
            "let output = min(a, b);".to_string(),
            "3.40282e+38",
            D::WGSL_TYPE,
        )
        .with_name("min"),
        dim,
    )
}

pub trait Min {
    type Output;

    fn min(&self, dim: usize) -> Self::Output;
}

impl_reduce!(1, Min, unchecked_min, min, dim: usize);
impl_reduce!(2, Min, unchecked_min, min, dim: usize);
impl_reduce!(3, Min, unchecked_min, min, dim: usize);
impl_reduce!(4, Min, unchecked_min, min, dim: usize);
impl_reduce!(5, Min, unchecked_min, min, dim: usize);
impl_reduce!(6, Min, unchecked_min, min, dim: usize);
impl_reduce!(7, Min, unchecked_min, min, dim: usize);
impl_reduce!(8, Min, unchecked_min, min, dim: usize);
impl_reduce!(9, Min, unchecked_min, min, dim: usize);
impl_reduce!(10, Min, unchecked_min, min, dim: usize);
impl_reduce!(11, Min, unchecked_min, min, dim: usize);
impl_reduce!(12, Min, unchecked_min, min, dim: usize);
impl_reduce!(13, Min, unchecked_min, min, dim: usize);
impl_reduce!(14, Min, unchecked_min, min, dim: usize);
impl_reduce!(15, Min, unchecked_min, min, dim: usize);
impl_reduce!(16, Min, unchecked_min, min, dim: usize);
impl_reduce!(17, Min, unchecked_min, min, dim: usize);
impl_reduce!(18, Min, unchecked_min, min, dim: usize);
impl_reduce!(19, Min, unchecked_min, min, dim: usize);
impl_reduce!(20, Min, unchecked_min, min, dim: usize);

#[cfg(test)]
#[tokio::test]
async fn test_reduce_min() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.min(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 2.);

    let output = tensor.min(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 3.);
    assert_eq!(output[[2]], 5.);
}

fn unchecked_product<const R1: usize, const R2: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    dim: usize,
) -> Tensor<R2, D> {
    tensor.reduce(
        ReduceFunction::new("let output = a * b;".to_string(), "1.0", D::WGSL_TYPE)
            .with_name("product"),
        dim,
    )
}

pub trait Product {
    type Output;

    fn product(&self, dim: usize) -> Self::Output;
}

impl_reduce!(1, Product, unchecked_product, product, dim: usize);
impl_reduce!(2, Product, unchecked_product, product, dim: usize);
impl_reduce!(3, Product, unchecked_product, product, dim: usize);
impl_reduce!(4, Product, unchecked_product, product, dim: usize);
impl_reduce!(5, Product, unchecked_product, product, dim: usize);
impl_reduce!(6, Product, unchecked_product, product, dim: usize);
impl_reduce!(7, Product, unchecked_product, product, dim: usize);
impl_reduce!(8, Product, unchecked_product, product, dim: usize);
impl_reduce!(9, Product, unchecked_product, product, dim: usize);
impl_reduce!(10, Product, unchecked_product, product, dim: usize);
impl_reduce!(11, Product, unchecked_product, product, dim: usize);
impl_reduce!(12, Product, unchecked_product, product, dim: usize);
impl_reduce!(13, Product, unchecked_product, product, dim: usize);
impl_reduce!(14, Product, unchecked_product, product, dim: usize);
impl_reduce!(15, Product, unchecked_product, product, dim: usize);
impl_reduce!(16, Product, unchecked_product, product, dim: usize);
impl_reduce!(17, Product, unchecked_product, product, dim: usize);
impl_reduce!(18, Product, unchecked_product, product, dim: usize);
impl_reduce!(19, Product, unchecked_product, product, dim: usize);
impl_reduce!(20, Product, unchecked_product, product, dim: usize);

#[cfg(test)]
#[tokio::test]
async fn test_reduce_product() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.product(0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 15.);
    assert_eq!(output[[1]], 48.);

    let output = tensor.product(1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 12.);
    assert_eq!(output[[2]], 30.);
}
