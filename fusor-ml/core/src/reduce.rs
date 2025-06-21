use std::fmt::{Display, Write};

use crate::{
    ElementWiseFunctions,
    mir::{
        globals::KernelGlobalSpace,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
};
use crate::{
    Layout, Tensor,
    compute_graph::AnyComputeKey,
    mir::{function::Function, inputs::MirValue, kernel::GenericKernel},
    tensor::{DataType, DataTypeEnum, TensorData},
};

#[derive(Debug, Clone)]
pub(crate) struct ReduceOperation {
    pub(crate) value: AnyComputeKey,
    pre_element_wise: ElementWiseFunctions,
    pub(crate) function: ReduceFunction,
    post_element_wise: ElementWiseFunctions,
    pub(crate) axis: usize,
    pub(crate) rank: u32,
}

impl ReduceOperation {
    pub fn new(value: AnyComputeKey, function: ReduceFunction, axis: usize, rank: u32) -> Self {
        let datatype = function.datatype();
        Self {
            value,
            pre_element_wise: ElementWiseFunctions::empty(datatype),
            function,
            post_element_wise: ElementWiseFunctions::empty(datatype),
            axis,
            rank,
        }
    }

    pub fn set_post_element_wise(&mut self, kernel: ElementWiseFunctions) {
        self.post_element_wise = kernel;
    }

    pub fn set_pre_element_wise(&mut self, kernel: ElementWiseFunctions) {
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
            self.function.datatype(),
            self.function.operation.clone(),
            [
                ("a".to_string(), self.function.datatype().to_string()),
                ("b".to_string(), self.function.datatype().to_string()),
            ],
        )
    }

    pub fn reduce_datatype(&self) -> DataTypeEnum {
        self.pre_element_wise.out_datatype()
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
    }

    fn kernel(&self, workgroup_shape: &WorkgroupShape, blocksize: u32, kernel: &mut GenericKernel) {
        let dtype = self.function.datatype();
        let out_datatype = self.out_datatype();
        let output_rank = self.rank - 1;
        // Based on v7 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        // And the mlx implementation https://github.com/ml-explore/mlx/blob/b05bcfd27f5f1293401b74dce02e38c8fd7ef66a/mlx/backend/metal/kernels/arg_reduce.metal
        // We can't query the warp size in WGSL, but we can use subgroup operations
        // https://github.com/gpuweb/gpuweb/issues/4437 would unlock a better equivalent to warp synchronization
        // We also can't synchronize among workgroups without atomics. storageBarrier() is a barrier for
        // the storage memory only inside the workgroup.
        // This kernel just uses one workgroup per reduction unit like the MLX kernel
        let input_tensor = kernel.add_tensor_input(output_rank, false, self.reduce_datatype());
        let output_tensor = kernel.add_tensor_input(output_rank, true, out_datatype);
        let reduce_size = kernel.add_integer_input();
        let reduce_stride = kernel.add_integer_input();
        let local_data =
            kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());
        let reduce = self.add_function(kernel);
        let pre_element_wise = self.add_pre_element_wise_functions(kernel);
        let post_element_wise = self.add_post_element_wise_functions(kernel);
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
            "var workgroup_index_remainder = {};",
            workgroup_shape.linearized_workgroup_index(kernel)
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
            self.function.initial_value
        )
        .unwrap();

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        writeln!(
            &mut kernel_body,
            "let bucket_size = ({reduce_size} + {blocksize}u - 1) / {blocksize}u;"
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
            self.function.initial_value,
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

        kernel.push_body(&kernel_body);
    }
}

impl Operation for ReduceOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        let limits = device.wgpu_device().limits();
        constraints.add_constraint(
            0,
            Constraint::less_than(limits.max_compute_workgroup_size_x + 1),
        );
        constraints.add_constraint(
            0,
            Constraint::more_than_or_equals(limits.min_subgroup_size.max(32)),
        );
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        let workgroup_size = output_tensor.layout().shape().iter().product::<usize>() as u32;
        let workgroup_dispatch_size = [workgroup_size, 1, 1];
        workgroup_dispatch_size
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.value);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let dim = self.axis;
        let tensor = nodes.cached_results.get(&self.value).unwrap();
        let layout = tensor.layout();
        let shape = layout.shape();
        let new_tensor_shape = shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect::<Vec<_>>();
        let output_type = self.out_datatype();
        let output_tensor =
            TensorData::new_for_shape(tensor.device(), &new_tensor_shape, output_type);

        let trimmed_tensor_layout = Layout::from_parts(
            tensor.layout().offset(),
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
        vec![
            MirValue::Tensor(trimmed_tensor.clone()),
            MirValue::Tensor(output_tensor.clone()),
            MirValue::Integer(tensor.layout().shape()[dim] as u32),
            MirValue::Integer(tensor.layout().strides()[dim] as u32),
        ]
    }

    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let max_blocksize = workgroup_shape.x();
        self.kernel(workgroup_shape, max_blocksize, kernel);
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> MirValue {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!("reduce_{}", self.function.name().to_string())
    }
}

#[derive(Clone, Debug)]
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
