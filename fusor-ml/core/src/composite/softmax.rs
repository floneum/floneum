use std::{
    fmt::{Display, Write},
    sync::Arc,
};

use crate::{
    DataType, DataTypeEnum, Layout, Max, Sum, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{
        globals::KernelGlobalSpace,
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
};

impl<const R: usize, const R2: usize, D: DataType> Tensor<R, D>
where
    Tensor<R, D>: Max<Output = Tensor<R2, D>>,
    Tensor<R, D>: Sum<Output = Tensor<R2, D>>,
{
    pub fn softmax_slow(&self, dim: usize) -> Self {
        let size = *self.shape();
        let max = self.max(dim);
        let normalized = self - &max.broadcast_as(size);
        let exp = normalized.exp();
        let sum = exp.sum(dim);
        exp / sum.broadcast_as(size)
    }

    pub fn softmax_slow_last_dim(&self) -> Self {
        self.softmax_slow(self.rank() - 1)
    }

    pub fn softmax(&self, axis: usize) -> Self {
        let operation = SoftmaxOperation::new(self.key(), self.datatype(), axis, self.shape());
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }

    pub fn softmax_last_dim(&self) -> Self {
        self.softmax(self.rank() - 1)
    }
}

fn online_update(f: &mut String, m: impl Display, d: impl Display, x: impl Display) {
    writeln!(f, "{{").unwrap();
    writeln!(f, "let original_m = {m};").unwrap();
    writeln!(f, "{m} = max({m}, {x});").unwrap();
    writeln!(f, "{d} = {d} * exp(original_m - {m}) + exp({x} - {m});").unwrap();
    writeln!(f, "}}").unwrap();
}

fn combine(
    f: &mut String,
    m: impl Display,
    d: impl Display,
    m_peer: impl Display,
    d_peer: impl Display,
) {
    writeln!(f, "let original_m = {m};").unwrap();
    writeln!(f, "{m} = max(original_m, {m_peer});").unwrap();
    writeln!(
        f,
        "{d} = {d} * exp(original_m - {m}) + {d_peer} * exp({m_peer} - {m});"
    )
    .unwrap();
}

#[derive(Debug, Clone)]
struct SoftmaxOperation {
    pub(crate) value: AnyComputeKey,
    pub(crate) axis: usize,
    pub(crate) shape: Box<[usize]>,
    pub(crate) datatype: DataTypeEnum,
}

impl SoftmaxOperation {
    pub fn new(value: AnyComputeKey, datatype: DataTypeEnum, axis: usize, shape: &[usize]) -> Self {
        Self {
            value,
            axis,
            shape: shape.into(),
            datatype,
        }
    }

    pub fn reduce_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    fn rank(&self) -> u32 {
        self.shape.len() as _
    }

    fn kernel(
        &self,
        workgroup_shape: &WorkgroupShape,
        blocksize: u32,
        kernel: &mut GenericKernel,
        device: &crate::Device,
    ) {
        let dtype = self.datatype;
        let out_datatype = self.out_datatype();
        let output_rank = self.rank() - 1;
        let large_reduction = self.shape[self.axis] > 256;

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
        let local_m_data =
            kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());
        let local_d_data =
            kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());

        let global_m_final = kernel.add_global_value(KernelGlobalSpace::Workgroup, dtype);
        let global_d_final = kernel.add_global_value(KernelGlobalSpace::Workgroup, dtype);

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

        writeln!(&mut kernel_body, "var m_lane = {dtype}(-3.40282e+38);").unwrap();
        writeln!(&mut kernel_body, "var d_lane = {dtype}(0.0);").unwrap();

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        writeln!(
            &mut kernel_body,
            "let bucket_size = ({reduce_size} + {blocksize}u - 1) / {blocksize}u;"
        )
        .unwrap();
        // Then loop over this thread's portion of the column and merge the values
        // First load in groups of 4
        writeln!(
            &mut kernel_body,
            "let base_axis_index = {workgroup_local_index} * bucket_size;"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let end_axis_index = min({workgroup_local_index} * bucket_size + bucket_size, {reduce_size});"
        )
        .unwrap();
        writeln!(&mut kernel_body, "var index = base_axis_index;").unwrap();

        // If this is a large reduction, process elements in groups of 4
        if large_reduction {
            writeln!(&mut kernel_body, "while (index + 4u <= end_axis_index) {{").unwrap();
            // Load the chunk of 4 elements at once
            write!(&mut kernel_body, "let data = vec4<{dtype}>(").unwrap();
            for i in 0..4 {
                if i > 0 {
                    write!(&mut kernel_body, ", ").unwrap();
                }
                write!(
                    &mut kernel_body,
                    "{input_tensor}[in_start_offset + (index + {i}u) * {reduce_stride}]"
                )
                .unwrap();
            }
            writeln!(&mut kernel_body, ");").unwrap();

            // Apply reduction to the 4 elements
            let components = ["data.x", "data.y", "data.z", "data.w"];
            for component in components {
                online_update(&mut kernel_body, "m_lane", "d_lane", component);
            }

            writeln!(&mut kernel_body, "index += 4u;").unwrap();
            writeln!(&mut kernel_body, "}}").unwrap();
            writeln!(&mut kernel_body).unwrap();
        }

        // Merge the < 4 remaining elements if the bucket size is not a multiple of 4
        writeln!(&mut kernel_body, "while (index < end_axis_index) {{").unwrap();
        // Load a single element
        writeln!(
            &mut kernel_body,
            "let data = {input_tensor}[in_start_offset + index * {reduce_stride}];"
        )
        .unwrap();
        // Apply the online update function to merge the single element
        online_update(&mut kernel_body, "m_lane", "d_lane", "data");
        writeln!(&mut kernel_body, "index += 1u;").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body).unwrap();

        let limits = device.limits();
        let max_subgroup_size = limits.max_subgroup_size;

        // Optimized subgroup reduction with unrolled shuffle operations
        let mut offset = max_subgroup_size;
        while offset > 1 {
            writeln!(&mut kernel_body, "if {subgroup_size} >= {offset}u {{").unwrap();
            offset /= 2;
            writeln!(
                &mut kernel_body,
                "let m_peer = subgroupShuffleDown(m_lane, {offset}u);"
            )
            .unwrap();
            writeln!(
                &mut kernel_body,
                "let d_peer = subgroupShuffleDown(d_lane, {offset}u);"
            )
            .unwrap();
            combine(&mut kernel_body, "m_lane", "d_lane", "m_peer", "d_peer");
            writeln!(&mut kernel_body, "}}").unwrap();
        }
        writeln!(&mut kernel_body).unwrap();

        // Write the output to the workgroup memory if this is the first thread in the subgroup
        writeln!(&mut kernel_body, "if {subgroup_local_id} == 0u {{").unwrap();
        writeln!(&mut kernel_body, "{local_m_data}[{subgroup_id}] = m_lane;").unwrap();
        writeln!(&mut kernel_body, "{local_d_data}[{subgroup_id}] = d_lane;").unwrap();
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
            "m_lane = {local_m_data}[{subgroup_local_id}];"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "d_lane = {local_d_data}[{subgroup_local_id}];"
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body, "else {{").unwrap();
        writeln!(&mut kernel_body, "m_lane = {dtype}(-3.40282e+38);").unwrap();
        writeln!(&mut kernel_body, "d_lane = {dtype}(0.0);").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

        // Optimized final subgroup reduction with unrolled operations
        let mut offset = max_subgroup_size;
        while offset > 1 {
            writeln!(&mut kernel_body, "if {subgroup_size} >= {offset}u {{").unwrap();
            offset /= 2;
            writeln!(
                &mut kernel_body,
                "let m_peer = subgroupShuffleDown(m_lane, {offset}u);"
            )
            .unwrap();
            writeln!(
                &mut kernel_body,
                "let d_peer = subgroupShuffleDown(d_lane, {offset}u);"
            )
            .unwrap();
            combine(&mut kernel_body, "m_lane", "d_lane", "m_peer", "d_peer");
            writeln!(&mut kernel_body, "}}").unwrap();
        }

        // Write the output to the output tensor if this is the first thread in the workgroup
        writeln!(&mut kernel_body, "if {workgroup_local_index} == 0u {{").unwrap();
        writeln!(&mut kernel_body, "{global_m_final} = m_lane;").unwrap();
        writeln!(&mut kernel_body, "{global_d_final} = d_lane;").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

        writeln!(&mut kernel_body, "workgroupBarrier();").unwrap();

        // Finally, write the normalized output to the output tensor
        writeln!(&mut kernel_body, "let m_all = {global_m_final};").unwrap();
        writeln!(&mut kernel_body, "let d_all = {global_d_final};").unwrap();
        writeln!(&mut kernel_body, "var out_index = base_axis_index;").unwrap();

        // If this is a large reduction, process elements in groups of 4
        if large_reduction {
            writeln!(
                &mut kernel_body,
                "while (out_index + 4u <= end_axis_index) {{"
            )
            .unwrap();
            // Load the chunk of 4 elements at once
            write!(&mut kernel_body, "let data = vec4<{dtype}>(").unwrap();
            for i in 0..4 {
                if i > 0 {
                    write!(&mut kernel_body, ", ").unwrap();
                }
                write!(
                    &mut kernel_body,
                    "{input_tensor}[in_start_offset + (out_index + {i}u) * {reduce_stride}]"
                )
                .unwrap();
            }
            writeln!(&mut kernel_body, ");").unwrap();

            // Apply the softmax function to each component
            let components = ["data.x", "data.y", "data.z", "data.w"];
            for (i, component) in components.iter().enumerate() {
                writeln!(
                &mut kernel_body,
                "{output_tensor}[out_start_offset + (out_index + {i}u) * {reduce_stride}] = exp({component} - m_all) / d_all;"
            )
            .unwrap();
            }
            writeln!(&mut kernel_body, "out_index += 4u;").unwrap();
            writeln!(&mut kernel_body, "}}").unwrap();
            writeln!(&mut kernel_body).unwrap();
        }

        // Handle the < 4 remaining elements
        writeln!(&mut kernel_body, "while (out_index < end_axis_index) {{").unwrap();
        writeln!(
            &mut kernel_body,
            "let data = {input_tensor}[in_start_offset + out_index * {reduce_stride}];"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "{output_tensor}[out_start_offset + out_index * {reduce_stride}] = exp(data - m_all) / d_all;"
        )
        .unwrap();
        writeln!(&mut kernel_body, "out_index += 1u;").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

        kernel.push_body(&kernel_body);
    }
}

impl Operation for SoftmaxOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        let limits = device.limits();
        constraints.add_constraint(
            0,
            Constraint::less_than(limits.max_compute_workgroup_size_x + 1),
        );
        constraints.add_constraint(0, Constraint::more_than_or_equals(limits.min_subgroup_size));
        constraints.add_constraint(0, Constraint::less_than_or_equals(limits.max_subgroup_size));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let trimmed_tensor: TensorData = inputs[0].as_tensor().unwrap().clone();
        let workgroup_size = trimmed_tensor.layout().shape().iter().product::<usize>() as u32;

        [workgroup_size, 1, 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.value);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let dim = self.axis;
        let tensor = nodes.cached_results.get(&self.value).unwrap();
        let layout = tensor.layout();
        let shape = layout.shape();
        let output_type = self.out_datatype();
        let output_tensor = TensorData::new_for_shape(tensor.device(), shape, output_type);

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
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let max_blocksize = workgroup_shape.x();
        self.kernel(workgroup_shape, max_blocksize, kernel, &graph.device);
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!("softmax_{}_{}", self.rank(), self.datatype)
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<AnyComputeKey, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let input_layout = map.get(&self.value).unwrap();
        input_layout.clone()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax_slow() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [1f32, -2., -3., 4., 5., -6.];
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 6] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 6] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 6] = std::array::from_fn(|i| exp[i] / sum);

    println!("{softmax_array:?}");

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax_slow(0);
    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [1f32, -2., -3., 4., 5., -6.];
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 6] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 6] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 6] = std::array::from_fn(|i| exp[i] / sum);

    println!("{softmax_array:?}");

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax(0);
    let output = tensor.as_slice().await.unwrap();
    println!("output: {output:?}");
    println!("expect: {softmax_array:?}");
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax_large() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data: [f32; 1024] = std::array::from_fn(|_| rand::random::<f32>() * 10.0 - 5.0);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 1024] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 1024] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 1024] = std::array::from_fn(|i| exp[i] / sum);

    println!("{softmax_array:?}");

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax(0);
    let output = tensor.as_slice().await.unwrap();
    println!("output: {output:?}");
    println!("expect: {softmax_array:?}");
    for i in 0..1024 {
        assert!(
            (output[[i]] - softmax_array[i]).abs() < 0.001,
            "Mismatch at index {i}"
        );
    }
}
