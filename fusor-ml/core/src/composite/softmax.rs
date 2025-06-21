use std::{fmt::Write, sync::Arc};

use crate::{
    DataType, DataTypeEnum, Layout, Max, Sum, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{
        function::Function,
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
        let normalized = self - &max.broadcast(size);
        let exp = normalized.exp();
        let sum = exp.sum(dim);
        exp / sum.broadcast(size)
    }

    pub fn softmax_slow_last_dim(&self) -> Self {
        self.softmax_slow(self.rank() - 1)
    }

    pub fn softmax(&self, axis: usize) -> Self {
        let operation =
            SoftmaxOperation::new(self.key(), self.datatype(), axis, self.rank() as u32);
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }

    pub fn softmax_last_dim(&self) -> Self {
        self.softmax(self.rank() - 1)
    }
}

fn online_update(generic_kernel: &mut GenericKernel, dtype: DataTypeEnum) -> Function {
    generic_kernel.add_function(
        format!("vec2<{dtype}>"),
        "let m = max(old_m, x);
    let d = old_d * exp(old_m - m) + exp(x - m);
    let output = vec2<f32>(m, d);",
        [
            ("old_m".into(), dtype.to_string()),
            ("old_d".into(), dtype.to_string()),
            ("x".into(), dtype.to_string()),
        ],
    )
}

fn combine(generic_kernel: &mut GenericKernel, dtype: DataTypeEnum) -> Function {
    generic_kernel.add_function(
        format!("vec2<{dtype}>"),
        "let m = max(ma, mb);
    let d = da * exp(ma - m) + db * exp(mb - m);
    let output = vec2<f32>(m, d);",
        [
            ("ma".into(), dtype.to_string()),
            ("da".into(), dtype.to_string()),
            ("mb".into(), dtype.to_string()),
            ("db".into(), dtype.to_string()),
        ],
    )
}

#[derive(Debug, Clone)]
struct SoftmaxOperation {
    pub(crate) value: AnyComputeKey,
    pub(crate) axis: usize,
    pub(crate) rank: u32,
    pub(crate) datatype: DataTypeEnum,
}

impl SoftmaxOperation {
    pub fn new(value: AnyComputeKey, datatype: DataTypeEnum, axis: usize, rank: u32) -> Self {
        Self {
            value,
            axis,
            rank,
            datatype,
        }
    }

    pub fn reduce_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    fn kernel(&self, workgroup_shape: &WorkgroupShape, blocksize: u32, kernel: &mut GenericKernel) {
        let dtype = self.datatype;
        let out_datatype = self.out_datatype();
        let output_rank = self.rank - 1;

        let online_update_fn = online_update(kernel, dtype);
        let combine_fn = combine(kernel, dtype);

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

        let global_m_final = kernel.add_global_value(KernelGlobalSpace::Workgroup, dtype.clone());
        let global_d_final = kernel.add_global_value(KernelGlobalSpace::Workgroup, dtype.clone());

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
        writeln!(&mut kernel_body, "let data = {input_tensor}[in_index];",).unwrap();
        writeln!(
            &mut kernel_body,
            "let updated = {};",
            online_update_fn.call(["m_lane".into(), "d_lane".into(), "data".into(),].into())
        )
        .unwrap();
        writeln!(&mut kernel_body, "m_lane = updated.x;").unwrap();
        writeln!(&mut kernel_body, "d_lane = updated.y;").unwrap();
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
            "let m_peer = subgroupShuffleDown(m_lane, offset);"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let d_peer = subgroupShuffleDown(d_lane, offset);"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let updated = {};",
            combine_fn.call(
                [
                    "m_lane".into(),
                    "d_lane".into(),
                    "m_peer".into(),
                    "d_peer".into(),
                ]
                .into()
            )
        )
        .unwrap();
        writeln!(&mut kernel_body, "m_lane = updated.x;").unwrap();
        writeln!(&mut kernel_body, "d_lane = updated.y;").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
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
        writeln!(
            &mut kernel_body,
            "for (var offset = {subgroup_size} / 2u; offset > 0u; offset /= 2u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let m_peer = subgroupShuffleDown(m_lane, offset);"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let d_peer = subgroupShuffleDown(d_lane, offset);"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let updated = {};",
            combine_fn.call(
                [
                    "m_lane".into(),
                    "d_lane".into(),
                    "m_peer".into(),
                    "d_peer".into(),
                ]
                .into()
            )
        )
        .unwrap();
        writeln!(&mut kernel_body, "m_lane = updated.x;").unwrap();
        writeln!(&mut kernel_body, "d_lane = updated.y;").unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();

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
        writeln!(
            &mut kernel_body,
            "for (var i = 0u; i < bucket_size; i += 1u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel_body,
            "let axis_index = {workgroup_local_index} * bucket_size + i;"
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
            "let out_index = out_start_offset + axis_index * {reduce_stride};"
        )
        .unwrap();
        writeln!(&mut kernel_body, "let data = {input_tensor}[in_index];",).unwrap();
        writeln!(
            &mut kernel_body,
            "{output_tensor}[out_index] = exp(data - m_all) / d_all;"
        )
        .unwrap();
        writeln!(&mut kernel_body, "}}").unwrap();
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
        let trimmed_tensor: TensorData = inputs[0].as_tensor().unwrap().clone();
        let workgroup_size = trimmed_tensor.layout().shape().iter().product::<usize>() as u32;
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
        let output_type = self.out_datatype();
        let output_tensor =
            TensorData::new_for_shape(tensor.device(), &shape, output_type);

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

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!("softmax_{}_{}", self.rank, self.datatype)
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

    println!("{:?}", softmax_array);

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax_slow(0);
    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
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

    println!("{:?}", softmax_array);

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax(0);
    let output = tensor.as_slice().await.unwrap();
    println!("output: {:?}", output);
    println!("expect: {:?}", softmax_array);
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}
