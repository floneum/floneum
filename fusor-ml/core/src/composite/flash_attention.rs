use std::{fmt::Write, sync::Arc};

use crate::{
    DataType, DataTypeEnum, LastRank, Tensor, TensorData,
    compute_graph::NodeIndex,
    min_for_dtype,
    mir::{
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
};

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn flash_attention<const R2: usize>(&self, k: &Self, v: &Self, scale: f32) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
        T: crate::FloatDataType,
    {
        let operation = FlashAttentionOperation::new(
            self.key(),
            k.key(),
            v.key(),
            self.datatype(),
            self.shape(),
            scale,
        );
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }
}

#[derive(Debug, Clone)]
struct FlashAttentionOperation {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) scale: f32,
}

impl FlashAttentionOperation {
    pub fn new(
        q: NodeIndex,
        k: NodeIndex,
        v: NodeIndex,
        datatype: DataTypeEnum,
        _shape: &[usize],
        scale: f32,
    ) -> Self {
        Self {
            q,
            k,
            v,
            datatype,
            scale,
        }
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    fn rank(&self) -> u32 {
        4 // Flash attention works on 4D tensors (batch, heads, seq, dim)
    }

    fn kernel(
        &self,
        workgroup_shape: &WorkgroupShape,
        _blocksize: u32,
        kernel: &mut GenericKernel,
        _device: &crate::Device,
    ) {
        let dtype = self.datatype;
        let out_datatype = self.out_datatype();

        // Input tensors
        let q_tensor = kernel.add_tensor_input(4, false, dtype);
        let k_tensor = kernel.add_tensor_input(4, false, dtype);
        let v_tensor = kernel.add_tensor_input(4, false, dtype);
        let output_tensor = kernel.add_tensor_input(4, true, out_datatype);

        // Dimensions
        let batch_size = q_tensor.shape_binding(0);
        let num_heads = q_tensor.shape_binding(1);
        let seq_len = q_tensor.shape_binding(2);
        let head_dim = q_tensor.shape_binding(3);

        // Workgroup indices
        let workgroup_index = workgroup_shape.linearized_workgroup_index(kernel);
        let workgroup_local_index = kernel.workgroup_local_index();

        let workgroup_size = workgroup_shape.x();

        // Each thread computes one output element [batch, head, seq, dim]
        writeln!(
            kernel,
            "let global_thread_id = {} * {workgroup_size}u + {};",
            workgroup_index, workgroup_local_index
        )
        .unwrap();

        // Calculate output indices from global thread id
        writeln!(kernel, "var idx = global_thread_id;").unwrap();
        writeln!(kernel, "let out_dim = idx % {head_dim};").unwrap();
        writeln!(kernel, "idx /= {head_dim};").unwrap();
        writeln!(kernel, "let seq_idx = idx % {seq_len};").unwrap();
        writeln!(kernel, "idx /= {seq_len};").unwrap();
        writeln!(kernel, "let head_idx = idx % {num_heads};").unwrap();
        writeln!(kernel, "idx /= {num_heads};").unwrap();
        writeln!(kernel, "let batch_idx = idx;").unwrap();

        // Early exit if we're beyond valid elements
        writeln!(
            kernel,
            "let total_elements = {batch_size} * {num_heads} * {seq_len} * {head_dim};"
        )
        .unwrap();
        writeln!(kernel, "if global_thread_id >= total_elements {{").unwrap();
        writeln!(kernel, "    return;").unwrap();
        writeln!(kernel, "}}").unwrap();

        // Initialize online softmax variables
        writeln!(kernel, "var m = {};", min_for_dtype(dtype)).unwrap();
        writeln!(kernel, "var d = {dtype}(0.0);").unwrap();
        writeln!(kernel, "var acc = {dtype}(0.0);").unwrap();

        // Process all sequence positions for attention
        writeln!(
            kernel,
            "for (var k_seq = 0u; k_seq < {seq_len}; k_seq++) {{"
        )
        .unwrap();
        {
            // Compute attention score as full dot product over all head dimensions
            writeln!(kernel, "    var score = {dtype}(0.0);").unwrap();
            writeln!(
                kernel,
                "    for (var d_idx = 0u; d_idx < {head_dim}; d_idx++) {{"
            )
            .unwrap();
            {
                // Load Q value
                write!(kernel, "        let q_idx = ").unwrap();
                q_tensor.strided_index(kernel, ["batch_idx", "head_idx", "seq_idx", "d_idx"]);
                writeln!(kernel, ";").unwrap();
                writeln!(kernel, "        let q_val = {q_tensor}[q_idx];").unwrap();

                // Load K value
                write!(kernel, "        let k_idx = ").unwrap();
                k_tensor.strided_index(kernel, ["batch_idx", "head_idx", "k_seq", "d_idx"]);
                writeln!(kernel, ";").unwrap();
                writeln!(kernel, "        let k_val = {k_tensor}[k_idx];").unwrap();

                writeln!(kernel, "        score += q_val * k_val;").unwrap();
            }
            writeln!(kernel, "    }}").unwrap();
            writeln!(kernel, "    score = score * {};", self.scale).unwrap();

            // Load V value for the output dimension we're computing
            write!(kernel, "    let v_idx = ").unwrap();
            v_tensor.strided_index(kernel, ["batch_idx", "head_idx", "k_seq", "out_dim"]);
            writeln!(kernel, ";").unwrap();
            writeln!(kernel, "    let v_val = {v_tensor}[v_idx];").unwrap();

            // Online softmax update
            writeln!(kernel, "    let old_m = m;").unwrap();
            writeln!(kernel, "    m = max(m, score);").unwrap();
            writeln!(kernel, "    let exp_old_m_diff = exp(old_m - m);").unwrap();
            writeln!(kernel, "    let exp_score_diff = exp(score - m);").unwrap();
            writeln!(kernel, "    d = d * exp_old_m_diff + exp_score_diff;").unwrap();
            writeln!(
                kernel,
                "    acc = acc * exp_old_m_diff + exp_score_diff * v_val;"
            )
            .unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Write output
        write!(kernel, "let out_idx = ").unwrap();
        output_tensor.strided_index(kernel, ["batch_idx", "head_idx", "seq_idx", "out_dim"]);
        writeln!(kernel, ";").unwrap();
        writeln!(kernel, "{output_tensor}[out_idx] = acc / d;").unwrap();
    }
}

impl Operation for FlashAttentionOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(256));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let q_tensor = inputs[0].as_tensor().unwrap();
        let shape = q_tensor.layout().shape();

        let workgroup_size = workgroup_shape.x();

        // Total output elements = batch * heads * seq * dim
        let total_elements = (shape[0] * shape[1] * shape[2] * shape[3]) as u32;
        let total_workgroups = total_elements.div_ceil(workgroup_size);

        [total_workgroups, 1, 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.q);
        f(self.k);
        f(self.v);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let q_tensor = nodes.get_cached_result(self.q).unwrap();
        let k_tensor = nodes.get_cached_result(self.k).unwrap();
        let v_tensor = nodes.get_cached_result(self.v).unwrap();

        let shape = q_tensor.layout().shape();
        let output_type = self.out_datatype();
        let output_tensor = TensorData::new_for_shape(q_tensor.device(), shape, output_type);

        vec![
            MirValue::Tensor(q_tensor.clone()),
            MirValue::Tensor(k_tensor.clone()),
            MirValue::Tensor(v_tensor.clone()),
            MirValue::Tensor(output_tensor.clone()),
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
        let output_tensor: TensorData = inputs[3].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        "flash_attention".to_string()
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let input_layout = map.get(&self.q).unwrap();
        input_layout.clone()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention_masked() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Test flash attention - 4D tensors [batch, heads, seq, dim]
    let q_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let k_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let v_data = [[[[1.0f32, 2.0], [3.0, 4.0]]]]; // [1, 1, 2, 2]

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);

    let scale = 1.0 / (2.0_f32.sqrt());

    // Test flash attention
    let output = q.flash_attention(&k, &v, scale);
    let result = output.as_slice().await.unwrap();

    // Compare with standard attention (non-fused implementation)
    let scores = q.mat_mul(&k.t()) * scale;
    let attn_weights = scores.softmax_last_dim();
    let expected = attn_weights.mat_mul(&v);
    let expected_result = expected.as_slice().await.unwrap();

    // Compare flash attention output against standard attention
    let tolerance = 0.01;
    for i in 0..2 {
        for j in 0..2 {
            let flash_val = result[[0, 0, i, j]];
            let std_val = expected_result[[0, 0, i, j]];
            assert!(
                (flash_val - std_val).abs() < tolerance,
                "Mismatch at [{}, {}]: flash={}, standard={}",
                i,
                j,
                flash_val,
                std_val
            );
        }
    }
}
