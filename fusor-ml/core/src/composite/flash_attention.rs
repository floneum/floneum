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
    /// Computes flash attention without masking.
    /// 
    /// Args:
    ///   - k: Key tensor of same shape as self
    ///   - v: Value tensor of same shape as self
    ///   - scale: Scale factor (typically 1/sqrt(head_dim))
    pub fn flash_attention<const R2: usize>(&self, k: &Self, v: &Self, scale: f32, mask: Option<&Tensor<2, T>>) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
        T: crate::FloatDataType,
    {
        let operation = FlashAttentionOperation::new(
            self.key(),
            k.key(),
            v.key(),
            mask.map(|m| m.key()),
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
    pub(crate) mask: Option<NodeIndex>,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) scale: f32,
}

impl FlashAttentionOperation {
    pub fn new(
        q: NodeIndex,
        k: NodeIndex,
        v: NodeIndex,
        mask: Option<NodeIndex>,
        datatype: DataTypeEnum,
        _shape: &[usize],
        scale: f32,
    ) -> Self {
        Self {
            q,
            k,
            v,
            mask,
            datatype,
            scale,
        }
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
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
        let has_mask = self.mask.is_some();

        // Input tensors
        let q_tensor = kernel.add_tensor_input(4, false, dtype);
        let k_tensor = kernel.add_tensor_input(4, false, dtype);
        let v_tensor = kernel.add_tensor_input(4, false, dtype);
        let mask_tensor = if has_mask {
            Some(kernel.add_tensor_input(2, false, dtype))
        } else {
            None
        };
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

            // Apply attention mask if provided
            if let Some(mask) = &mask_tensor {
                write!(kernel, "    let mask_idx = ").unwrap();
                mask.strided_index(kernel, ["seq_idx", "k_seq"]);
                writeln!(kernel, ";").unwrap();
                writeln!(kernel, "    score = score + {mask}[mask_idx];").unwrap();
            }

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
        if let Some(mask) = self.mask {
            f(mask);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let q_tensor = nodes.get_cached_result(self.q).unwrap();
        let k_tensor = nodes.get_cached_result(self.k).unwrap();
        let v_tensor = nodes.get_cached_result(self.v).unwrap();

        let shape = q_tensor.layout().shape();
        let output_type = self.out_datatype();
        let output_tensor = TensorData::new_for_shape(q_tensor.device(), shape, output_type);

        let mut inputs = vec![
            MirValue::Tensor(q_tensor.clone()),
            MirValue::Tensor(k_tensor.clone()),
            MirValue::Tensor(v_tensor.clone()),
        ];

        if let Some(mask_idx) = self.mask {
            let mask_tensor = nodes.get_cached_result(mask_idx).unwrap();
            inputs.push(MirValue::Tensor(mask_tensor.clone()));
        }

        inputs.push(MirValue::Tensor(output_tensor.clone()));
        inputs
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
        // Output is the last input (after q, k, v, and optional mask)
        let output_idx = if self.mask.is_some() { 4 } else { 3 };
        let output_tensor: TensorData = inputs[output_idx].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        if self.mask.is_some() {
            "flash_attention_masked".to_string()
        } else {
            "flash_attention".to_string()
        }
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
async fn test_flash_attention() {
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
    let output = q.flash_attention(&k, &v, scale, None);
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

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention_causal_mask() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Test flash attention with causal mask - 4D tensors [batch, heads, seq, dim]
    let q_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let k_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let v_data = [[[[1.0f32, 2.0], [3.0, 4.0]]]]; // [1, 1, 2, 2]

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);

    // Create causal mask: lower triangular with 0s, upper triangular with -inf
    // For seq_len=2: [[0, -inf], [0, 0]]
    let neg_inf = f32::NEG_INFINITY;
    let causal_mask_data = [[0.0f32, neg_inf], [0.0, 0.0]]; // [2, 2]
    let causal_mask = Tensor::new(&device, &causal_mask_data);

    let scale = 1.0 / (2.0_f32.sqrt());

    // Test flash attention with causal mask
    let output = q.flash_attention_masked(&k, &v, scale, Some(&causal_mask));
    let result = output.as_slice().await.unwrap();

    // Compare with standard masked attention (non-fused implementation)
    let scores = q.mat_mul(&k.t()) * scale;
    // Reshape 2D mask [seq, seq] to 4D [1, 1, seq, seq] for broadcasting
    let causal_mask_4d: Tensor<4, f32> = causal_mask.reshape([1, 1, 2, 2]);
    let masked_scores = scores + causal_mask_4d;
    let attn_weights = masked_scores.softmax_last_dim();
    let expected = attn_weights.mat_mul(&v);
    let expected_result = expected.as_slice().await.unwrap();

    // Compare flash attention output against standard masked attention
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

    // Additional check: for causal mask, first row should only attend to first position
    // So first row output should equal first row of V
    assert!(
        (result[[0, 0, 0, 0]] - v_data[0][0][0][0]).abs() < tolerance,
        "First position should attend only to itself with causal mask: got {}, expected {}",
        result[[0, 0, 0, 0]],
        v_data[0][0][0][0]
    );
    assert!(
        (result[[0, 0, 0, 1]] - v_data[0][0][0][1]).abs() < tolerance,
        "First position should attend only to itself with causal mask: got {}, expected {}",
        result[[0, 0, 0, 1]],
        v_data[0][0][0][1]
    );
}
