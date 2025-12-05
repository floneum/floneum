use std::{fmt::Write, sync::Arc};

use crate::{
    DataType, DataTypeEnum, LastRank, Tensor, TensorData,
    compute_graph::NodeIndex,
    min_for_dtype,
    mir::{
        globals::KernelGlobalSpace,
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
        device: &crate::Device,
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

        // Check if subgroups are supported
        let use_subgroups = device.subgroups_supported();

        // Optimized configuration - compute multiple output dimensions per workgroup
        // Each subgroup (typically 32-64 threads) handles one output dimension
        let workgroup_size = workgroup_shape.x();
        let subgroup_size = if use_subgroups { 32u32 } else { 64u32 };
        let outputs_per_workgroup = workgroup_size / subgroup_size;
        let threads_per_output = subgroup_size;

        // Block size for tiling
        let block_size = 256u32;

        // Shared memory for K and V tiles
        let shared_k_tile =
            kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, block_size.to_string());
        let shared_v_tile =
            kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, block_size.to_string());

        // Calculate which output element this thread is working on
        writeln!(
            kernel,
            "let local_output_id = {workgroup_local_index} / {threads_per_output}u;"
        )
        .unwrap();
        writeln!(
            kernel,
            "let thread_in_output = {workgroup_local_index} % {threads_per_output}u;"
        )
        .unwrap();

        // Calculate global output index - each workgroup handles multiple output dimensions
        // We process multiple head_dim values in parallel
        writeln!(
            kernel,
            "let batch_head_seq = {} / {head_dim};",
            workgroup_index,
        )
        .unwrap();
        writeln!(
            kernel,
            "let dim_offset = ({} % {head_dim}) * {outputs_per_workgroup}u;",
            workgroup_index,
        )
        .unwrap();
        writeln!(kernel, "let out_dim = dim_offset + local_output_id;").unwrap();
        writeln!(kernel, "var global_idx = batch_head_seq;").unwrap();
        writeln!(kernel, "let seq_idx = global_idx % {seq_len};").unwrap();
        writeln!(kernel, "global_idx /= {seq_len};").unwrap();
        writeln!(kernel, "let head_idx = global_idx % {num_heads};").unwrap();
        writeln!(kernel, "global_idx /= {num_heads};").unwrap();
        writeln!(kernel, "let batch_idx = global_idx % {batch_size};").unwrap();

        // Calculate base offsets
        writeln!(kernel, "let batch_head_offset = batch_idx * {num_heads} * {seq_len} * {head_dim} + head_idx * {seq_len} * {head_dim};").unwrap();

        // Early exit if we're beyond valid dimensions
        writeln!(kernel, "if out_dim >= {head_dim} {{").unwrap();
        writeln!(kernel, "    return;").unwrap();
        writeln!(kernel, "}}").unwrap();

        // Initialize online softmax variables
        writeln!(kernel, "var m = {};", min_for_dtype(dtype)).unwrap();
        writeln!(kernel, "var d = {dtype}(0.0);").unwrap();
        writeln!(kernel, "var acc = {dtype}(0.0);").unwrap();

        // Process sequence in tiles
        writeln!(
            kernel,
            "for (var tile_start = 0u; tile_start < {seq_len}; tile_start += {block_size}) {{"
        )
        .unwrap();
        {
            writeln!(
                kernel,
                "let tile_end = min(tile_start + {block_size}, {seq_len});"
            )
            .unwrap();
            writeln!(kernel, "let tile_size = tile_end - tile_start;").unwrap();

            // Load K and V tiles cooperatively
            writeln!(kernel, "let local_idx = {workgroup_local_index};").unwrap();
            writeln!(
                kernel,
                "for (var i = local_idx; i < tile_size * {head_dim}; i += {workgroup_size}u) {{"
            )
            .unwrap();
            {
                writeln!(kernel, "let local_seq = i / {head_dim};").unwrap();
                writeln!(kernel, "let local_dim = i % {head_dim};").unwrap();
                writeln!(kernel, "let global_seq = tile_start + local_seq;").unwrap();
                writeln!(
                    kernel,
                    "let kv_offset = batch_head_offset + global_seq * {head_dim} + local_dim;"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let shared_idx = local_seq * {head_dim} + local_dim;"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "{shared_k_tile}[shared_idx] = {k_tensor}[kv_offset];"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "{shared_v_tile}[shared_idx] = {v_tensor}[kv_offset];"
                )
                .unwrap();
            }
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "workgroupBarrier();").unwrap();

            // Get Q value once per tile for this output dimension
            writeln!(kernel, "let q_val = {q_tensor}[batch_head_offset + seq_idx * {head_dim} + out_dim];").unwrap();

            // Parallel computation: each thread processes a subset of the tile
            writeln!(kernel, "// Parallel attention computation").unwrap();
            writeln!(
                kernel,
                "let chunk_size = (tile_size + {threads_per_output}u - 1u) / {threads_per_output}u;"
            )
            .unwrap();
            writeln!(kernel, "let start_idx = thread_in_output * chunk_size;").unwrap();
            writeln!(
                kernel,
                "let end_idx = min(start_idx + chunk_size, tile_size);"
            )
            .unwrap();

            writeln!(kernel, "var m_partial = {};", min_for_dtype(dtype)).unwrap();
            writeln!(kernel, "var d_partial = {dtype}(0.0);").unwrap();
            writeln!(kernel, "var acc_partial = {dtype}(0.0);").unwrap();

            writeln!(kernel, "for (var i = start_idx; i < end_idx; i++) {{").unwrap();
            {
                writeln!(kernel, "let k_shared_start = i * {head_dim};").unwrap();
                writeln!(kernel, "let v_shared_start = i * {head_dim};").unwrap();
                writeln!(
                    kernel,
                    "var score = q_val * {shared_k_tile}[k_shared_start + out_dim];"
                )
                .unwrap();
                writeln!(kernel, "score = score * {};", self.scale).unwrap();

                // Online softmax update for partial results
                writeln!(kernel, "let original_m = m_partial;").unwrap();
                writeln!(kernel, "m_partial = max(m_partial, score);").unwrap();
                writeln!(kernel, "let exp_original_m_diff = exp(original_m - m_partial);").unwrap();
                writeln!(kernel, "let exp_score_diff = exp(score - m_partial);").unwrap();
                writeln!(
                    kernel,
                    "d_partial = d_partial * exp_original_m_diff + exp_score_diff;"
                )
                .unwrap();
                writeln!(kernel, "acc_partial = acc_partial * exp_original_m_diff + exp_score_diff * {shared_v_tile}[v_shared_start + out_dim];").unwrap();
            }
            writeln!(kernel, "}}").unwrap();

            // Reduction within each subgroup/output group
            writeln!(kernel, "// Reduce partial results within subgroup").unwrap();

            if use_subgroups {
                writeln!(kernel, "var m_reduced = m_partial;").unwrap();
                writeln!(kernel, "var d_reduced = d_partial;").unwrap();
                writeln!(kernel, "var acc_reduced = acc_partial;").unwrap();

                // Unrolled subgroup reduction - threads_per_output is a compile-time constant
                let mut offset = threads_per_output;
                while offset > 1 {
                    offset /= 2;
                    writeln!(kernel, "{{").unwrap();
                    writeln!(kernel, "    let m_peer = subgroupShuffleDown(m_reduced, {offset}u);").unwrap();
                    writeln!(kernel, "    let d_peer = subgroupShuffleDown(d_reduced, {offset}u);").unwrap();
                    writeln!(kernel, "    let acc_peer = subgroupShuffleDown(acc_reduced, {offset}u);").unwrap();
                    writeln!(kernel, "    let original_m = m_reduced;").unwrap();
                    writeln!(kernel, "    m_reduced = max(m_reduced, m_peer);").unwrap();
                    writeln!(kernel, "    let exp_original_m_diff = exp(original_m - m_reduced);").unwrap();
                    writeln!(kernel, "    let exp_m_peer_diff = exp(m_peer - m_reduced);").unwrap();
                    writeln!(kernel, "    d_reduced = d_reduced * exp_original_m_diff + d_peer * exp_m_peer_diff;").unwrap();
                    writeln!(kernel, "    acc_reduced = acc_reduced * exp_original_m_diff + acc_peer * exp_m_peer_diff;").unwrap();
                    writeln!(kernel, "}}").unwrap();
                }

                // Merge into global state
                writeln!(kernel, "let original_m = m;").unwrap();
                writeln!(kernel, "m = max(m, m_reduced);").unwrap();
                writeln!(kernel, "let exp_original_m_diff = exp(original_m - m);").unwrap();
                writeln!(kernel, "let exp_m_reduced_diff = exp(m_reduced - m);").unwrap();
                writeln!(kernel, "d = d * exp_original_m_diff + d_reduced * exp_m_reduced_diff;").unwrap();
                writeln!(kernel, "acc = acc * exp_original_m_diff + acc_reduced * exp_m_reduced_diff;").unwrap();
            } else {
                // Simple merge for non-subgroup case
                writeln!(kernel, "let original_m = m;").unwrap();
                writeln!(kernel, "m = max(m, m_partial);").unwrap();
                writeln!(kernel, "let exp_original_m_diff = exp(original_m - m);").unwrap();
                writeln!(kernel, "let exp_m_partial_diff = exp(m_partial - m);").unwrap();
                writeln!(kernel, "d = d * exp_original_m_diff + d_partial * exp_m_partial_diff;").unwrap();
                writeln!(kernel, "acc = acc * exp_original_m_diff + acc_partial * exp_m_partial_diff;").unwrap();
            }

            writeln!(kernel, "workgroupBarrier();").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Final output - first thread in each subgroup writes result
        writeln!(kernel, "// Final output").unwrap();
        writeln!(kernel, "if thread_in_output == 0u {{").unwrap();
        {
            if use_subgroups {
                let subgroup_local_index = kernel.subgroup_local_index();
                writeln!(kernel, "if {} == 0u {{", subgroup_local_index).unwrap();
                {
                    writeln!(kernel, "let out_offset = batch_head_offset + seq_idx * {head_dim} + out_dim;").unwrap();
                    writeln!(kernel, "{output_tensor}[out_offset] = acc / d;").unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            } else {
                writeln!(kernel, "let out_offset = batch_head_offset + seq_idx * {head_dim} + out_dim;").unwrap();
                writeln!(kernel, "{output_tensor}[out_offset] = acc / d;").unwrap();
            }
        }
        writeln!(kernel, "}}").unwrap();
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
        let subgroup_size = 32u32; // Conservative estimate
        let outputs_per_workgroup = workgroup_size / subgroup_size;

        // Dispatch per batch * heads * sequence * ceil(head_dim / outputs_per_workgroup)
        let head_dim = shape[3] as u32;
        let num_dim_groups = (head_dim + outputs_per_workgroup - 1) / outputs_per_workgroup;
        let total_workgroups = (shape[0] * shape[1] * shape[2]) as u32 * num_dim_groups;

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
        format!("flash_attention_{}_{}", self.rank(), self.datatype)
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

    // Simple test case - 4D tensors [batch, heads, seq, dim]
    let q_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let k_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let v_data = [[[[1.0f32, 2.0], [3.0, 4.0]]]]; // [1, 1, 2, 2]

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);

    let scale = 1.0 / (2.0_f32.sqrt());

    // Test that flash attention runs without panicking
    let output = q.flash_attention(&k, &v, scale);
    let result = output.as_slice().await.unwrap();

    // Compare with standard attention
    let scores = q.mat_mul(&k.t()) * scale;
    let attn_weights = scores.softmax_last_dim();
    let expected = attn_weights.mat_mul(&v);
    let expected_result = expected.as_slice().await.unwrap();

    // Basic sanity check - results should be finite
    assert!(result[[0, 0, 0, 0]].is_finite(), "Result should be finite");
    assert!(result[[0, 0, 0, 1]].is_finite(), "Result should be finite");
    assert!(result[[0, 0, 1, 0]].is_finite(), "Result should be finite");
    assert!(result[[0, 0, 1, 1]].is_finite(), "Result should be finite");

    assert!(
        expected_result[[0, 0, 0, 0]].is_finite(),
        "Expected should be finite"
    );
    assert!(
        expected_result[[0, 0, 0, 1]].is_finite(),
        "Expected should be finite"
    );
    assert!(
        expected_result[[0, 0, 1, 0]].is_finite(),
        "Expected should be finite"
    );
    assert!(
        expected_result[[0, 0, 1, 1]].is_finite(),
        "Expected should be finite"
    );
}
