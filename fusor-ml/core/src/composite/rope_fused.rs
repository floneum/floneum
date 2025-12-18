use std::fmt::Write;
use std::sync::Arc;

use crate::{
    DataType, Device, Tensor, TensorData,
    compute_graph::NodeIndex,
    layout::TILE_SIZE,
    mir::{
        inputs::MirValue, kernel::GenericKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    tensor::DataTypeEnum,
    visit_tiled::{
        MaybeQTensorInput, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

impl<D: DataType> Tensor<4, D> {
    /// Apply fused interleaved RoPE (rotary position embedding).
    /// This pairs adjacent elements: (0, 1), (2, 3), etc.
    pub fn rope_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, true)
    }

    /// Apply fused normal RoPE (rotary position embedding).
    /// This pairs first half with second half: (0, head_dim/2), (1, head_dim/2+1), etc.
    pub fn rope_normal_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, false)
    }

    fn rope_fused_impl(
        &self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
        interleaved: bool,
    ) -> Tensor<4, D> {
        let [_, _, sequence_length, _] = *self.shape();
        // Narrow cos/sin to the sequence length
        let cos = cos.narrow(0, 0, sequence_length);
        let sin = sin.narrow(0, 0, sequence_length);

        let operation = RopeFusedOperation::new(
            self.key(),
            cos.key(),
            sin.key(),
            self.datatype(),
            self.shape(),
            interleaved,
        );
        let data = self.data();

        Tensor::from_parts(data.custom(Arc::new(operation)))
    }
}

#[derive(Debug, Clone)]
struct RopeFusedOperation {
    pub(crate) input: NodeIndex,
    pub(crate) cos: NodeIndex,
    pub(crate) sin: NodeIndex,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) shape: Box<[usize]>,
    pub(crate) interleaved: bool,
}

impl RopeFusedOperation {
    pub fn new(
        input: NodeIndex,
        cos: NodeIndex,
        sin: NodeIndex,
        datatype: DataTypeEnum,
        shape: &[usize],
        interleaved: bool,
    ) -> Self {
        Self {
            input,
            cos,
            sin,
            datatype,
            shape: shape.into(),
            interleaved,
        }
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    fn kernel(
        &self,
        device: &Device,
        _workgroup_shape: &WorkgroupShape,
        kernel: &mut GenericKernel,
    ) {
        // We add cos and sin inputs manually first (bindings 0 and 1)
        let cos_input = kernel.add_tensor_input(2, true, self.datatype);
        let sin_input = kernel.add_tensor_input(2, true, self.datatype);

        // Then we let build_visit_tiled_kernel handle input (binding 2) and output (binding 3)
        let datatypes = vec![
            self.datatype.into(), // input
            self.datatype.into(), // output
        ];

        let interleaved = self.interleaved;
        let head_dim = *self.shape.last().unwrap();

        build_visit_tiled_kernel(
            device,
            &self.shape,
            TILE_SIZE,
            datatypes,
            |kernel, indexes, tensors, values| {
                let input_tensor = &tensors[0]; // This is 'self'
                let input_val = &values[0];

                // Output
                let output_index = &indexes[1];
                let out_tensor = &tensors[1];

                let rank = self.rank();
                // Dimensions are dim_0, dim_1, ..., dim_{rank-1}
                // For rank 4: dim_0=batch, dim_1=head, dim_2=seq, dim_3=head_dim

                let dim_last = format!("dim_{}", rank - 1);
                let dim_seq = format!("dim_{}", rank - 2);

                if interleaved {
                    // Interleaved RoPE: pairs (0,1), (2,3), etc.
                    // Cos/Sin index: [dim_seq, dim_last / 2]
                    let freq_idx_0 = dim_seq;
                    let freq_idx_1 = format!("{} / 2u", dim_last);

                    let cos_idx_var = "cos_idx";
                    let sin_idx_var = "sin_idx";

                    write!(kernel, "let {} = ", cos_idx_var).unwrap();
                    cos_input.strided_index(kernel, vec![freq_idx_0.clone(), freq_idx_1.clone()]);
                    writeln!(kernel, ";").unwrap();

                    write!(kernel, "let {} = ", sin_idx_var).unwrap();
                    sin_input.strided_index(kernel, vec![freq_idx_0.clone(), freq_idx_1.clone()]);
                    writeln!(kernel, ";").unwrap();

                    let cos_val = format!("{}[{}]", cos_input, cos_idx_var);
                    let sin_val = format!("{}[{}]", sin_input, sin_idx_var);

                    // Neighbor index
                    let neighbor_last_dim = "neighbor_last_dim";
                    writeln!(kernel, "let is_even = ({} % 2u) == 0u;", dim_last).unwrap();
                    writeln!(
                        kernel,
                        "let {} = select({} - 1u, {} + 1u, is_even);",
                        neighbor_last_dim, dim_last, dim_last
                    )
                    .unwrap();

                    let neighbor_idx_var = "neighbor_idx";
                    write!(kernel, "let {} = ", neighbor_idx_var).unwrap();

                    // Reconstruct neighbor dims: dim_0, ..., dim_{rank-2}, neighbor_last_dim
                    let mut neighbor_dims = Vec::new();
                    for i in 0..rank - 1 {
                        neighbor_dims.push(format!("dim_{}", i));
                    }
                    neighbor_dims.push(neighbor_last_dim.to_string());

                    match input_tensor {
                        MaybeQTensorInput::Tensor(t) => t.strided_index(kernel, neighbor_dims),
                        _ => panic!("Expected tensor input"),
                    }
                    writeln!(kernel, ";").unwrap();

                    let neighbor_val = format!("{}[{}]", input_tensor, neighbor_idx_var);

                    format!(
                        "{out_tensor}[{output_index}] = {input_val} * {cos_val} + {neighbor_val} * select({sin_val}, -{sin_val}, is_even);"
                    )
                } else {
                    // Normal RoPE: pairs (0, half), (1, half+1), etc.
                    // For d < half: output[d] = x[d] * cos - x[d + half] * sin
                    // For d >= half: output[d] = x[d] * cos + x[d - half] * sin
                    let half = head_dim / 2;

                    // Cos/Sin index: [dim_seq, dim_last % half]
                    // (cos/sin are the same for both halves)
                    let freq_idx_0 = dim_seq;
                    let freq_idx_1 = format!("{} % {}u", dim_last, half);

                    let cos_idx_var = "cos_idx";
                    let sin_idx_var = "sin_idx";

                    write!(kernel, "let {} = ", cos_idx_var).unwrap();
                    cos_input.strided_index(kernel, vec![freq_idx_0.clone(), freq_idx_1.clone()]);
                    writeln!(kernel, ";").unwrap();

                    write!(kernel, "let {} = ", sin_idx_var).unwrap();
                    sin_input.strided_index(kernel, vec![freq_idx_0.clone(), freq_idx_1.clone()]);
                    writeln!(kernel, ";").unwrap();

                    let cos_val = format!("{}[{}]", cos_input, cos_idx_var);
                    let sin_val = format!("{}[{}]", sin_input, sin_idx_var);

                    // Neighbor index
                    let neighbor_last_dim = "neighbor_last_dim";
                    writeln!(kernel, "let in_first_half = {} < {}u;", dim_last, half).unwrap();
                    writeln!(
                        kernel,
                        "let {} = select({} - {}u, {} + {}u, in_first_half);",
                        neighbor_last_dim, dim_last, half, dim_last, half
                    )
                    .unwrap();

                    let neighbor_idx_var = "neighbor_idx";
                    write!(kernel, "let {} = ", neighbor_idx_var).unwrap();

                    // Reconstruct neighbor dims: dim_0, ..., dim_{rank-2}, neighbor_last_dim
                    let mut neighbor_dims = Vec::new();
                    for i in 0..rank - 1 {
                        neighbor_dims.push(format!("dim_{}", i));
                    }
                    neighbor_dims.push(neighbor_last_dim.to_string());

                    match input_tensor {
                        MaybeQTensorInput::Tensor(t) => t.strided_index(kernel, neighbor_dims),
                        _ => panic!("Expected tensor input"),
                    }
                    writeln!(kernel, ";").unwrap();

                    let neighbor_val = format!("{}[{}]", input_tensor, neighbor_idx_var);

                    // For first half: x * cos - neighbor * sin
                    // For second half: x * cos + neighbor * sin
                    format!(
                        "{out_tensor}[{output_index}] = {input_val} * {cos_val} + {neighbor_val} * select({sin_val}, -{sin_val}, in_first_half);"
                    )
                }
            },
            kernel,
        );
    }
}

impl Operation for RopeFusedOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.shape, device)
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.shape)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.cos);
        f(self.sin);
        f(self.input);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap();
        let cos = nodes.get_cached_result(self.cos).unwrap();
        let sin = nodes.get_cached_result(self.sin).unwrap();

        let output_tensor =
            TensorData::new_for_shape(input.device(), input.layout().shape(), self.datatype);

        // Order must match kernel binding order: cos, sin, input, output
        vec![
            cos.clone().into(),
            sin.clone().into(),
            input.clone().into(),
            output_tensor.into(),
        ]
    }

    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        self.kernel(&graph.device, workgroup_shape, kernel);
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        // Output is the last input
        inputs[3].clone()
    }

    fn name(&self) -> String {
        let mode = if self.interleaved {
            "interleaved"
        } else {
            "normal"
        };
        format!("rope_fused_{}_{}_{}", mode, self.rank(), self.datatype)
    }

    fn output_layout(
        &self,
        layouts: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let input = layouts.get(&self.input).unwrap();
        input.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_rope_fused_interleaved() {
        let device = Device::test_instance();

        let pos_shape = [11, 32]; // seq_len=11, head_dim/2=32 -> head_dim=64
        let cos_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect();
        let sin_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .sin()
                    })
                    .collect()
            })
            .collect();

        let cos = Tensor::new(&device, &cos_data);
        let sin = Tensor::new(&device, &sin_data);

        // Input: [1, 3, 11, 64]
        let shape = [1, 3, 11, 64];
        let data: Vec<Vec<Vec<Vec<f32>>>> = (0..shape[0])
            .map(|_| {
                (0..shape[1])
                    .map(|_| {
                        (0..shape[2])
                            .map(|_| {
                                (0..shape[3])
                                    .map(|k| (k as f32).sin()) // Some deterministic data
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let x = Tensor::new(&device, &data);

        // Compare interleaved vs fused
        let rope_original = x.rope_interleaved(&cos, &sin);
        let rope_fused = x.rope_fused(&cos, &sin);

        let original = rope_original.as_slice().await.unwrap();
        let fused = rope_fused.as_slice().await.unwrap();

        // Verify
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        let original_val = original[[b, h, s, d]];
                        let fused_val = fused[[b, h, s, d]];
                        let diff = (original_val - fused_val).abs();
                        assert!(
                            diff < 1e-4,
                            "Mismatch at index [{}, {}, {}, {}]: original {} vs fused {}, diff {}",
                            b,
                            h,
                            s,
                            d,
                            original_val,
                            fused_val,
                            diff
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_rope_fused_normal() {
        let device = Device::test_instance();

        let pos_shape = [11, 32]; // seq_len=11, head_dim/2=32 -> head_dim=64
        let cos_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect();
        let sin_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .sin()
                    })
                    .collect()
            })
            .collect();

        let cos = Tensor::new(&device, &cos_data);
        let sin = Tensor::new(&device, &sin_data);

        // Input: [1, 3, 11, 64]
        let shape = [1, 3, 11, 64];
        let data: Vec<Vec<Vec<Vec<f32>>>> = (0..shape[0])
            .map(|_| {
                (0..shape[1])
                    .map(|_| {
                        (0..shape[2])
                            .map(|_| {
                                (0..shape[3])
                                    .map(|k| (k as f32).sin()) // Some deterministic data
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let x = Tensor::new(&device, &data);

        // Compare normal rope vs fused normal
        let rope_original = x.rope(&cos, &sin);
        let rope_fused = x.rope_normal_fused(&cos, &sin);

        let original = rope_original.as_slice().await.unwrap();
        let fused = rope_fused.as_slice().await.unwrap();

        // Verify
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        let original_val = original[[b, h, s, d]];
                        let fused_val = fused[[b, h, s, d]];
                        let diff = (original_val - fused_val).abs();
                        assert!(
                            diff < 1e-4,
                            "Mismatch at index [{}, {}, {}, {}]: original {} vs fused {}, diff {}",
                            b,
                            h,
                            s,
                            d,
                            original_val,
                            fused_val,
                            diff
                        );
                    }
                }
            }
        }
    }
}
