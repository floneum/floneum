use std::ops::Range;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    DataTypeEnum, Layout, MatMulOperation, PairWiseFunction, ReduceFunction, Result, TensorInfo,
    composite::where_cond::WhereCondOperation,
    map_layout::{MapLayoutKind, MapLayoutOperation},
    mir::operation::Operation,
    pair_wise::PairWiseOperation,
    reduce::ReduceOperation,
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
    tensor::{LazyTensorData, TensorData},
};

use super::{BackwardRule, ComputeGraph, ComputeGraphInner, ComputeGraphNodeVariant, NodeIndex};

#[derive(Clone)]
struct NodeSnapshot {
    variant: ComputeGraphNodeVariant,
    info: TensorInfo,
    backward: Option<BackwardRule>,
}

impl ComputeGraph {
    pub(crate) fn backward(
        &self,
        target: NodeIndex,
        seed: LazyTensorData,
    ) -> Result<FxHashMap<NodeIndex, LazyTensorData>> {
        let snapshots = {
            let inner = self.inner.read();
            snapshot_subgraph(&inner, target)?
        };
        let target_info = snapshots
            .get(&target)
            .ok_or_else(|| crate::Error::msg("backpropagation target was not found"))?;

        if seed.info().shape() != target_info.info.shape()
            || seed.info().datatype() != target_info.info.datatype()
        {
            return Err(crate::Error::msg(format!(
                "gradient seed shape/datatype mismatch: expected {:?} {}, got {:?} {}",
                target_info.info.shape(),
                target_info.info.datatype(),
                seed.info().shape(),
                seed.info().datatype()
            )));
        }

        let mut gradients = FxHashMap::default();
        gradients.insert(target, seed);

        let mut visited = FxHashSet::default();
        let mut topo = Vec::new();
        build_topological_order(&snapshots, target, &mut visited, &mut topo);

        for node in topo.into_iter().rev() {
            let Some(grad) = gradients.get(&node).cloned() else {
                continue;
            };
            let snapshot = snapshots
                .get(&node)
                .ok_or_else(|| crate::Error::msg("missing node while backpropagating"))?;
            propagate_gradient(&snapshots, node, snapshot, ErasedTensor::from_lazy(grad), &mut gradients)?;
        }

        Ok(gradients)
    }
}

fn snapshot_subgraph(
    graph: &ComputeGraphInner,
    target: NodeIndex,
) -> Result<FxHashMap<NodeIndex, NodeSnapshot>> {
    let mut visited = FxHashSet::default();
    let mut order = Vec::new();
    collect_postorder(graph, target, &mut visited, &mut order)?;

    let mut infos = FxHashMap::default();
    let mut snapshots = FxHashMap::default();
    for node in order {
        let variant = graph
            .nodes
            .nodes
            .node_weight(node)
            .ok_or_else(|| crate::Error::msg(format!("missing node {node:?} in compute graph")))?
            .variant
            .clone();
        let info = infer_info(&variant, &infos)?;
        infos.insert(node, info.clone());
        let backward = graph
            .nodes
            .nodes
            .node_weight(node)
            .and_then(|node| node.backward.clone());
        snapshots.insert(
            node,
            NodeSnapshot {
                variant,
                info,
                backward,
            },
        );
    }
    Ok(snapshots)
}

fn collect_postorder(
    graph: &ComputeGraphInner,
    node: NodeIndex,
    visited: &mut FxHashSet<NodeIndex>,
    order: &mut Vec<NodeIndex>,
) -> Result<()> {
    if !visited.insert(node) {
        return Ok(());
    }

    let variant = graph
        .nodes
        .nodes
        .node_weight(node)
        .ok_or_else(|| crate::Error::msg(format!("missing node {node:?} in compute graph")))?
        .variant
        .clone();

    let mut dependencies = Vec::new();
    variant.visit_dependencies(&mut |dependency| {
        dependencies.push(dependency);
    });
    for dependency in dependencies {
        collect_postorder(graph, dependency, visited, order)?;
    }
    order.push(node);
    Ok(())
}

fn build_topological_order(
    snapshots: &FxHashMap<NodeIndex, NodeSnapshot>,
    node: NodeIndex,
    visited: &mut FxHashSet<NodeIndex>,
    order: &mut Vec<NodeIndex>,
) {
    if !visited.insert(node) {
        return;
    }

    if let Some(snapshot) = snapshots.get(&node) {
        let mut dependencies = Vec::new();
        snapshot.variant.visit_dependencies(&mut |dependency| {
            dependencies.push(dependency);
        });
        for dependency in dependencies {
            build_topological_order(snapshots, dependency, visited, order);
        }
    }

    order.push(node);
}

fn infer_info(
    variant: &ComputeGraphNodeVariant,
    infos: &FxHashMap<NodeIndex, TensorInfo>,
) -> Result<TensorInfo> {
    let info = match variant {
        ComputeGraphNodeVariant::ElementWise(op) => {
            TensorInfo::new(op.shape().into(), op.functions.out_datatype())
        }
        ComputeGraphNodeVariant::PairWise(op) => {
            TensorInfo::new(op.shape().into(), op.function.datatype)
        }
        ComputeGraphNodeVariant::Nary(op) => TensorInfo::new(op.shape.clone(), op.output_datatype),
        ComputeGraphNodeVariant::SliceAssign(op) => {
            let input = infos
                .get(&op.input)
                .ok_or_else(|| crate::Error::msg("slice_assign input info missing"))?;
            TensorInfo::new(op.input_shape.clone(), input.datatype())
        }
        ComputeGraphNodeVariant::Resize(op) => {
            let input = infos
                .get(&op.input)
                .ok_or_else(|| crate::Error::msg("resize input info missing"))?;
            TensorInfo::new(op.new_shape.clone(), input.datatype())
        }
        ComputeGraphNodeVariant::MapLayout(op) => {
            let input = infos
                .get(&op.input)
                .ok_or_else(|| crate::Error::msg("map_layout input info missing"))?;
            let layout = op.map_layout(&Layout::contiguous(input.shape()));
            TensorInfo::new(layout.shape().into(), input.datatype())
        }
        ComputeGraphNodeVariant::Dequantize(op) => {
            TensorInfo::new(
                op.matrix.shape().to_vec().into_boxed_slice(),
                op.post_dequantize.out_datatype(),
            )
        }
        ComputeGraphNodeVariant::MatMul(op) => {
            TensorInfo::new(op.out_shape.clone(), op.post_element_wise.out_datatype())
        }
        ComputeGraphNodeVariant::QMatMul(op) => TensorInfo::new(op.out_shape.clone(), op.input_datatype),
        ComputeGraphNodeVariant::Tensor(data) => {
            TensorInfo::new(data.layout().shape().into(), data.datatype())
        }
        ComputeGraphNodeVariant::Reduce(op) => {
            let shape = op
                .shape
                .iter()
                .enumerate()
                .filter_map(|(index, dim)| (index != op.axis).then_some(*dim))
                .collect();
            TensorInfo::new(shape, op.out_datatype())
        }
        ComputeGraphNodeVariant::IndexSelect(op) => {
            let input = infos
                .get(&op.input)
                .ok_or_else(|| crate::Error::msg("index_select input info missing"))?;
            TensorInfo::new(op.output_shape(), input.datatype())
        }
        ComputeGraphNodeVariant::WhereCond(op) => {
            TensorInfo::new(op.shape.clone(), op.output_datatype)
        }
        ComputeGraphNodeVariant::Custom(op) => {
            let layouts = infos
                .iter()
                .map(|(node, info)| {
                    (
                        *node,
                        crate::TensorLayoutInfo::new(Layout::contiguous(info.shape()), info.datatype()),
                    )
                })
                .collect();
            let layout = op.output_layout(&layouts);
            TensorInfo::new(layout.shape().into(), layout.datatype())
        }
    };
    Ok(info)
}

fn propagate_gradient(
    snapshots: &FxHashMap<NodeIndex, NodeSnapshot>,
    _node: NodeIndex,
    snapshot: &NodeSnapshot,
    gradient: ErasedTensor,
    gradients: &mut FxHashMap<NodeIndex, LazyTensorData>,
) -> Result<()> {
    if let Some(backward) = &snapshot.backward {
        for (dependency, dependency_gradient) in backward(gradient.into_lazy())? {
            accumulate_gradient(gradients, dependency, ErasedTensor::from_lazy(dependency_gradient));
        }
        return Ok(());
    }

    match &snapshot.variant {
        ComputeGraphNodeVariant::Tensor(_) => Ok(()),
        ComputeGraphNodeVariant::ElementWise(_) | ComputeGraphNodeVariant::PairWise(_) => {
            Err(crate::Error::msg(format!(
                "backpropagation does not support op `{}` without an attached backward rule",
                variant_name(&snapshot.variant)
            )))
        }
        ComputeGraphNodeVariant::MatMul(op) => {
            let first_info = snapshots
                .get(&op.first)
                .ok_or_else(|| crate::Error::msg("matmul lhs info missing"))?
                .info
                .clone();
            let second_info = snapshots
                .get(&op.second)
                .ok_or_else(|| crate::Error::msg("matmul rhs info missing"))?
                .info
                .clone();
            let first = ErasedTensor::reference(gradient.device().clone(), first_info, op.first);
            let second = ErasedTensor::reference(gradient.device().clone(), second_info, op.second);
            accumulate_gradient(gradients, op.first, gradient.mat_mul(&second.transpose_last_two()));
            accumulate_gradient(gradients, op.second, first.transpose_last_two().mat_mul(&gradient));
            Ok(())
        }
        ComputeGraphNodeVariant::Reduce(op) => {
            if op.function.name() != "sum" {
                return Err(crate::Error::msg(format!(
                    "backpropagation does not support reduce op `{}`",
                    op.function.name()
                )));
            }

            let input_shape = op.shape.clone();
            let mut keepdim_shape = input_shape.to_vec();
            keepdim_shape[op.axis] = 1;
            let input_grad = gradient.reshape(&keepdim_shape).broadcast_to(&input_shape);
            accumulate_gradient(gradients, op.value, input_grad);
            Ok(())
        }
        ComputeGraphNodeVariant::MapLayout(op) => {
            match &op.kind {
                MapLayoutKind::Slice(slices) => {
                    let input_info = snapshots
                        .get(&op.input)
                        .ok_or_else(|| crate::Error::msg("slice input info missing"))?
                        .info
                        .clone();
                    let zeros = ErasedTensor::zeros(
                        gradient.device().clone(),
                        input_info.shape(),
                        input_info.datatype(),
                    );
                    accumulate_gradient(gradients, op.input, zeros.slice_assign(&gradient, slices));
                }
                MapLayoutKind::Permute(axes) => {
                    let mut inverse = vec![0; axes.len()];
                    for (new_axis, old_axis) in axes.iter().copied().enumerate() {
                        inverse[old_axis] = new_axis;
                    }
                    accumulate_gradient(gradients, op.input, gradient.permute(&inverse));
                }
                MapLayoutKind::Broadcast => {
                    let input_info = snapshots
                        .get(&op.input)
                        .ok_or_else(|| crate::Error::msg("broadcast input info missing"))?
                        .info
                        .clone();
                    let reduce_axes = broadcast_reduce_axes(input_info.shape(), gradient.shape())?;
                    let mut reduced = gradient;
                    for axis in reduce_axes.into_iter().rev() {
                        reduced = reduced.sum(axis);
                    }
                    accumulate_gradient(gradients, op.input, reduced.reshape(input_info.shape()));
                }
                MapLayoutKind::Custom => {
                    return Err(crate::Error::msg(format!(
                        "backpropagation does not support custom layout op `{}`",
                        op.name()
                    )));
                }
            }
            Ok(())
        }
        ComputeGraphNodeVariant::Resize(op) => {
            let full_fill = op.fill_shape == op.new_shape;
            let same_elements =
                op.current_shape.iter().product::<usize>() == op.new_shape.iter().product::<usize>();
            if !full_fill || !same_elements {
                return Err(crate::Error::msg(format!(
                    "backpropagation only supports reshape-style resize ops, found `{}`",
                    op.name()
                )));
            }
            accumulate_gradient(gradients, op.input, gradient.reshape(&op.current_shape));
            Ok(())
        }
        ComputeGraphNodeVariant::SliceAssign(op) => {
            let value_info = snapshots
                .get(&op.value)
                .ok_or_else(|| crate::Error::msg("slice_assign value info missing"))?
                .info
                .clone();
            let zero_value = ErasedTensor::zeros(
                gradient.device().clone(),
                value_info.shape(),
                value_info.datatype(),
            );
            accumulate_gradient(
                gradients,
                op.input,
                gradient.clone().slice_assign(&zero_value, &op.slices),
            );
            accumulate_gradient(gradients, op.value, gradient.slice(&op.slices));
            Ok(())
        }
        ComputeGraphNodeVariant::WhereCond(op) => {
            let condition_info = snapshots
                .get(&op.condition)
                .ok_or_else(|| crate::Error::msg("where condition info missing"))?
                .info
                .clone();
            let condition = ErasedTensor::reference(
                gradient.device().clone(),
                condition_info,
                op.condition,
            );
            let zeros = ErasedTensor::zeros(
                gradient.device().clone(),
                gradient.shape(),
                gradient.datatype(),
            );
            accumulate_gradient(
                gradients,
                op.on_true,
                condition.where_cond(&gradient, &zeros),
            );
            accumulate_gradient(
                gradients,
                op.on_false,
                condition.where_cond(&zeros, &gradient),
            );
            Ok(())
        }
        ComputeGraphNodeVariant::Dequantize(_)
        | ComputeGraphNodeVariant::QMatMul(_)
        | ComputeGraphNodeVariant::IndexSelect(_)
        | ComputeGraphNodeVariant::Nary(_)
        | ComputeGraphNodeVariant::Custom(_) => Err(crate::Error::msg(format!(
            "backpropagation does not support op `{}`",
            variant_name(&snapshot.variant)
        ))),
    }
}

fn variant_name(variant: &ComputeGraphNodeVariant) -> &'static str {
    match variant {
        ComputeGraphNodeVariant::ElementWise(_) => "element_wise",
        ComputeGraphNodeVariant::PairWise(_) => "pair_wise",
        ComputeGraphNodeVariant::Nary(_) => "nary",
        ComputeGraphNodeVariant::SliceAssign(_) => "slice_assign",
        ComputeGraphNodeVariant::Resize(_) => "resize",
        ComputeGraphNodeVariant::MapLayout(_) => "map_layout",
        ComputeGraphNodeVariant::Dequantize(_) => "dequantize",
        ComputeGraphNodeVariant::MatMul(_) => "mat_mul",
        ComputeGraphNodeVariant::QMatMul(_) => "q_mat_mul",
        ComputeGraphNodeVariant::Tensor(_) => "tensor",
        ComputeGraphNodeVariant::Reduce(_) => "reduce",
        ComputeGraphNodeVariant::IndexSelect(_) => "index_select",
        ComputeGraphNodeVariant::WhereCond(_) => "where_cond",
        ComputeGraphNodeVariant::Custom(_) => "custom",
    }
}

fn accumulate_gradient(
    gradients: &mut FxHashMap<NodeIndex, LazyTensorData>,
    node: NodeIndex,
    gradient: ErasedTensor,
) {
    if let Some(existing) = gradients.get(&node).cloned() {
        let combined = ErasedTensor::from_lazy(existing).add(&gradient);
        gradients.insert(node, combined.into_lazy());
    } else {
        gradients.insert(node, gradient.into_lazy());
    }
}

fn broadcast_reduce_axes(input_shape: &[usize], output_shape: &[usize]) -> Result<Vec<usize>> {
    let mut reduce_axes = Vec::new();
    let mut input_iter = input_shape.iter().rev().peekable();

    for (axis, &target_dim) in output_shape.iter().enumerate().rev() {
        let reduce = if let Some(&&source_dim) = input_iter.peek() {
            if source_dim == target_dim || (source_dim == 1 && target_dim > 1) {
                input_iter.next();
                source_dim == 1 && target_dim > 1
            } else {
                target_dim > 1
            }
        } else {
            target_dim > 1
        };

        if reduce {
            reduce_axes.push(axis);
        }
    }

    if input_iter.next().is_some() {
        return Err(crate::Error::msg(format!(
            "failed to match broadcast input shape {input_shape:?} to output shape {output_shape:?}"
        )));
    }

    Ok(reduce_axes)
}

#[derive(Clone)]
struct ErasedTensor {
    data: LazyTensorData,
}

impl ErasedTensor {
    fn from_lazy(data: LazyTensorData) -> Self {
        Self { data }
    }

    fn reference(device: crate::Device, info: TensorInfo, key: NodeIndex) -> Self {
        Self {
            data: LazyTensorData::reference(device, info, key),
        }
    }

    fn zeros(device: crate::Device, shape: &[usize], datatype: DataTypeEnum) -> Self {
        let data = match datatype {
            DataTypeEnum::F32 => TensorData::new_splat(&device, shape, 0.0f32),
            DataTypeEnum::F16 => TensorData::new_splat(&device, shape, half::f16::ZERO),
            DataTypeEnum::U32 => TensorData::new_splat(&device, shape, 0u32),
        };
        Self::from_lazy(LazyTensorData::new(data))
    }

    fn into_lazy(self) -> LazyTensorData {
        self.data
    }

    fn shape(&self) -> &[usize] {
        self.data.info().shape()
    }

    fn datatype(&self) -> DataTypeEnum {
        self.data.info().datatype()
    }

    fn device(&self) -> &crate::Device {
        self.data.device()
    }

    fn key(&self) -> NodeIndex {
        self.data.key()
    }

    fn map_layout(
        &self,
        map_layout_fn: impl Fn(&Layout) -> Layout + Send + Sync + 'static,
    ) -> Self {
        Self::from_lazy(self.data.map_layout(MapLayoutOperation::new(
            self.key(),
            map_layout_fn,
        )))
    }

    fn reshape(&self, new_shape: &[usize]) -> Self {
        Self::from_lazy(self.data.resize(ResizeOperation::new(
            self.key(),
            self.shape().into(),
            new_shape.into(),
            new_shape.into(),
        )))
    }

    fn broadcast_to(&self, target_shape: &[usize]) -> Self {
        let target_shape: Box<[usize]> = target_shape.into();
        self.map_layout(move |layout| layout.broadcast_to(&target_shape))
    }

    fn permute(&self, axes: &[usize]) -> Self {
        let axes: Box<[usize]> = axes.into();
        self.map_layout(move |layout| layout.permute(&axes))
    }

    fn slice(&self, slices: &[Range<usize>]) -> Self {
        let slices: Box<[Range<usize>]> = slices.into();
        self.map_layout(move |layout| layout.slice(&slices))
    }

    fn slice_assign(&self, value: &Self, slices: &[Range<usize>]) -> Self {
        Self::from_lazy(self.data.slice_assign(SliceAssignOperation::new(
            self.key(),
            value.key(),
            slices.into(),
            self.shape().into(),
        )))
    }

    fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        let operation = WhereCondOperation::new(
            self.key(),
            on_true.key(),
            on_false.key(),
            self.datatype(),
            on_true.datatype(),
            on_true.shape(),
        );
        Self::from_lazy(on_true.data.where_cond(operation))
    }

    fn mat_mul(&self, other: &Self) -> Self {
        Self::from_lazy(self.data.mat_mul(MatMulOperation::new(
            self.datatype(),
            self.key(),
            other.key(),
            self.shape(),
            other.shape(),
            None,
        )))
    }

    fn sum(&self, axis: usize) -> Self {
        Self::from_lazy(self.data.reduce(ReduceOperation::new(
            self.key(),
            ReduceFunction {
                name: Some("sum".to_string()),
                operation: "let output = a + b;".to_string(),
                initial_value: "0.0".to_string(),
                datatype: self.datatype(),
            },
            axis,
            self.shape(),
        )))
    }

    fn add(&self, other: &Self) -> Self {
        Self::from_lazy(
            self.data
                .pair_wise(PairWiseOperation::new(
                    PairWiseFunction::new("let output = a + b;", self.datatype()),
                    self.key(),
                    other.key(),
                    self.shape(),
                )),
        )
    }

    fn transpose_last_two(&self) -> Self {
        let rank = self.shape().len();
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(rank - 1, rank - 2);
        self.permute(&axes)
    }
}
