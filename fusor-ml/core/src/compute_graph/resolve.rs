use std::fmt::Write;
use std::sync::Arc;

use rustc_hash::FxHashSet;
use wgpu::CommandEncoder;

use crate::{
    DataTypeEnum, ElementWiseFunctions, ElementWiseOperation, MatMulOperation, PairWiseOperation,
    ReduceOperation,
    dequantize::DequantizeOperation,
    index_select::IndexSelectOperation,
    mir::{
        inputs::{KernelInputValue, MirValue},
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{self, WorkgroupShapeConstraints},
    },
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};

use super::{ComputeGraphInner, ComputeGraphNodeVariant, NodeIndex, queue::ComputeQueue};

pub(crate) struct Resolver<'a> {
    command_encoder: &'a mut CommandEncoder,
    queued_operations: Vec<(NodeIndex, Arc<dyn Operation>)>,
    target: NodeIndex,
    queue: ComputeQueue,
    resolved_set: FxHashSet<NodeIndex>,
}

impl<'a> Resolver<'a> {
    pub(crate) fn new(
        graph: &mut ComputeGraphInner,
        target: NodeIndex,
        command_encoder: &'a mut CommandEncoder,
    ) -> Self {
        let resolved_set = graph
            .nodes
            .nodes
            .node_indices()
            .filter(|&idx| {
                graph
                    .nodes
                    .nodes
                    .node_weight(idx)
                    .map(|n| n.cached.is_some())
                    .unwrap_or(false)
            })
            .collect();
        Self {
            command_encoder,
            target,
            queued_operations: Vec::new(),
            queue: Default::default(),
            resolved_set,
        }
    }

    pub(crate) fn run(&mut self, graph: &mut ComputeGraphInner) -> TensorData {
        let limits = graph.device.limits();
        self.queue_operations(graph);
        let queued_operations = std::mem::take(&mut self.queued_operations);

        // Find runs of compatible dispatch shapes
        let mut current_constraints = WorkgroupShapeConstraints::new();
        let mut pending_operations = Vec::new();
        let mut inputs = Vec::new();
        let mut all_input_values = Vec::new();
        let mut kernel = GenericKernel::new();

        for (node, operation) in queued_operations {
            let new_inputs = operation.inputs(graph);
            let constraint = operation.workgroup_shape_constraints(&graph.device);
            let mut new_merged = current_constraints.clone();
            new_merged.merge(&constraint);
            let old_best = current_constraints.solve(&limits).unwrap_or_else(|| {
                panic!(
                    "Failed to find a valid workgroup shape for constraints {current_constraints:?}"
                )
            });
            let mut extend = self.should_extend_kernel(new_inputs.clone(), &inputs);
            extend &= new_merged.solve(&limits).is_some();
            if extend {
                current_constraints = new_merged;
            } else {
                self.flush_operations(
                    graph,
                    &mut kernel,
                    &pending_operations,
                    &inputs,
                    &all_input_values,
                    old_best,
                );
                pending_operations.clear();
                all_input_values.clear();
                inputs.clear();
                kernel.clear();
                current_constraints = constraint;
            }
            // Map layout isn't really a kernel. Resolve it immediately
            let map_layout = if let Some(node_data) = graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::MapLayout(map_layout) => Some(map_layout.clone()),
                    ComputeGraphNodeVariant::Resize(resize) => resize.lower(graph),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(map_layout) = map_layout {
                let result = map_layout.run(graph);
                // Cache the result
                graph.set_cached_result(node, result);
            } else {
                self.push_operation(
                    graph,
                    new_inputs,
                    &mut kernel,
                    node,
                    operation,
                    &mut inputs,
                    &mut all_input_values,
                    &mut pending_operations,
                );
            };
        }

        if !pending_operations.is_empty() {
            let old_best = current_constraints.solve(&limits).unwrap_or_else(|| {
                panic!(
                    "Failed to find a valid workgroup shape for constraints {current_constraints:?}"
                )
            });
            self.flush_operations(
                graph,
                &mut kernel,
                &pending_operations,
                &inputs,
                &all_input_values,
                old_best,
            );
        }

        graph
            .get_result(self.target)
            .expect("Target result not cached")
    }

    fn should_extend_kernel(&mut self, _: Vec<MirValue>, _: &[Vec<MirValue>]) -> bool {
        // TODO: Restore with better testing. This passes all tests in fusor, but breaks rbert and rwhisper
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn push_operation(
        &mut self,
        graph: &mut ComputeGraphInner,
        new_inputs: Vec<MirValue>,
        kernel: &mut GenericKernel,
        key: NodeIndex,
        operation: Arc<dyn Operation>,
        inputs: &mut Vec<Vec<MirValue>>,
        all_input_values: &mut Vec<KernelInputValue>,
        queued_operations: &mut Vec<(NodeIndex, Arc<dyn Operation>)>,
    ) {
        for input in &new_inputs {
            input.visit_input_values(|value| {
                if let Some(index) = all_input_values.iter().position(|x| *x == value) {
                    kernel.pre_register_binding(index as _);
                } else {
                    kernel.pre_register_binding(all_input_values.len() as _);
                    all_input_values.push(value.clone());
                }
            });
        }
        let result = operation.output(graph, &new_inputs);
        let MirValue::Tensor(resolved) = result else {
            panic!("Kernel input value is not a tensor");
        };
        // Cache the result
        graph.set_cached_result(key, resolved);
        inputs.push(new_inputs);
        queued_operations.push((key, operation));
    }

    fn flush_operations(
        &mut self,
        graph: &mut ComputeGraphInner,
        mut kernel: &mut GenericKernel,
        queued_operations: &[(NodeIndex, Arc<dyn Operation>)],
        inputs: &[Vec<MirValue>],
        all_input_values: &[KernelInputValue],
        workgroup_shape: workgroup_shape::WorkgroupShape,
    ) {
        let mut max_dispatch_size = [0; 3];
        for ((key, operation), inputs) in queued_operations.iter().zip(inputs) {
            // Map layout isn't really a kernel. Skip it
            if let Some(node) = graph.nodes.nodes.node_weight(*key)
                && matches!(node.variant, ComputeGraphNodeVariant::MapLayout(_))
            {
                continue;
            }

            let dispatch_size = operation.dispatch_size(&workgroup_shape, inputs);
            for (new, max) in dispatch_size.iter().zip(max_dispatch_size.iter_mut()) {
                *max = (*max).max(*new);
            }
            if cfg!(debug_assertions) {
                writeln!(&mut kernel, "{{ // start {}", operation.name()).unwrap();
            } else {
                writeln!(&mut kernel, "{{").unwrap();
            }
            operation.build_kernel(graph, &workgroup_shape, inputs, kernel);
            let name = kernel.name_mut();
            if !name.is_empty() {
                *name += "->";
            }
            *name += &operation.name();
            if cfg!(debug_assertions) {
                writeln!(&mut kernel, "}} // end {}", operation.name()).unwrap();
            } else {
                writeln!(&mut kernel, "}}").unwrap();
            }
            // Check if that makes any of this node's dependencies dead
            let mut dependencies = Vec::new();
            graph.visit_dependencies(*key, &mut |dependent_key| {
                dependencies.push(dependent_key);
            });
            for dependency in dependencies {
                graph.check_life(dependency);
            }
        }
        kernel.set_workgroup_size(workgroup_shape);
        kernel.run(
            &graph.device,
            all_input_values,
            self.command_encoder,
            max_dispatch_size,
        );
    }

    pub(crate) fn queue_operations(&mut self, graph: &mut ComputeGraphInner) {
        self.queue.push_back(self.target);

        while let Some(node) = self.queue.pop_front() {
            if self.resolved_set.contains(&node) {
                continue;
            }

            let node_data = graph
                .nodes
                .nodes
                .node_weight(node)
                .expect("Node not found in graph");
            let variant = node_data.variant.clone();
            let resolved = match variant {
                ComputeGraphNodeVariant::ElementWise(op) => {
                    self.resolve_element_wise(graph, node, &op)
                }
                ComputeGraphNodeVariant::PairWise(op) => self.resolve_pair_wise(graph, &op),
                ComputeGraphNodeVariant::MatMul(op) => self.resolve_mat_mul(graph, &op),
                ComputeGraphNodeVariant::Reduce(op) => self.resolve_reduce(graph, &op),
                ComputeGraphNodeVariant::Tensor(op) => {
                    self.resolve_tensor(graph, node, &op);
                    continue;
                }
                ComputeGraphNodeVariant::MapLayout(op) => self.resolve_map_layout(&op),
                ComputeGraphNodeVariant::Resize(op) => self.resolve_resize(&op),
                ComputeGraphNodeVariant::SliceAssign(op) => self.resolve_slice_assign(&op),
                ComputeGraphNodeVariant::IndexSelect(op) => self.resolve_index_select(graph, &op),
                ComputeGraphNodeVariant::QMatMul(op) => self.resolve_q_mat_mul(&op),
                ComputeGraphNodeVariant::Dequantize(op) => self.resolve_dequantize(&op),
                ComputeGraphNodeVariant::Custom(op) => self.resolve_custom(&op),
            };
            let mut dependencies = Vec::new();
            resolved.visit_dependencies(&mut |dependency| {
                if !self.resolved_set.contains(&dependency) {
                    dependencies.push(dependency);
                }
            });
            if !dependencies.is_empty() {
                for dependency in dependencies {
                    self.queue.push_back(dependency);
                }
                // If there are dependencies that are not resolved, push them to the queue then
                // revisit this node
                self.queue.push_back(node);
                continue;
            }

            // Mark this node as resolved
            self.resolved_set.insert(node);
            self.queued_operations.push((node, resolved));
        }
    }

    fn collect_element_wise_ops(
        &mut self,
        graph: &mut ComputeGraphInner,
        key: NodeIndex,
        initial_op: Option<&ElementWiseOperation>,
    ) -> ElementWiseOperation {
        let mut functions = Vec::new();
        let mut current_key = key;
        let mut ty = DataTypeEnum::F32;
        let mut shape = Box::new([]) as Box<[usize]>;

        if let Some(op) = initial_op {
            ty = op.input_datatype();
            shape = op.shape().into();
            functions.extend(op.functions.iter().cloned());
            current_key = op.value;
        }

        while let Some(node) = graph.nodes.nodes.node_weight(current_key) {
            if let ComputeGraphNodeVariant::ElementWise(operation) = &node.variant {
                // If the result is already cached, stop collecting element wise ops
                if let Some(cached) = &node.cached {
                    ty = cached.datatype();
                    shape = cached.layout().shape().into();
                    break;
                }
                ty = operation.input_datatype();
                shape = operation.shape().into();
                functions.extend(operation.functions.iter().cloned());
                current_key = operation.value;
            } else {
                break;
            }
        }
        let functions = ElementWiseFunctions::new(functions, ty);
        ElementWiseOperation::from_element_wise(current_key, functions, shape)
    }

    fn resolve_element_wise(
        &mut self,
        graph: &mut ComputeGraphInner,
        key: NodeIndex,
        operation: &ElementWiseOperation,
    ) -> Arc<dyn Operation> {
        // First collect all element wise ops in this chain
        let functions = self.collect_element_wise_ops(graph, key, Some(operation));
        let input = functions.value;
        let input_cached = self.resolved_set.contains(&input);

        if !input_cached {
            let variant = graph
                .nodes
                .nodes
                .node_weight(input)
                .map(|node| node.variant.clone());

            if let Some(variant) = variant {
                match variant {
                    // Merge into the output of the reduce kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::Reduce(op) => {
                        return self.resolve_reduce_then(graph, &op, Some(functions));
                    }
                    // Merge into the output of the pair wise kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::PairWise(op) => {
                        return self.resolve_pair_wise_then(graph, &op, Some(functions));
                    }
                    // Merge into the output of the mat mul kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::MatMul(op) => {
                        return self.resolve_mat_mul_then(graph, &op, Some(functions));
                    }
                    // Merge into the output of the dequantize kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::Dequantize(op) => {
                        return self.resolve_dequantize_then(&op, Some(functions));
                    }
                    _ => {}
                }
            }
        }
        let shape: Box<[_]> = functions.shape().into();
        // Otherwise, just run the element wise kernel
        let kernel = ElementWiseOperation::from_element_wise(input, functions.functions, shape);

        Arc::new(kernel)
    }

    fn resolve_pair_wise(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &PairWiseOperation,
    ) -> Arc<dyn Operation> {
        self.resolve_pair_wise_then(graph, operation, None)
    }

    fn resolve_pair_wise_then(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &PairWiseOperation,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let function = operation.function.clone();
        let shape: Box<[usize]> = operation.shape().into();

        let mut first_input = operation.first;
        let first_pre_element_wise = operation.pre_element_wise[0].clone();
        let mut second_input = operation.second;
        let second_pre_element_wise = operation.pre_element_wise[1].clone();

        let first_pre_element_wise = {
            let op = graph.nodes.nodes.node_weight(first_input).and_then(|node| {
                if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                    Some(op.clone())
                } else {
                    None
                }
            });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, first_input, Some(&op));
                first_input = functions.value;
                functions.functions
            } else {
                first_pre_element_wise
            }
        };

        let second_pre_element_wise = {
            let op = graph
                .nodes
                .nodes
                .node_weight(second_input)
                .and_then(|node| {
                    if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                        Some(op.clone())
                    } else {
                        None
                    }
                });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, second_input, Some(&op));
                second_input = functions.value;
                functions.functions
            } else {
                second_pre_element_wise
            }
        };

        let mut kernel = PairWiseOperation::new(function, first_input, second_input, &shape);
        let first_pre = first_pre_element_wise;
        let second_pre = second_pre_element_wise;
        kernel.set_pre_element_wise([first_pre, second_pre]);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Arc::new(kernel)
    }

    fn resolve_mat_mul(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &MatMulOperation,
    ) -> Arc<dyn Operation> {
        self.resolve_mat_mul_then(graph, operation, None)
    }

    fn resolve_mat_mul_then(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &MatMulOperation,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let mut first = operation.first;
        let first_shape = operation.first_shape.clone();
        let first_pre_element_wise = operation.pre_element_wise[0].clone();
        let mut second = operation.second;
        let second_shape = operation.second_shape.clone();
        let second_pre_element_wise = operation.pre_element_wise[1].clone();
        let parameters = operation.parameters.clone();

        let first_pre_element_wise = {
            let op = graph.nodes.nodes.node_weight(first).and_then(|node| {
                if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                    Some(op.clone())
                } else {
                    None
                }
            });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, first, Some(&op));
                first = functions.value;
                functions.functions
            } else {
                first_pre_element_wise
            }
        };

        let second_pre_element_wise = {
            let op = graph.nodes.nodes.node_weight(second).and_then(|node| {
                if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                    Some(op.clone())
                } else {
                    None
                }
            });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, second, Some(&op));
                second = functions.value;
                functions.functions
            } else {
                second_pre_element_wise
            }
        };

        let mut kernel = MatMulOperation::new_with_parameters(
            first_pre_element_wise.input_datatype(),
            first,
            second,
            &first_shape,
            &second_shape,
            parameters,
        );
        let first_pre = first_pre_element_wise;
        let second_pre = second_pre_element_wise;
        kernel.set_pre_element_wise([first_pre, second_pre]);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Arc::new(kernel)
    }

    fn resolve_q_mat_mul(&mut self, operation: &QMatMulOperation) -> Arc<dyn Operation> {
        let input = operation.input;
        let matrix = operation.matrix.clone();

        let kernel =
            QMatMulOperation::new(operation.input_datatype, &operation.in_shape, input, matrix);

        Arc::new(kernel)
    }

    fn resolve_dequantize(&mut self, operation: &DequantizeOperation) -> Arc<dyn Operation> {
        self.resolve_dequantize_then(operation, None)
    }

    fn resolve_dequantize_then(
        &mut self,
        operation: &DequantizeOperation,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let mut kernel = DequantizeOperation::new(operation.matrix.clone(), operation.datatype);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Arc::new(kernel)
    }

    fn resolve_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &ReduceOperation,
    ) -> Arc<dyn Operation> {
        self.resolve_reduce_then(graph, operation, None)
    }

    fn resolve_reduce_then(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &ReduceOperation,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let mut input_key = operation.value;

        let element_wise_before = {
            let op = graph
                .nodes
                .nodes
                .node_weight(operation.value)
                .and_then(|node| {
                    if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                        Some(op.clone())
                    } else {
                        None
                    }
                });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, operation.value, Some(&op));
                input_key = functions.value;
                functions.functions
            } else {
                ElementWiseFunctions::empty(operation.reduce_datatype())
            }
        };

        let mut kernel = ReduceOperation::new(
            input_key,
            operation.function.clone(),
            operation.axis,
            &operation.shape,
        );
        let element_wise_after = then
            .map(|op| op.functions)
            .unwrap_or_else(|| ElementWiseFunctions::empty(element_wise_before.out_datatype()));
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);

        Arc::new(kernel)
    }

    fn resolve_map_layout(
        &mut self,
        operation: &crate::map_layout::MapLayoutOperation,
    ) -> Arc<dyn Operation> {
        Arc::new(operation.clone())
    }

    fn resolve_resize(&mut self, operation: &crate::resize::ResizeOperation) -> Arc<dyn Operation> {
        Arc::new(operation.clone())
    }

    fn resolve_slice_assign(
        &mut self,
        operation: &crate::slice_assign::SliceAssignOperation,
    ) -> Arc<dyn Operation> {
        Arc::new(operation.clone())
    }

    fn resolve_index_select(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &IndexSelectOperation,
    ) -> Arc<dyn Operation> {
        self.resolve_index_select_then(graph, operation, None)
    }

    fn resolve_index_select_then(
        &mut self,
        graph: &mut ComputeGraphInner,
        operation: &IndexSelectOperation,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let dimension = operation.dimension;
        let mut input = operation.input;
        let input_datatype = operation.input_datatype();
        let indexes_datatype = operation.indexes_datatype();
        let mut indexes = operation.indexes;
        let value_shape = operation.value_shape.clone();
        let indexes_shape = operation.indexes_shape.clone();

        let mut input_pre_element_wise = {
            let op = graph.nodes.nodes.node_weight(input).and_then(|node| {
                if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                    Some(op.clone())
                } else {
                    None
                }
            });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, input, Some(&op));
                input = functions.value;
                functions
            } else {
                ElementWiseOperation::from_element_wise(
                    input,
                    ElementWiseFunctions::empty(input_datatype),
                    value_shape.as_ref(),
                )
            }
        };

        if let Some(then) = then {
            // pre and post elementwise are the same since the index select operation doesn't effect element wise values
            for function in then.functions.iter() {
                input_pre_element_wise.functions.push(function.clone());
            }
        }

        let indexes_pre_element_wise = {
            let op = graph.nodes.nodes.node_weight(indexes).and_then(|node| {
                if let ComputeGraphNodeVariant::ElementWise(op) = &node.variant {
                    Some(op.clone())
                } else {
                    None
                }
            });
            if let Some(op) = op {
                let functions = self.collect_element_wise_ops(graph, indexes, Some(&op));
                indexes = functions.value;
                functions
            } else {
                ElementWiseOperation::from_element_wise(
                    indexes,
                    ElementWiseFunctions::empty(indexes_datatype),
                    indexes_shape.as_ref(),
                )
            }
        };

        let mut kernel = IndexSelectOperation::new(
            input,
            indexes,
            input_pre_element_wise.input_datatype(),
            dimension,
            input_pre_element_wise.shape(),
            indexes_pre_element_wise.shape(),
        );
        kernel.set_pre_element_wise_input(input_pre_element_wise.functions);
        kernel.set_pre_element_wise_indexes(indexes_pre_element_wise.functions);

        Arc::new(kernel)
    }

    fn resolve_tensor(
        &mut self,
        graph: &mut ComputeGraphInner,
        key: NodeIndex,
        operation: &TensorData,
    ) {
        graph.set_cached_result(key, operation.clone());
        // Mark this node as resolved
        self.resolved_set.insert(key);
    }

    fn resolve_custom(
        &mut self,
        operation: &Arc<dyn Operation + Send + Sync>,
    ) -> Arc<dyn Operation + Send + Sync> {
        Arc::clone(operation)
    }
}
