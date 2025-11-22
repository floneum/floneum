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

use super::{
    NodeIndex, ComputeGraphInner, ComputeGraphNodeVariant,
    queue::ComputeQueue,
};

pub(crate) struct Resolver<'a> {
    graph: &'a mut ComputeGraphInner,
    command_encoder: &'a mut CommandEncoder,
    queued_operations: Vec<(NodeIndex, Arc<dyn Operation>)>,
    target: NodeIndex,
    queue: ComputeQueue,
    resolved_set: FxHashSet<NodeIndex>,
}

impl<'a> Resolver<'a> {
    pub(crate) fn new(
        graph: &'a mut ComputeGraphInner,
        target: NodeIndex,
        command_encoder: &'a mut CommandEncoder,
    ) -> Self {
        let resolved_set = graph.cached_results.keys().cloned().collect();
        Self {
            graph,
            command_encoder,
            target,
            queued_operations: Vec::new(),
            queue: Default::default(),
            resolved_set,
        }
    }

    pub(crate) fn run(&mut self) -> TensorData {
        let limits = self.graph.device.limits();
        self.queue_operations();
        let queued_operations = std::mem::take(&mut self.queued_operations);

        // Find runs of compatible dispatch shapes
        let mut current_constraints = WorkgroupShapeConstraints::new();
        let mut pending_operations = Vec::new();
        let mut inputs = Vec::new();
        let mut all_input_values = Vec::new();
        let mut kernel = GenericKernel::new();

        for (node, operation) in queued_operations {
            let new_inputs = operation.inputs(self.graph);
            let constraint = operation.workgroup_shape_constraints(&self.graph.device);
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
            let map_layout = if let Some(node_data) = self.graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::MapLayout(map_layout) => Some(map_layout.clone()),
                    ComputeGraphNodeVariant::Resize(resize) => resize.lower(self.graph),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(map_layout) = map_layout {
                let result = map_layout.run(self.graph);
                // Cache the result
                self.graph.cached_results.insert(node, result);
            } else {
                self.push_operation(
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
                &mut kernel,
                &pending_operations,
                &inputs,
                &all_input_values,
                old_best,
            );
        }

        self.graph.cached_results[&self.target].clone()
    }

    fn should_extend_kernel(&mut self, _: Vec<MirValue>, _: &[Vec<MirValue>]) -> bool {
        // TODO: Restore with better testing. This passes all tests in fusor, but breaks rbert and rwhisper
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn push_operation(
        &mut self,
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
        let result = operation.output(self.graph, &new_inputs);
        let MirValue::Tensor(resolved) = result else {
            panic!("Kernel input value is not a tensor");
        };
        // Cache the result
        self.graph.cached_results.insert(key, resolved);
        inputs.push(new_inputs);
        queued_operations.push((key, operation));
    }

    fn flush_operations(
        &mut self,
        mut kernel: &mut GenericKernel,
        queued_operations: &[(NodeIndex, Arc<dyn Operation>)],
        inputs: &[Vec<MirValue>],
        all_input_values: &[KernelInputValue],
        workgroup_shape: workgroup_shape::WorkgroupShape,
    ) {
        let mut max_dispatch_size = [0; 3];
        for ((key, operation), inputs) in queued_operations.iter().zip(inputs) {
            // Map layout isn't really a kernel. Skip it
            if let Some(node) = self.graph.nodes.nodes.node_weight(*key) {
                if matches!(node.variant, ComputeGraphNodeVariant::MapLayout(_)) {
                    continue;
                }
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
            operation.build_kernel(self.graph, &workgroup_shape, inputs, kernel);
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
            self.graph.visit_dependencies(*key, &mut |dependent_key| {
                dependencies.push(dependent_key);
            });
            for dependency in dependencies {
                self.graph.check_life(dependency);
            }
        }
        kernel.set_workgroup_size(workgroup_shape);
        kernel.run(
            &self.graph.device,
            all_input_values,
            self.command_encoder,
            max_dispatch_size,
        );
    }

    pub(crate) fn queue_operations(&mut self) {
        self.queue.push_back(self.target);

        while let Some(node) = self.queue.pop_front() {
            if self.resolved_set.contains(&node) {
                continue;
            }

            let node_data = self.graph.nodes.nodes.node_weight(node).expect("Node not found in graph");
            let resolved = match &node_data.variant {
                ComputeGraphNodeVariant::ElementWise(_) => {
                    self.resolve_element_wise(node)
                }
                ComputeGraphNodeVariant::PairWise(_) => {
                    self.resolve_pair_wise(node)
                }
                ComputeGraphNodeVariant::MatMul(_) => {
                    self.resolve_mat_mul(node)
                }
                ComputeGraphNodeVariant::Reduce(_) => {
                    self.resolve_reduce(node)
                }
                ComputeGraphNodeVariant::Tensor(_) => {
                    self.resolve_tensor(node);
                    continue;
                }
                ComputeGraphNodeVariant::MapLayout(_) => {
                    self.resolve_map_layout(node)
                }
                ComputeGraphNodeVariant::Resize(_) => {
                    self.resolve_resize(node)
                }
                ComputeGraphNodeVariant::SliceAssign(_) => {
                    self.resolve_slice_assign(node)
                }
                ComputeGraphNodeVariant::IndexSelect(_) => {
                    self.resolve_index_select(node)
                }
                ComputeGraphNodeVariant::QMatMul(_) => {
                    self.resolve_q_mat_mul(node)
                }
                ComputeGraphNodeVariant::Dequantize(_) => {
                    self.resolve_dequantize(node)
                }
                ComputeGraphNodeVariant::Custom(_) => {
                    self.resolve_custom(node)
                }
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

    fn collect_element_wise_ops(&mut self, key: NodeIndex) -> ElementWiseOperation {
        let mut functions = Vec::new();
        let mut current_key = key;
        let mut ty = DataTypeEnum::F32;
        let mut shape = Box::new([]) as Box<[usize]>;

        while let Some(node) = self.graph.nodes.nodes.node_weight(current_key) {
            if let ComputeGraphNodeVariant::ElementWise(operation) = &node.variant {
                // If the result is already cached, stop collecting element wise ops
                if let Some(cached) = self.graph.cached_results.get(&current_key) {
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

    fn resolve_element_wise(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        // First collect all element wise ops in this chain
        let functions = self.collect_element_wise_ops(key);
        let input = functions.value;
        let input_cached = self.resolved_set.contains(&input);

        if !input_cached {
            if let Some(node) = self.graph.nodes.nodes.node_weight(input) {
                match &node.variant {
                    // Merge into the output of the reduce kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::Reduce(_) => {
                        return self.resolve_reduce_then(input, Some(functions));
                    }
                    // Merge into the output of the pair wise kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::PairWise(_) => {
                        return self.resolve_pair_wise_then(input, Some(functions));
                    }
                    // Merge into the output of the mat mul kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::MatMul(_) => {
                        return self.resolve_mat_mul_then(input, Some(functions));
                    }
                    // Merge into the output of the dequantize kernel if possible and it isn't already cached
                    ComputeGraphNodeVariant::Dequantize(_) => {
                        return self.resolve_dequantize_then(input, Some(functions));
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

    fn resolve_pair_wise(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        self.resolve_pair_wise_then(key, None)
    }

    fn resolve_pair_wise_then(
        &mut self,
        key: NodeIndex,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::PairWise(op) => op,
            _ => panic!("Expected PairWise node"),
        };
        let function = operation.function.clone();
        let shape: Box<[usize]> = operation.shape().into();

        let mut first_input = operation.first;
        let first_pre_element_wise = operation.pre_element_wise[0].clone();
        let mut second_input = operation.second;
        let second_pre_element_wise = operation.pre_element_wise[1].clone();

        let first_pre_element_wise = if let Some(node) = self.graph.nodes.nodes.node_weight(first_input) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(first_input);
                first_input = functions.value;
                functions.functions
            } else {
                first_pre_element_wise
            }
        } else {
            first_pre_element_wise
        };

        let second_pre_element_wise = if let Some(node) = self.graph.nodes.nodes.node_weight(second_input) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(second_input);
                second_input = functions.value;
                functions.functions
            } else {
                second_pre_element_wise
            }
        } else {
            second_pre_element_wise
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

    fn resolve_mat_mul(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        self.resolve_mat_mul_then(key, None)
    }

    fn resolve_mat_mul_then(
        &mut self,
        key: NodeIndex,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::MatMul(op) => op,
            _ => panic!("Expected MatMul node"),
        };
        let mut first = operation.first;
        let first_shape = operation.first_shape.clone();
        let first_pre_element_wise = operation.pre_element_wise[0].clone();
        let mut second = operation.second;
        let second_shape = operation.second_shape.clone();
        let second_pre_element_wise = operation.pre_element_wise[1].clone();
        let parameters = operation.parameters.clone();

        let first_pre_element_wise = if let Some(node) = self.graph.nodes.nodes.node_weight(first) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(first);
                first = functions.value;
                functions.functions
            } else {
                first_pre_element_wise
            }
        } else {
            first_pre_element_wise
        };

        let second_pre_element_wise = if let Some(node) = self.graph.nodes.nodes.node_weight(second) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(second);
                second = functions.value;
                functions.functions
            } else {
                second_pre_element_wise
            }
        } else {
            second_pre_element_wise
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

    fn resolve_q_mat_mul(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::QMatMul(op) => op,
            _ => panic!("Expected QMatMul node"),
        };
        let input = operation.input;
        let matrix = operation.matrix.clone();

        let kernel =
            QMatMulOperation::new(operation.input_datatype, &operation.in_shape, input, matrix);

        Arc::new(kernel)
    }

    fn resolve_dequantize(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        self.resolve_dequantize_then(key, None)
    }

    fn resolve_dequantize_then(
        &mut self,
        key: NodeIndex,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Dequantize(op) => op,
            _ => panic!("Expected Dequantize node"),
        };

        let mut kernel = DequantizeOperation::new(operation.matrix.clone(), operation.datatype);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Arc::new(kernel)
    }

    fn resolve_reduce(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        self.resolve_reduce_then(key, None)
    }

    fn resolve_reduce_then(
        &mut self,
        key: NodeIndex,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Reduce(op) => op.clone(),
            _ => panic!("Expected Reduce node"),
        };
        let mut input_key = operation.value;

        let element_wise_before = if let Some(node) = self.graph.nodes.nodes.node_weight(operation.value) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(operation.value);
                input_key = functions.value;
                functions.functions
            } else {
                ElementWiseFunctions::empty(operation.reduce_datatype())
            }
        } else {
            ElementWiseFunctions::empty(operation.reduce_datatype())
        };

        let mut kernel = ReduceOperation::new(
            input_key,
            operation.function,
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

    fn resolve_map_layout(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let kernel = match &node.variant {
            ComputeGraphNodeVariant::MapLayout(op) => op,
            _ => panic!("Expected MapLayout node"),
        };

        Arc::new(kernel.clone())
    }

    fn resolve_resize(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let kernel = match &node.variant {
            ComputeGraphNodeVariant::Resize(op) => op,
            _ => panic!("Expected Resize node"),
        };

        Arc::new(kernel.clone())
    }

    fn resolve_slice_assign(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let kernel = match &node.variant {
            ComputeGraphNodeVariant::SliceAssign(op) => op,
            _ => panic!("Expected SliceAssign node"),
        };

        Arc::new(kernel.clone())
    }

    fn resolve_index_select(&mut self, key: NodeIndex) -> Arc<dyn Operation> {
        self.resolve_index_select_then(key, None)
    }

    fn resolve_index_select_then(
        &mut self,
        key: NodeIndex,
        then: Option<ElementWiseOperation>,
    ) -> Arc<dyn Operation> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::IndexSelect(op) => op,
            _ => panic!("Expected IndexSelect node"),
        };

        let dimension = operation.dimension;
        let mut input = operation.input;
        let input_datatype = operation.input_datatype();
        let indexes_datatype = operation.indexes_datatype();
        let mut indexes = operation.indexes;
        let value_shape = operation.value_shape.clone();
        let indexes_shape = operation.indexes_shape.clone();

        let mut input_pre_element_wise = if let Some(node) = self.graph.nodes.nodes.node_weight(input) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(input);
                input = functions.value;
                functions
            } else {
                ElementWiseOperation::from_element_wise(
                    input,
                    ElementWiseFunctions::empty(input_datatype),
                    value_shape.as_ref(),
                )
            }
        } else {
            ElementWiseOperation::from_element_wise(
                input,
                ElementWiseFunctions::empty(input_datatype),
                value_shape.as_ref(),
            )
        };

        if let Some(then) = then {
            // pre and post elementwise are the same since the index select operation doesn't effect element wise values
            for function in then.functions.iter() {
                input_pre_element_wise.functions.push(function.clone());
            }
        }

        let indexes_pre_element_wise = if let Some(node) = self.graph.nodes.nodes.node_weight(indexes) {
            if let ComputeGraphNodeVariant::ElementWise(_) = &node.variant {
                let functions = self.collect_element_wise_ops(indexes);
                indexes = functions.value;
                functions
            } else {
                ElementWiseOperation::from_element_wise(
                    indexes,
                    ElementWiseFunctions::empty(indexes_datatype),
                    indexes_shape.as_ref(),
                )
            }
        } else {
            ElementWiseOperation::from_element_wise(
                indexes,
                ElementWiseFunctions::empty(indexes_datatype),
                indexes_shape.as_ref(),
            )
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

    fn resolve_tensor(&mut self, key: NodeIndex) {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        let tensor = match &node.variant {
            ComputeGraphNodeVariant::Tensor(data) => data.clone(),
            _ => panic!("Expected Tensor node"),
        };

        self.graph.cached_results.insert(key, tensor);
        // Mark this node as resolved
        self.resolved_set.insert(key);
    }

    fn resolve_custom(&mut self, key: NodeIndex) -> Arc<dyn Operation + Send + Sync> {
        let node = self.graph.nodes.nodes.node_weight(key).expect("Node not found");
        match &node.variant {
            ComputeGraphNodeVariant::Custom(op) => Arc::clone(op),
            _ => panic!("Expected Custom node"),
        }
    }
}