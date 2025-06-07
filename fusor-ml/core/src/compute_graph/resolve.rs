use rustc_hash::FxHashSet;
use wgpu::CommandEncoder;

use crate::{
    DataTypeEnum, ElementWiseFunctions, ElementWiseOperation, MatMulOperation, PairWiseOperation,
    ReduceOperation,
    dequantize::DequantizeOperation,
    index_select::IndexSelectOperation,
    mir::{
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{self, WorkgroupShapeConstraints},
    },
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};

use super::{
    AnyComputeKey, ComputeGraphInner, DequantizeComputeKey, ElementWiseComputeNodeKey,
    IndexSelectComputeNodeKey, MapLayoutComputeNodeKey, MatMulComputeNodeKey,
    PairWiseComputeNodeKey, QMatMulComputeNodeKey, ReduceComputeNodeKey, ResizeComputeNodeKey,
    SliceAssignComputeNodeKey, TensorComputeNodeKey, dependency_map::visit_dependencies,
    queue::ComputeQueue,
};

pub(crate) struct Resolver<'a> {
    graph: &'a mut ComputeGraphInner,
    command_encoder: &'a mut CommandEncoder,
    queued_operations: Vec<(AnyComputeKey, Box<dyn Operation>)>,
    target: AnyComputeKey,
    queue: ComputeQueue,
    resolved_set: FxHashSet<AnyComputeKey>,
}

impl<'a> Resolver<'a> {
    pub(crate) fn new(
        graph: &'a mut ComputeGraphInner,
        target: AnyComputeKey,
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
        self.queue_operations();
        let queued_operations = std::mem::take(&mut self.queued_operations);

        // Find runs of compatible dispatch shapes
        let mut current_constraints = WorkgroupShapeConstraints::new();
        let mut pending_operations = Vec::new();

        for (node, operation) in queued_operations {
            let constraint = operation.workgroup_shape_constraints(&self.graph.device);
            let mut new_merged = current_constraints.clone();
            new_merged.merge(&constraint);
            let new_best = new_merged.solve();
            let old_best = current_constraints.solve().unwrap();
            current_constraints = new_merged;
            if new_best.is_none() {
                self.flush_operations(&mut pending_operations, old_best);
                current_constraints.clear();
            }
            pending_operations.push((node, operation));
        }

        if !pending_operations.is_empty() {
            let old_best = current_constraints.solve().unwrap();
            self.flush_operations(&mut pending_operations, old_best);
        }

        self.graph.cached_results[&self.target].clone()
    }

    fn flush_operations(
        &mut self,
        queued_operations: &mut Vec<(AnyComputeKey, Box<dyn Operation>)>,
        workgroup_shape: workgroup_shape::WorkgroupShape,
    ) {
        let mut kernel = GenericKernel::new();
        let mut all_inputs = Vec::new();
        let mut max_dispatch_size = [0; 3];
        for (key, operation) in queued_operations.drain(..) {
            // Map layout isn't really a kernel. Resolve it immediately
            if let AnyComputeKey::MapLayout(key) = key {
                let map_layout = self.graph.nodes.map_layout[&key].clone();
                let result = map_layout.run(&mut self.graph);
                // Cache the result
                self.graph.cached_results.insert(key.into(), result);
                continue;
            }

            let inputs = operation.inputs(&self.graph);
            for input in &inputs {
                input.visit_input_values(|value| {
                    if let Some(index) = all_inputs.iter().position(|x| *x == value) {
                        kernel.pre_register_binding(index as _);
                    } else {
                        kernel.pre_register_binding(all_inputs.len() as _);
                        all_inputs.push(value.clone());
                    }
                });
            }
            let dispatch_size = operation.dispatch_size(&workgroup_shape, &inputs);
            for (new, max) in dispatch_size.iter().zip(max_dispatch_size.iter_mut()) {
                *max = (*max).max(*new);
            }
            kernel.push_body("{");
            operation.build_kernel(&self.graph, &workgroup_shape, &inputs, &mut kernel);
            kernel.push_body("}");
            kernel.push_body("storageBarrier();");
            let result = operation.output(&self.graph, &inputs);
            let MirValue::Tensor(resolved) = result else {
                panic!("Kernel input value is not a tensor");
            };
            // Cache the result
            self.graph.cached_results.insert(key, resolved);
            // Check if that makes any of this nodes dependents dead
            let mut dependencies = Vec::new();
            visit_dependencies(&self.graph.nodes, key, |dependent_key| {
                dependencies.push(dependent_key);
            });
            for dependency in dependencies {
                self.graph.check_life(dependency);
            }
        }
        kernel.set_workgroup_size(workgroup_shape);
        kernel.run(
            &self.graph.device,
            all_inputs,
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

            let resolved = match node {
                AnyComputeKey::ElementWise(element_wise_compute_node_key) => {
                    self.resolve_element_wise(element_wise_compute_node_key)
                }
                AnyComputeKey::PairWise(pair_wise_compute_node_key) => {
                    self.resolve_pair_wise(pair_wise_compute_node_key)
                }
                AnyComputeKey::MatMul(mat_mul_compute_node_key) => {
                    self.resolve_mat_mul(mat_mul_compute_node_key)
                }
                AnyComputeKey::Reduce(reduce_compute_node_key) => {
                    self.resolve_reduce(reduce_compute_node_key)
                }
                AnyComputeKey::Tensor(tensor_compute_node_key) => {
                    self.resolve_tensor(tensor_compute_node_key);
                    continue;
                }
                AnyComputeKey::MapLayout(slice_compute_node_key) => {
                    self.resolve_map_layout(slice_compute_node_key)
                }
                AnyComputeKey::Resize(resize_compute_node_key) => {
                    self.resolve_resize(resize_compute_node_key)
                }
                AnyComputeKey::SliceAssign(slice_assign_compute_node_key) => {
                    self.resolve_slice_assign(slice_assign_compute_node_key)
                }
                AnyComputeKey::IndexSelect(index_select_compute_node_key) => {
                    self.resolve_index_select(index_select_compute_node_key)
                }
                AnyComputeKey::QMatMul(q_mat_mul_compute_node_key) => {
                    self.resolve_q_mat_mul(q_mat_mul_compute_node_key)
                }
                AnyComputeKey::Dequantize(dequantize_compute_node_key) => {
                    self.resolve_dequantize(dequantize_compute_node_key)
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

    fn collect_element_wise_ops(&mut self, key: ElementWiseComputeNodeKey) -> ElementWiseOperation {
        let mut functions = Vec::new();
        let mut current_key = AnyComputeKey::ElementWise(key);
        let mut ty = DataTypeEnum::F32;
        let mut shape = Box::new([]) as Box<[usize]>;
        while let AnyComputeKey::ElementWise(key) = current_key {
            // If the result is already cached, stop collecting element wise ops
            if let Some(cached) = self.graph.cached_results.get(&current_key) {
                ty = cached.datatype();
                shape = cached.layout().shape().into();
                break;
            }
            let operation = &self.graph.nodes.element_wise[&key];
            ty = operation.input_datatype();
            shape = operation.shape().into();
            functions.extend(operation.functions.iter().cloned());
            current_key = operation.value;
        }
        let functions = ElementWiseFunctions::new(functions, ty);
        ElementWiseOperation::from_element_wise(current_key, functions, shape)
    }

    fn resolve_element_wise(&mut self, key: ElementWiseComputeNodeKey) -> Box<dyn Operation> {
        // First collect all element wise ops in this chain
        let functions = self.collect_element_wise_ops(key);
        let input = functions.value;
        let input_cached = self.resolved_set.contains(&input);

        if !input_cached {
            // Merge into the output of the reduce kernel if possible and it isn't already cached
            if let AnyComputeKey::Reduce(key) = input {
                return self.resolve_reduce_then(key, Some(functions));
            }
            // Merge into the output of the pair wise kernel if possible and it isn't already cached
            if let AnyComputeKey::PairWise(key) = input {
                return self.resolve_pair_wise_then(key, Some(functions));
            }
            // Merge into the output of the mat mul kernel if possible and it isn't already cached
            if let AnyComputeKey::MatMul(key) = input {
                return self.resolve_mat_mul_then(key, Some(functions));
            }
            // Merge into the output of the dequantize kernel if possible and it isn't already cached
            if let AnyComputeKey::Dequantize(key) = input {
                return self.resolve_dequantize_then(key, Some(functions));
            }
        }
        let shape: Box<[_]> = functions.shape().into();
        // Otherwise, just run the element wise kernel
        let kernel =
            ElementWiseOperation::from_element_wise(input.into(), functions.functions, shape);

        Box::new(kernel)
    }

    fn resolve_pair_wise(&mut self, key: PairWiseComputeNodeKey) -> Box<dyn Operation> {
        self.resolve_pair_wise_then(key, None)
    }

    fn resolve_pair_wise_then(
        &mut self,
        key: PairWiseComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Box<dyn Operation> {
        let operation = &self.graph.nodes.pair_wise[&key];
        let function = operation.function.clone();
        let rank = operation.rank();

        let mut first_input = operation.first;
        let first_pre_element_wise = operation.pre_element_wise[0].clone();
        let mut second_input = operation.second;
        let second_pre_element_wise = operation.pre_element_wise[1].clone();
        let first_pre_element_wise = if let AnyComputeKey::ElementWise(key) = first_input {
            let functions = self.collect_element_wise_ops(key);
            first_input = functions.value;
            functions.functions
        } else {
            first_pre_element_wise
        };
        let second_pre_element_wise = if let AnyComputeKey::ElementWise(key) = second_input {
            let functions = self.collect_element_wise_ops(key);
            second_input = functions.value;
            functions.functions
        } else {
            second_pre_element_wise
        };

        let mut kernel = PairWiseOperation::new(function, first_input, second_input, rank);
        let first_pre = first_pre_element_wise;
        let second_pre = second_pre_element_wise;
        kernel.set_pre_element_wise([first_pre, second_pre]);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Box::new(kernel)
    }

    fn resolve_mat_mul(&mut self, key: MatMulComputeNodeKey) -> Box<dyn Operation> {
        self.resolve_mat_mul_then(key, None)
    }

    fn resolve_mat_mul_then(
        &mut self,
        key: MatMulComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Box<dyn Operation> {
        let operation = &self.graph.nodes.mat_mul[&key];
        let mut first = operation.first;
        let first_shape = operation.first_shape.clone();
        let first_pre_element_wise = operation.pre_element_wise[0].clone();
        let mut second = operation.second;
        let second_shape = operation.second_shape.clone();
        let second_pre_element_wise = operation.pre_element_wise[1].clone();

        let first_pre_element_wise = if let AnyComputeKey::ElementWise(key) = first {
            let functions = self.collect_element_wise_ops(key);
            first = functions.value;
            functions.functions
        } else {
            first_pre_element_wise
        };
        let second_pre_element_wise = if let AnyComputeKey::ElementWise(key) = second {
            let functions = self.collect_element_wise_ops(key);
            second = functions.value;
            functions.functions
        } else {
            second_pre_element_wise
        };

        let mut kernel = MatMulOperation::new(
            first_pre_element_wise.input_datatype(),
            first,
            second,
            &first_shape,
            &second_shape,
        );
        let first_pre = first_pre_element_wise;
        let second_pre = second_pre_element_wise;
        kernel.set_pre_element_wise([first_pre, second_pre]);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Box::new(kernel)
    }

    fn resolve_q_mat_mul(&mut self, key: QMatMulComputeNodeKey) -> Box<dyn Operation> {
        let operation = &self.graph.nodes.q_mat_mul[&key];
        let input = operation.input;
        let matrix = operation.matrix.clone();

        let kernel =
            QMatMulOperation::new(operation.input_datatype, &operation.in_shape, input, matrix);

        Box::new(kernel)
    }

    fn resolve_dequantize(&mut self, key: DequantizeComputeKey) -> Box<dyn Operation> {
        self.resolve_dequantize_then(key, None)
    }

    fn resolve_dequantize_then(
        &mut self,
        key: DequantizeComputeKey,
        then: Option<ElementWiseOperation>,
    ) -> Box<dyn Operation> {
        let operation = &self.graph.nodes.dequantize[&key];

        let mut kernel = DequantizeOperation::new(operation.matrix.clone(), operation.datatype);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Box::new(kernel)
    }

    fn resolve_reduce(&mut self, key: ReduceComputeNodeKey) -> Box<dyn Operation> {
        self.resolve_reduce_then(key, None)
    }

    fn resolve_reduce_then(
        &mut self,
        key: ReduceComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Box<dyn Operation> {
        let operation = self.graph.nodes.reduce[&key].clone();
        let mut input_key = operation.value;

        let element_wise_before = if let AnyComputeKey::ElementWise(key) = operation.value {
            let functions = self.collect_element_wise_ops(key);
            input_key = functions.value;
            functions.functions
        } else {
            ElementWiseFunctions::empty(operation.reduce_datatype())
        };

        let mut kernel = ReduceOperation::new(
            input_key,
            operation.function,
            operation.axis,
            operation.rank,
        );
        let element_wise_before = element_wise_before;
        let element_wise_after = then
            .map(|op| op.functions)
            .unwrap_or_else(|| ElementWiseFunctions::empty(element_wise_before.out_datatype()));
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);

        Box::new(kernel)
    }

    fn resolve_map_layout(&mut self, key: MapLayoutComputeNodeKey) -> Box<dyn Operation> {
        let kernel = self.graph.nodes.map_layout.get(&key).unwrap();

        Box::new(kernel.clone())
    }

    fn resolve_resize(&mut self, key: ResizeComputeNodeKey) -> Box<dyn Operation> {
        let kernel = self.graph.nodes.resize.get(&key).unwrap();

        Box::new(kernel.clone())
    }

    fn resolve_slice_assign(&mut self, key: SliceAssignComputeNodeKey) -> Box<dyn Operation> {
        let kernel = self.graph.nodes.slice_assign.get(&key).unwrap();

        Box::new(kernel.clone())
    }

    fn resolve_index_select(&mut self, key: IndexSelectComputeNodeKey) -> Box<dyn Operation> {
        self.resolve_index_select_then(key, None)
    }

    fn resolve_index_select_then(
        &mut self,
        key: IndexSelectComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Box<dyn Operation> {
        let operation = &self.graph.nodes.index_select[&key];

        let dimension = operation.dimension;
        let mut input = operation.input;
        let input_datatype = operation.input_datatype();
        let indexes_datatype = operation.indexes_datatype();
        let mut indexes = operation.indexes;
        let value_shape = operation.value_shape.clone();
        let indexes_shape = operation.indexes_shape.clone();
        let mut input_pre_element_wise = if let AnyComputeKey::ElementWise(key) = input {
            let functions = self.collect_element_wise_ops(key);
            input = functions.value;
            functions
        } else {
            ElementWiseOperation::from_element_wise(
                input,
                ElementWiseFunctions::empty(input_datatype),
                value_shape,
            )
        };
        if let Some(then) = then {
            // pre and post elementwise are the same since the index select operation doesn't effect element wise values
            for function in then.functions.iter() {
                input_pre_element_wise.functions.push(function.clone());
            }
        }
        let indexes_pre_element_wise = if let AnyComputeKey::ElementWise(key) = indexes {
            let functions = self.collect_element_wise_ops(key);
            indexes = functions.value;
            functions
        } else {
            ElementWiseOperation::from_element_wise(
                indexes,
                ElementWiseFunctions::empty(indexes_datatype),
                indexes_shape,
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

        Box::new(kernel)
    }

    fn resolve_tensor(&mut self, key: TensorComputeNodeKey) {
        let tensor = self.graph.nodes.tensor.get(&key).unwrap().clone();

        self.graph.cached_results.insert(key.into(), tensor);
        // Mark this node as resolved
        self.resolved_set.insert(AnyComputeKey::Tensor(key));
    }
}
