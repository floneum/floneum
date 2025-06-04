use std::collections::HashSet;

use rustc_hash::FxHashSet;
use wgpu::CommandEncoder;

use crate::{
    DataTypeEnum, ElementWiseFunction, ElementWiseFunctions, ElementWiseOperation, MatMulOperation,
    PairWiseOperation, ReduceOperation,
    dequantize::DequantizeOperation,
    element_wise,
    index_select::IndexSelectOperation,
    mir::{inputs::KernelInputValue, operation::Operation},
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
    visit_tiled::MaybeQData,
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
        Self {
            graph,
            command_encoder,
            target,
            queued_operations: Vec::new(),
            queue: Default::default(),
            resolved_set: Default::default(),
        }
    }

    fn is_resolved_or_qmatrix(&self, key: &AnyComputeKey) -> bool {
        self.resolved_set.contains(key) || matches!(key, AnyComputeKey::Dequantize(_))
    }

    pub(crate) fn run(&mut self) -> TensorData {
        self.queue_operations();
        let queued_operations = std::mem::take(&mut self.queued_operations);

        for (node, operation) in queued_operations {
            let result = operation.run(&self.graph, &mut *self.command_encoder);

            let KernelInputValue::Tensor(resolved) = result else {
                panic!("Kernel input value is not a tensor");
            };

            // Cache the result
            self.graph.cached_results.insert(node, resolved.clone());
            // Check if that makes any of this nodes dependents dead
            let mut dependencies = Vec::new();
            visit_dependencies(&self.graph.nodes, node, |dependent_key| {
                dependencies.push(dependent_key);
            });
            for dependency in dependencies {
                self.graph.check_life(dependency);
            }
        }

        self.graph.cached_results[&self.target].clone()
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
                AnyComputeKey::Tensor(_) => {
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
            let Some(resolved) = resolved else {
                // If there are dependencies that are not resolved, push them to the queue then
                // revisit this node
                self.queue.push_back(node);
                continue;
            };
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
            let operation = &self.graph.nodes.element_wise[&key];
            ty = operation.input_datatype();
            shape = operation.shape().into();
            // If the result is already cached, stop collecting element wise ops
            if self.resolved_set.contains(&current_key) {
                break;
            }
            functions.extend(operation.functions.iter().cloned());
            current_key = operation.value;
        }
        let functions = ElementWiseFunctions::new(functions, ty);
        ElementWiseOperation::from_element_wise(current_key, functions, shape)
    }

    fn resolve_element_wise(
        &mut self,
        key: ElementWiseComputeNodeKey,
    ) -> Option<Box<dyn Operation>> {
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
            // If the input is not cached, we need to wait for it to be resolved
            self.queue.push_back(input);
            return None;
        }
        let shape: Box<[_]> = functions.shape().into();
        // Otherwise, just run the element wise kernel
        let kernel =
            ElementWiseOperation::from_element_wise(input.into(), functions.functions, shape);

        Some(Box::new(kernel))
    }

    fn resolve_pair_wise(&mut self, key: PairWiseComputeNodeKey) -> Option<Box<dyn Operation>> {
        self.resolve_pair_wise_then(key, None)
    }

    fn resolve_pair_wise_then(
        &mut self,
        key: PairWiseComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Option<Box<dyn Operation>> {
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

        if !self.is_resolved_or_qmatrix(&first_input) {
            self.queue.push_back(first_input);
            return None;
        }
        if !self.is_resolved_or_qmatrix(&second_input) {
            self.queue.push_back(second_input);
            return None;
        }
        let mut kernel =
            PairWiseOperation::new(function, first_input, second_input, rank);
        let first_pre = first_pre_element_wise;
        let second_pre = second_pre_element_wise;
        kernel.set_pre_element_wise([first_pre, second_pre]);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Some(Box::new(kernel))
    }

    fn resolve_mat_mul(&mut self, key: MatMulComputeNodeKey) -> Option<Box<dyn Operation>> {
        self.resolve_mat_mul_then(key, None)
    }

    fn resolve_mat_mul_then(
        &mut self,
        key: MatMulComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Option<Box<dyn Operation>> {
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

        if !self.resolved_set.contains(&first) {
            self.queue.push_back(first);
            return None;
        };
        if !self.resolved_set.contains(&second) {
            self.queue.push_back(second);
            return None;
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

        Some(Box::new(kernel))
    }

    fn resolve_q_mat_mul(&mut self, key: QMatMulComputeNodeKey) -> Option<Box<dyn Operation>> {
        let operation = &self.graph.nodes.q_mat_mul[&key];
        let input = operation.input;
        let matrix = operation.matrix.clone();

        if !self.resolved_set.contains(&input) {
            self.queue.push_back(input);
            return None;
        };
        let kernel =
            QMatMulOperation::new(operation.input_datatype, &operation.in_shape, input, matrix);

        Some(Box::new(kernel))
    }

    fn resolve_dequantize(&mut self, key: DequantizeComputeKey) -> Option<Box<dyn Operation>> {
        self.resolve_dequantize_then(key, None)
    }

    fn resolve_dequantize_then(
        &mut self,
        key: DequantizeComputeKey,
        then: Option<ElementWiseOperation>,
    ) -> Option<Box<dyn Operation>> {
        let operation = &self.graph.nodes.dequantize[&key];

        let mut kernel = DequantizeOperation::new(operation.matrix.clone(), operation.datatype);
        if let Some(then) = then {
            kernel.set_post_element_wise(then.functions);
        }

        Some(Box::new(kernel))
    }

    fn resolve_reduce(&mut self, key: ReduceComputeNodeKey) -> Option<Box<dyn Operation>> {
        self.resolve_reduce_then(key, None)
    }

    fn resolve_reduce_then(
        &mut self,
        key: ReduceComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Option<Box<dyn Operation>> {
        let operation = self.graph.nodes.reduce[&key].clone();
        let mut input_key = operation.value;

        let element_wise_before = if let AnyComputeKey::ElementWise(key) = operation.value {
            let functions = self.collect_element_wise_ops(key);
            input_key = functions.value;
            functions.functions
        } else {
            ElementWiseFunctions::empty(operation.reduce_datatype())
        };

        if !self.resolved_set.contains(&input_key) {
            self.queue.push_back(input_key);
            return None;
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

        Some(Box::new(kernel))
    }

    fn resolve_map_layout(&mut self, key: MapLayoutComputeNodeKey) -> Option<Box<dyn Operation>> {
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();
        if !self.resolved_set.contains(&operation.input) {
            self.queue.push_back(operation.input);
            return None;
        }
        let kernel = self.graph.nodes.map_layout.get(&key).unwrap();

        Some(Box::new(kernel.clone()))
    }

    fn resolve_resize(&mut self, key: ResizeComputeNodeKey) -> Option<Box<dyn Operation>> {
        let kernel = self.graph.nodes.resize.get(&key).unwrap();
        let input = kernel.input;
        if !self.resolved_set.contains(&input) {
            self.queue.push_back(input);
            return None;
        }

        Some(Box::new(kernel.clone()))
    }

    fn resolve_slice_assign(
        &mut self,
        key: SliceAssignComputeNodeKey,
    ) -> Option<Box<dyn Operation>> {
        let kernel = self.graph.nodes.slice_assign.get(&key).unwrap();
        let input = kernel.input;
        let value = kernel.value;
        if !self.resolved_set.contains(&input) {
            self.queue.push_back(input);
            return None;
        }
        if !self.resolved_set.contains(&value) {
            self.queue.push_back(value);
            return None;
        }

        Some(Box::new(kernel.clone()))
    }

    fn resolve_index_select(
        &mut self,
        key: IndexSelectComputeNodeKey,
    ) -> Option<Box<dyn Operation>> {
        self.resolve_index_select_then(key, None)
    }

    fn resolve_index_select_then(
        &mut self,
        key: IndexSelectComputeNodeKey,
        then: Option<ElementWiseOperation>,
    ) -> Option<Box<dyn Operation>> {
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

        if !self.resolved_set.contains(&input) {
            self.queue.push_back(input);
            return None;
        };
        if !self.resolved_set.contains(&indexes) {
            self.queue.push_back(indexes);
            return None;
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

        Some(Box::new(kernel))
    }

    fn resolve_tensor(&mut self, key: TensorComputeNodeKey) {
        let tensor = self.graph.nodes.tensor.get(&key).unwrap().clone();

        self.graph.cached_results.insert(key.into(), tensor);
    }
}
