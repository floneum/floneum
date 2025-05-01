use wgpu::CommandEncoder;

use crate::{
    ElementWiseFunction, UntypedElementWiseKernel, UntypedPairWiseKernel, UntypedReduceKernel,
    dequantize::UntypedDequantize, element_wise, index_select::UntypedIndexSelectKernel,
    matmul::UntypedMatMul, quantized::matmul::UntypedQMatMul, resize::UntypedResizeKernel,
    slice_assign::UntypedSliceAssignKernel, tensor::TensorData, visit_tiled::MaybeQData,
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
    target: AnyComputeKey,
    queue: ComputeQueue,
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
            queue: Default::default(),
        }
    }

    pub(crate) fn run(&mut self) -> TensorData {
        self.queue.push_back(self.target);

        while let Some(node) = self.queue.pop_front() {
            if self.graph.cached_results.contains_key(&node) {
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
                    self.resolve_tensor(tensor_compute_node_key)
                }
                AnyComputeKey::MapLayout(slice_compute_node_key) => {
                    self.resolve_slice(slice_compute_node_key)
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

    fn collect_element_wise_ops(
        &mut self,
        key: ElementWiseComputeNodeKey,
    ) -> (Vec<ElementWiseFunction>, AnyComputeKey) {
        let mut functions = Vec::new();
        let mut current_key = AnyComputeKey::ElementWise(key);
        while let AnyComputeKey::ElementWise(key) = current_key {
            // If the result is already cached, stop collecting element wise ops
            if self.graph.cached_results.contains_key(&current_key) {
                break;
            }
            let operation = &self.graph.nodes.element_wise[&key];
            functions.push(operation.function.clone());
            current_key = operation.value;
        }
        (functions, current_key)
    }

    fn resolve_element_wise(&mut self, key: ElementWiseComputeNodeKey) -> Option<TensorData> {
        // First collect all element wise ops in this chain
        let (functions, input) = self.collect_element_wise_ops(key);
        let input_cached = self.graph.cached_results.contains_key(&input);

        if !input_cached {
            // Merge into the output of the reduce kernel if possible and it isn't already cached
            if let AnyComputeKey::Reduce(key) = input {
                return self.resolve_reduce_then(key, functions);
            }
            // Merge into the output of the pair wise kernel if possible and it isn't already cached
            if let AnyComputeKey::PairWise(key) = input {
                return self.resolve_pair_wise_then(key, functions);
            }
            // Merge into the output of the mat mul kernel if possible and it isn't already cached
            if let AnyComputeKey::MatMul(key) = input {
                return self.resolve_mat_mul_then(key, functions);
            }
            // Merge into the output of the dequantize kernel if possible and it isn't already cached
            if let AnyComputeKey::Dequantize(key) = input {
                return self.resolve_dequantize_then(key, functions);
            }
        }
        // Otherwise, just run the element wise kernel
        let Some(input) = self.graph.cached_results.get(&input).cloned() else {
            self.queue.push_back(input);
            return None;
        };
        let kernel = UntypedElementWiseKernel::new(functions, input.datatype());

        let result = kernel.run(input.into(), &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_pair_wise(&mut self, key: PairWiseComputeNodeKey) -> Option<TensorData> {
        self.resolve_pair_wise_then(key, Vec::new())
    }

    fn resolve_pair_wise_then(
        &mut self,
        key: PairWiseComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> Option<TensorData> {
        let operation = &self.graph.nodes.pair_wise[&key];
        let function = operation.function.clone();

        let mut first_input = operation.first;
        let mut second_input = operation.second;
        let first_pre_element_wise = if let AnyComputeKey::ElementWise(key) = first_input {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            first_input = element_wise_input;
            functions
        } else {
            Vec::new()
        };
        let second_pre_element_wise = if let AnyComputeKey::ElementWise(key) = second_input {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            second_input = element_wise_input;
            functions
        } else {
            Vec::new()
        };

        let first: MaybeQData = if let AnyComputeKey::Dequantize(key) = first_input {
            self.graph
                .nodes
                .dequantize
                .get(&key)
                .unwrap()
                .matrix
                .clone()
                .into()
        } else {
            let Some(first) = self.graph.cached_results.get(&first_input) else {
                self.queue.push_back(first_input);
                return None;
            };
            first.clone().into()
        };
        let second: MaybeQData = if let AnyComputeKey::Dequantize(key) = second_input {
            self.graph
                .nodes
                .dequantize
                .get(&key)
                .unwrap()
                .matrix
                .clone()
                .into()
        } else {
            let Some(second) = self.graph.cached_results.get(&second_input) else {
                self.queue.push_back(second_input);
                return None;
            };
            second.clone().into()
        };
        let mut kernel = UntypedPairWiseKernel::new(function);
        let first_pre =
            UntypedElementWiseKernel::new(first_pre_element_wise, first.dequantized_datatype());
        let second_pre =
            UntypedElementWiseKernel::new(second_pre_element_wise, second.dequantized_datatype());
        let pre_element_wise_output = first_pre.out_datatype();
        kernel.set_pre_element_wise([first_pre, second_pre]);
        kernel.set_post_element_wise(UntypedElementWiseKernel::new(then, pre_element_wise_output));

        let result = kernel.run(first, second, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_mat_mul(&mut self, key: MatMulComputeNodeKey) -> Option<TensorData> {
        self.resolve_mat_mul_then(key, Vec::new())
    }

    fn resolve_mat_mul_then(
        &mut self,
        key: MatMulComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> Option<TensorData> {
        let operation = &self.graph.nodes.mat_mul[&key];
        let mut first = operation.first;
        let mut second = operation.second;

        let first_pre_element_wise = if let AnyComputeKey::ElementWise(key) = first {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            first = element_wise_input;
            functions
        } else {
            Vec::new()
        };
        let second_pre_element_wise = if let AnyComputeKey::ElementWise(key) = second {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            second = element_wise_input;
            functions
        } else {
            Vec::new()
        };

        let Some(first) = self.graph.cached_results.get(&first).cloned() else {
            self.queue.push_back(first);
            return None;
        };
        let Some(second) = self.graph.cached_results.get(&second).cloned() else {
            self.queue.push_back(second);
            return None;
        };
        let mut kernel = UntypedMatMul::new(first.datatype(), first.layout().rank() as u32);
        let first_pre = UntypedElementWiseKernel::new(first_pre_element_wise, first.datatype());
        let second_pre = UntypedElementWiseKernel::new(second_pre_element_wise, second.datatype());
        let pre_element_wise_output = first_pre.out_datatype();
        kernel.set_pre_element_wise([first_pre, second_pre]);
        kernel.set_post_element_wise(UntypedElementWiseKernel::new(then, pre_element_wise_output));
        let result = kernel.run(&first, &second, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_q_mat_mul(&mut self, key: QMatMulComputeNodeKey) -> Option<TensorData> {
        let operation = &self.graph.nodes.q_mat_mul[&key];
        let input = operation.input;
        let matrix = operation.matrix.clone();

        let Some(input) = self.graph.cached_results.get(&input).cloned() else {
            self.queue.push_back(input);
            return None;
        };
        let kernel = UntypedQMatMul::new(input.datatype(), matrix);

        let result = kernel.run(&input, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_dequantize(&mut self, key: DequantizeComputeKey) -> Option<TensorData> {
        self.resolve_dequantize_then(key, Vec::new())
    }

    fn resolve_dequantize_then(
        &mut self,
        key: DequantizeComputeKey,
        then: Vec<ElementWiseFunction>,
    ) -> Option<TensorData> {
        let operation = &self.graph.nodes.dequantize[&key];

        let mut kernel = UntypedDequantize::new(operation.datatype, operation.matrix.clone());
        let then = element_wise::UntypedElementWiseKernel::new(then, operation.datatype);
        kernel.set_post_element_wise(then);
        let result = kernel.run(&self.graph.device, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_reduce(&mut self, key: ReduceComputeNodeKey) -> Option<TensorData> {
        self.resolve_reduce_then(key, Vec::new())
    }

    fn resolve_reduce_then(
        &mut self,
        key: ReduceComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> Option<TensorData> {
        let operation = &self.graph.nodes.reduce[&key];
        let mut input = operation.value;
        let axis = operation.axis;
        let function = operation.function.clone();

        let element_wise_before = if let AnyComputeKey::ElementWise(key) = operation.value {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            input = element_wise_input;
            functions
        } else {
            Vec::new()
        };

        let Some(input) = self.graph.cached_results.get(&input).cloned() else {
            self.queue.push_back(input);
            return None;
        };
        let mut kernel = UntypedReduceKernel::new(function, input.datatype());
        let element_wise_before =
            element_wise::UntypedElementWiseKernel::new(element_wise_before, input.datatype());
        let element_wise_after =
            element_wise::UntypedElementWiseKernel::new(then, element_wise_before.out_datatype());
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);
        let result = kernel.run(&input, axis, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_slice(&mut self, key: MapLayoutComputeNodeKey) -> Option<TensorData> {
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();
        let Some(input) = self.graph.cached_results.get(&operation.input) else {
            self.queue.push_back(operation.input);
            return None;
        };
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();

        let result = operation.run(input);

        Some(result)
    }

    fn resolve_resize(&mut self, key: ResizeComputeNodeKey) -> Option<TensorData> {
        let operation = self.graph.nodes.resize.get(&key).unwrap();
        let input = operation.input;
        let new_shape = operation.new_shape.clone();
        let fill_shape = operation.fill_shape.clone();
        let Some(input) = self.graph.cached_results.get(&input).cloned() else {
            self.queue.push_back(input);
            return None;
        };
        let kernel = UntypedResizeKernel::new(&new_shape, &fill_shape);

        let result = kernel.run(&input, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_slice_assign(&mut self, key: SliceAssignComputeNodeKey) -> Option<TensorData> {
        let operation = self.graph.nodes.slice_assign.get(&key).unwrap();
        let input = operation.input;
        let value = operation.value;
        let kernel = UntypedSliceAssignKernel::new(&operation.slices);
        let Some(input) = self.graph.cached_results.get(&input).cloned() else {
            self.queue.push_back(input);
            return None;
        };
        let Some(value) = self.graph.cached_results.get(&value).cloned() else {
            self.queue.push_back(value);
            return None;
        };

        let result = kernel.run(&input, &value, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_index_select(&mut self, key: IndexSelectComputeNodeKey) -> Option<TensorData> {
        self.resolve_index_select_then(key, Vec::new())
    }

    fn resolve_index_select_then(
        &mut self,
        key: IndexSelectComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> Option<TensorData> {
        let operation = &self.graph.nodes.index_select[&key];

        let dimension = operation.dimension;
        let mut input = operation.input;
        let mut indexes = operation.indexes;
        let mut input_pre_element_wise = if let AnyComputeKey::ElementWise(key) = input {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            input = element_wise_input;
            functions
        } else {
            Vec::new()
        };
        // pre and post elementwise are the same since the index select operation doesn't effect element wise values
        for function in then {
            input_pre_element_wise.push(function);
        }
        let indexes_pre_element_wise = if let AnyComputeKey::ElementWise(key) = indexes {
            let (functions, element_wise_input) = self.collect_element_wise_ops(key);
            indexes = element_wise_input;
            functions
        } else {
            Vec::new()
        };

        let Some(input) = self.graph.cached_results.get(&input).cloned() else {
            self.queue.push_back(input);
            return None;
        };
        let Some(indexes) = self.graph.cached_results.get(&indexes).cloned() else {
            self.queue.push_back(indexes);
            return None;
        };
        let mut kernel =
            UntypedIndexSelectKernel::new(dimension, input.datatype(), input.layout().rank());
        kernel.set_pre_element_wise_input(UntypedElementWiseKernel::new(
            input_pre_element_wise,
            input.datatype(),
        ));
        kernel.set_pre_element_wise_indexes(UntypedElementWiseKernel::new(
            indexes_pre_element_wise,
            indexes.datatype(),
        ));

        let result = kernel.run(&input, &indexes, &mut *self.command_encoder);

        Some(result)
    }

    fn resolve_tensor(&mut self, key: TensorComputeNodeKey) -> Option<TensorData> {
        Some(self.graph.nodes.tensor.get(&key).unwrap().clone())
    }
}
