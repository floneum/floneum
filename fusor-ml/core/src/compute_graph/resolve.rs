use wgpu::CommandEncoder;

use crate::{
    ElementWiseFunction, PerformanceQueries, UntypedElementWiseKernel, UntypedPairWiseKernel,
    UntypedReduceKernel, dequantize::UntypedDequantize, element_wise,
    index_select::UntypedIndexSelectKernel, matmul::UntypedMatMul,
    quantized::matmul::UntypedQMatMul, resize::UntypedResizeKernel,
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
    timing_information: bool,
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
            timing_information: false,
        }
    }

    fn with_query<O>(
        &mut self,
        key: AnyComputeKey,
        with_query: impl FnOnce(&mut Self, Option<&PerformanceQueries>) -> O,
    ) -> O {
        let query = self
            .timing_information
            .then(|| PerformanceQueries::new(&self.graph.device));
        let result = with_query(self, query.as_ref());
        if let Some(query) = query {
            self.graph.timing_information.insert(key, query);
        }
        result
    }

    pub(crate) fn run(&mut self) -> TensorData {
        self.queue.push_back(self.target);

        while let Some(node) = self.queue.pop_front() {
            if self.graph.cached_results.contains_key(&node) {
                continue;
            }

            let mut dependencies = Vec::new();
            visit_dependencies(&self.graph.nodes, node, |dependent_key| {
                dependencies.push(dependent_key);
            });
            dependencies.retain(|dependency| !self.graph.cached_results.contains_key(dependency));
            if !dependencies.is_empty() {
                // If there are dependencies that are not resolved, push them to the queue then
                // revisit this node
                for dependency in dependencies {
                    self.queue.push_back(dependency);
                }
                self.queue.push_back(node);
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
            let operation = self.graph.nodes.element_wise.get(&key).unwrap();
            functions.push(operation.function.clone());
            current_key = operation.value;
        }
        (functions, current_key)
    }

    fn resolve_element_wise(&mut self, key: ElementWiseComputeNodeKey) -> TensorData {
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
            // Merge into the output of the dequantize kernel if possible and it isn't already cached
            if let AnyComputeKey::Dequantize(key) = input {
                return self.resolve_dequantize_then(key, functions);
            }
        }
        // Otherwise, just run the element wise kernel
        let input = self.graph.cached_results[&input].clone();
        let kernel = UntypedElementWiseKernel::new(functions, input.datatype());
        let result = self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(input.into(), query, &mut *resolver.command_encoder)
        });
        result
    }

    fn resolve_pair_wise(&mut self, key: PairWiseComputeNodeKey) -> TensorData {
        self.resolve_pair_wise_then(key, Vec::new())
    }

    fn resolve_pair_wise_then(
        &mut self,
        key: PairWiseComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> TensorData {
        let operation = self.graph.nodes.pair_wise.get(&key).unwrap();
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
            self.graph.cached_results[&first_input].clone().into()
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
            self.graph.cached_results[&second_input].clone().into()
        };
        let mut kernel = UntypedPairWiseKernel::new(function);
        let first_pre =
            UntypedElementWiseKernel::new(first_pre_element_wise, first.dequantized_datatype());
        let second_pre =
            UntypedElementWiseKernel::new(second_pre_element_wise, second.dequantized_datatype());
        let pre_element_wise_output = first_pre.out_datatype();
        kernel.set_pre_element_wise([first_pre, second_pre]);
        kernel.set_post_element_wise(UntypedElementWiseKernel::new(then, pre_element_wise_output));
        let result = self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(first, second, query, &mut *resolver.command_encoder)
        });
        result
    }

    fn resolve_mat_mul(&mut self, key: MatMulComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.mat_mul.get(&key).unwrap();
        let first = operation.first;
        let second = operation.second;

        let first = self.graph.cached_results[&first].clone();
        let second = self.graph.cached_results[&second].clone();
        let kernel = UntypedMatMul::new(first.datatype(), first.layout().rank() as u32);
        self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(&first, &second, query, &mut *resolver.command_encoder)
        })
    }

    fn resolve_q_mat_mul(&mut self, key: QMatMulComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.q_mat_mul.get(&key).unwrap();
        let input = operation.input;
        let matrix = operation.matrix.clone();

        let input = self.graph.cached_results[&input].clone();
        let kernel = UntypedQMatMul::new(input.datatype(), matrix);
        let result = self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(&input, query, &mut *resolver.command_encoder)
        });
        result
    }

    fn resolve_dequantize(&mut self, key: DequantizeComputeKey) -> TensorData {
        self.resolve_dequantize_then(key, Vec::new())
    }

    fn resolve_dequantize_then(
        &mut self,
        key: DequantizeComputeKey,
        then: Vec<ElementWiseFunction>,
    ) -> TensorData {
        let operation = self.graph.nodes.dequantize.get(&key).unwrap();

        let mut kernel = UntypedDequantize::new(operation.datatype, operation.matrix.clone());
        let then = element_wise::UntypedElementWiseKernel::new(then, operation.datatype);
        kernel.set_post_element_wise(then);
        self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(
                &resolver.graph.device,
                query,
                &mut *resolver.command_encoder,
            )
        })
    }

    fn resolve_reduce(&mut self, key: ReduceComputeNodeKey) -> TensorData {
        self.resolve_reduce_then(key, Vec::new())
    }

    fn resolve_reduce_then(
        &mut self,
        key: ReduceComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> TensorData {
        let operation = self.graph.nodes.reduce.get(&key).unwrap();
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

        let input = self.graph.cached_results[&input].clone();
        let mut kernel = UntypedReduceKernel::new(function, input.datatype());
        let element_wise_before =
            element_wise::UntypedElementWiseKernel::new(element_wise_before, input.datatype());
        let element_wise_after =
            element_wise::UntypedElementWiseKernel::new(then, element_wise_before.out_datatype());
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);
        self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(&input, axis, query, &mut *resolver.command_encoder)
        })
    }

    fn resolve_slice(&mut self, key: MapLayoutComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();
        let input = self.graph.cached_results[&operation.input].clone();
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();

        operation.run(&input)
    }

    fn resolve_resize(&mut self, key: ResizeComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.resize.get(&key).unwrap();
        let input = operation.input;
        let new_shape = operation.new_shape.clone();
        let fill_shape = operation.fill_shape.clone();
        let input = self.graph.cached_results[&input].clone();
        let kernel = UntypedResizeKernel::new(&new_shape, &fill_shape);

        self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(&input, query, &mut *resolver.command_encoder)
        })
    }

    fn resolve_slice_assign(&mut self, key: SliceAssignComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.slice_assign.get(&key).unwrap();
        let input = operation.input;
        let value = operation.value;
        let kernel = UntypedSliceAssignKernel::new(&operation.slices);
        let input = self.graph.cached_results[&input].clone();
        let value = self.graph.cached_results[&value].clone();

        self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(&input, &value, query, &mut *resolver.command_encoder)
        })
    }

    fn resolve_index_select(&mut self, key: IndexSelectComputeNodeKey) -> TensorData {
        self.resolve_index_select_then(key, Vec::new())
    }

    fn resolve_index_select_then(
        &mut self,
        key: IndexSelectComputeNodeKey,
        then: Vec<ElementWiseFunction>,
    ) -> TensorData {
        let operation = self.graph.nodes.index_select.get(&key).unwrap();

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

        let input = self.graph.cached_results[&input].clone();
        let indexes = self.graph.cached_results[&indexes].clone();
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

        self.with_query(key.into(), |resolver, query| {
            kernel.run_with_query(&input, &indexes, query, &mut *resolver.command_encoder)
        })
    }

    fn resolve_tensor(&mut self, key: TensorComputeNodeKey) -> TensorData {
        self.graph.nodes.tensor.get(&key).unwrap().clone()
    }
}
