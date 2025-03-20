use wgpu::CommandEncoder;

use crate::{
    ElementWiseFunction, PerformanceQueries, UntypedElementWiseKernel, UntypedPairWiseKernel,
    UntypedReduceKernel, element_wise, matmul::UntypedMatMul, quantized::UntypedQMatMul,
    resize::UntypedResizeKernel, slice_assign::UntypedSliceAssignKernel, tensor::TensorData,
};

use super::{
    AnyComputeKey, ComputeGraphInner, ElementWiseComputeNodeKey, MapLayoutComputeNodeKey,
    MatMulComputeNodeKey, PairWiseComputeNodeKey, QMatMulComputeNodeKey, ReduceComputeNodeKey,
    ResizeComputeNodeKey, SliceAssignComputeNodeKey, TensorComputeNodeKey,
    dependency_map::visit_dependencies,
};

pub(crate) struct Resolver<'a> {
    graph: &'a mut ComputeGraphInner,
    command_encoder: &'a mut CommandEncoder,
    target: AnyComputeKey,
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
        }
    }

    pub(crate) fn run(&mut self) -> TensorData {
        self.resolve(self.target)
    }

    fn resolve(&mut self, key: AnyComputeKey) -> TensorData {
        // Check if the key is already resolved
        if let Some(tensor) = self.graph.cached_results.get(&key) {
            return tensor.clone();
        }

        let resolved = match key {
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
            AnyComputeKey::QMatMul(q_mat_mul_compute_node_key) => {
                self.resolve_q_mat_mul(q_mat_mul_compute_node_key)
            }
        };

        // Cache the result
        self.graph.cached_results.insert(key, resolved.clone());
        // Check if that makes any of this nodes dependents dead
        let mut dependencies = Vec::new();
        visit_dependencies(&self.graph.nodes, key, |dependent_key| {
            dependencies.push(dependent_key);
        });
        for dependency in dependencies {
            self.graph.check_life(dependency);
        }

        resolved
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

        // Merge into the output of the reduce kernel if possible and it isn't already cached
        if let AnyComputeKey::Reduce(key) = input {
            if !input_cached {
                return self.resolve_reduce_then(key, functions);
            }
        }
        // Merge into the output of the pair wise kernel if possible and it isn't already cached
        if let AnyComputeKey::PairWise(key) = input {
            if !input_cached {
                return self.resolve_pair_wise_then(key, functions);
            }
        }
        // Otherwise, just run the element wise kernel
        let input = self.resolve(input);
        let kernel = UntypedElementWiseKernel::new(functions, input.datatype());
        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(input, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
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

        let first = self.resolve(first_input);
        let second = self.resolve(second_input);
        let mut kernel = UntypedPairWiseKernel::new(function, first.datatype());
        let first_pre = UntypedElementWiseKernel::new(first_pre_element_wise, first.datatype());
        let second_pre = UntypedElementWiseKernel::new(second_pre_element_wise, first.datatype());
        let pre_element_wise_output = first_pre.out_datatype();
        kernel.set_pre_element_wise([first_pre, second_pre]);
        kernel.set_post_element_wise(UntypedElementWiseKernel::new(then, pre_element_wise_output));
        let query = PerformanceQueries::new(first.device());
        let result = kernel.run_with_query(first, second, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_mat_mul(&mut self, key: MatMulComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.mat_mul.get(&key).unwrap();
        let first = operation.first;
        let second = operation.second;

        let first = self.resolve(first);
        let second = self.resolve(second);
        let kernel = UntypedMatMul::new(first.datatype());
        let query = PerformanceQueries::new(first.device());
        let result =
            kernel.run_with_query(&first, &second, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_q_mat_mul(&mut self, key: QMatMulComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.q_mat_mul.get(&key).unwrap();
        let input = operation.input;
        let matrix = operation.matrix.clone();

        let input = self.resolve(input);
        let kernel = UntypedQMatMul::new(input.datatype(), matrix);
        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(&input, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
        result
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

        let input = self.resolve(input);
        let mut kernel = UntypedReduceKernel::new(function, input.datatype());
        let element_wise_before =
            element_wise::UntypedElementWiseKernel::new(element_wise_before, input.datatype());
        let element_wise_after =
            element_wise::UntypedElementWiseKernel::new(then, element_wise_before.out_datatype());
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);
        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(&input, axis, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_slice(&mut self, key: MapLayoutComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();
        let input = self.resolve(operation.input);
        let operation = self.graph.nodes.map_layout.get(&key).unwrap();

        operation.run(&input)
    }

    fn resolve_resize(&mut self, key: ResizeComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.resize.get(&key).unwrap();
        let input = operation.input;
        let new_shape = operation.new_shape.clone();
        let fill_shape = operation.fill_shape.clone();
        let input = self.resolve(input);
        let kernel = UntypedResizeKernel::new(&new_shape, &fill_shape);

        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(&input, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_slice_assign(&mut self, key: SliceAssignComputeNodeKey) -> TensorData {
        let operation = self.graph.nodes.slice_assign.get(&key).unwrap();
        let input = operation.input;
        let value = operation.value;
        let kernel = UntypedSliceAssignKernel::new(&operation.slices);
        let input = self.resolve(input);
        let value = self.resolve(value);

        let query = PerformanceQueries::new(input.device());
        let result =
            kernel.run_with_query(&input, &value, Some(&query), &mut *self.command_encoder);
        self.graph.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_tensor(&mut self, key: TensorComputeNodeKey) -> TensorData {
        self.graph.nodes.tensor.get(&key).unwrap().clone()
    }
}
