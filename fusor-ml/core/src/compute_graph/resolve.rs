use wgpu::CommandEncoder;

use crate::{
    ElementWiseFunction, PerformanceQueries, UntypedElementWiseKernel, UntypedPairWiseKernel,
    UntypedReduceKernel, element_wise, matmul::UntypedMatMul, resize::UntypedResizeKernel,
    slice_assign::UntypedSliceAssignKernel, tensor::TensorData,
};

use super::{
    AnyComputeKey, ComputeGraphInner, ElementWiseComputeNodeKey, MapLayoutComputeNodeKey,
    MatMulComputeNodeKey, PairWiseComputeNodeKey, ReduceComputeNodeKey, ResizeComputeNodeKey,
    SliceAssignComputeNodeKey, TensorComputeNodeKey,
};

impl ComputeGraphInner {
    pub(crate) fn resolve(
        &mut self,
        key: AnyComputeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        match key {
            AnyComputeKey::ElementWiseComputeNodeKey(element_wise_compute_node_key) => {
                self.resolve_element_wise(element_wise_compute_node_key, command_encoder)
            }
            AnyComputeKey::PairWiseComputeNodeKey(pair_wise_compute_node_key) => {
                self.resolve_pair_wise(pair_wise_compute_node_key, command_encoder)
            }
            AnyComputeKey::MatMulComputeNodeKey(mat_mul_compute_node_key) => {
                self.resolve_mat_mul(mat_mul_compute_node_key, command_encoder)
            }
            AnyComputeKey::ReduceComputeNodeKey(reduce_compute_node_key) => {
                self.resolve_reduce(reduce_compute_node_key, command_encoder)
            }
            AnyComputeKey::TensorComputeNodeKey(tensor_compute_node_key) => {
                self.resolve_tensor(tensor_compute_node_key, command_encoder)
            }
            AnyComputeKey::MapLayoutComputeNodeKey(slice_compute_node_key) => {
                self.resolve_slice(slice_compute_node_key, command_encoder)
            }
            AnyComputeKey::ResizeComputeNodeKey(resize_compute_node_key) => {
                self.resolve_resize(resize_compute_node_key, command_encoder)
            }
            AnyComputeKey::SliceAssignComputeNodeKey(slice_assign_compute_node_key) => {
                self.resolve_slice_assign(slice_assign_compute_node_key, command_encoder)
            }
        }
    }

    fn collect_element_wise_ops(
        &mut self,
        key: ElementWiseComputeNodeKey,
    ) -> (Vec<ElementWiseFunction>, AnyComputeKey) {
        let mut functions = Vec::new();
        let mut current_key = AnyComputeKey::ElementWiseComputeNodeKey(key);
        while let AnyComputeKey::ElementWiseComputeNodeKey(key) = current_key {
            let operation = self.element_wise.get(&key).unwrap();
            functions.push(operation.function.clone());
            current_key = operation.value;
        }
        (functions, current_key)
    }

    fn resolve_element_wise(
        &mut self,
        key: ElementWiseComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        // First collect all element wise ops in this chain
        let (functions, input) = self.collect_element_wise_ops(key);

        // Merge into the output of the reduce kernel if possible
        if let AnyComputeKey::ReduceComputeNodeKey(key) = input {
            self.resolve_reduce_then(key, functions, command_encoder)
        }
        // Merge into the output of the pair wise kernel if possible
        else if let AnyComputeKey::PairWiseComputeNodeKey(key) = input {
            self.resolve_pair_wise_then(key, functions, command_encoder)
        } else {
            let input = self.resolve(input, &mut *command_encoder);
            let kernel = UntypedElementWiseKernel::new(functions, input.datatype());
            let query = PerformanceQueries::new(input.device());
            let result = kernel.run_with_query(input, Some(&query), command_encoder);
            self.timing_information.insert(key.into(), query);
            result
        }
    }

    fn resolve_pair_wise(
        &mut self,
        key: PairWiseComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        self.resolve_pair_wise_then(key, Vec::new(), command_encoder)
    }

    fn resolve_pair_wise_then(
        &mut self,
        key: PairWiseComputeNodeKey,
        then: Vec<ElementWiseFunction>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.pair_wise.get(&key).unwrap();
        let function = operation.function.clone();

        let mut first_input = operation.first;
        let mut second_input = operation.second;
        let first_pre_element_wise =
            if let AnyComputeKey::ElementWiseComputeNodeKey(key) = first_input {
                let (functions, element_wise_input) = self.collect_element_wise_ops(key);
                first_input = element_wise_input;
                functions
            } else {
                Vec::new()
            };
        let second_pre_element_wise =
            if let AnyComputeKey::ElementWiseComputeNodeKey(key) = second_input {
                let (functions, element_wise_input) = self.collect_element_wise_ops(key);
                second_input = element_wise_input;
                functions
            } else {
                Vec::new()
            };

        let first = self.resolve(first_input, &mut *command_encoder);
        let second = self.resolve(second_input, &mut *command_encoder);
        let mut kernel = UntypedPairWiseKernel::new(function, first.datatype());
        let first_pre = UntypedElementWiseKernel::new(first_pre_element_wise, first.datatype());
        let second_pre = UntypedElementWiseKernel::new(second_pre_element_wise, first.datatype());
        let pre_element_wise_output = first_pre.out_datatype();
        kernel.set_pre_element_wise([first_pre, second_pre]);
        kernel.set_post_element_wise(UntypedElementWiseKernel::new(then, pre_element_wise_output));
        let query = PerformanceQueries::new(first.device());
        let result = kernel.run_with_query(first, second, Some(&query), command_encoder);
        self.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_mat_mul(
        &mut self,
        key: MatMulComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.mat_mul.get(&key).unwrap();
        let first = operation.first;
        let second = operation.second;

        let first = self.resolve(first, &mut *command_encoder);
        let second = self.resolve(second, &mut *command_encoder);
        let kernel = UntypedMatMul::new(first.datatype());
        let query = PerformanceQueries::new(first.device());
        let result = kernel.run_with_query(&first, &second, Some(&query), command_encoder);
        self.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_reduce(
        &mut self,
        key: ReduceComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        self.resolve_reduce_then(key, Vec::new(), command_encoder)
    }

    fn resolve_reduce_then(
        &mut self,
        key: ReduceComputeNodeKey,
        then: Vec<ElementWiseFunction>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.reduce.get(&key).unwrap();
        let mut input = operation.value;
        let axis = operation.axis;
        let function = operation.function.clone();

        let element_wise_before =
            if let AnyComputeKey::ElementWiseComputeNodeKey(key) = operation.value {
                let (functions, element_wise_input) = self.collect_element_wise_ops(key);
                input = element_wise_input;
                functions
            } else {
                Vec::new()
            };

        let input = self.resolve(input, &mut *command_encoder);
        let mut kernel = UntypedReduceKernel::new(function, input.datatype());
        let element_wise_before =
            element_wise::UntypedElementWiseKernel::new(element_wise_before, input.datatype());
        let element_wise_after =
            element_wise::UntypedElementWiseKernel::new(then, element_wise_before.out_datatype());
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);
        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(&input, axis, Some(&query), command_encoder);
        self.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_slice(
        &mut self,
        key: MapLayoutComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.map_layout.get(&key).unwrap();
        let input = self.resolve(operation.input, &mut *command_encoder);
        let operation = self.map_layout.get(&key).unwrap();

        operation.run(&input)
    }

    fn resolve_resize(
        &mut self,
        key: ResizeComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.resize.get(&key).unwrap();
        let input = operation.input;
        let new_shape = operation.new_shape.clone();
        let fill_shape = operation.fill_shape.clone();
        let input = self.resolve(input, &mut *command_encoder);
        let kernel = UntypedResizeKernel::new(&new_shape, &fill_shape);

        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(&input, Some(&query), command_encoder);
        self.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_slice_assign(
        &mut self,
        key: SliceAssignComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.slice_assign.get(&key).unwrap();
        let input = operation.input;
        let value = operation.value;
        let kernel = UntypedSliceAssignKernel::new(&operation.slices);
        let input = self.resolve(input, &mut *command_encoder);
        let value = self.resolve(value, &mut *command_encoder);

        let query = PerformanceQueries::new(input.device());
        let result = kernel.run_with_query(&input, &value, Some(&query), command_encoder);
        self.timing_information.insert(key.into(), query);
        result
    }

    fn resolve_tensor(&mut self, key: TensorComputeNodeKey, _: &mut CommandEncoder) -> TensorData {
        self.tensor.get(&key).unwrap().clone()
    }
}
