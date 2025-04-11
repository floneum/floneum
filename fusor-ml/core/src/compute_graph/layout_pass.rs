use std::collections::HashMap;

use crate::{Layout, TensorLayoutInfo, index_select::IndexSelectOperation};

use super::{AnyComputeKey, queue::ComputeQueue};

#[derive(Default)]
pub(crate) struct LayoutPass {
    queue: ComputeQueue,
    pub(crate) output_layout: HashMap<AnyComputeKey, TensorLayoutInfo>,
}

impl LayoutPass {
    pub fn visit(&mut self, graph: &super::ComputeGraphNodes, key: AnyComputeKey) {
        self.queue.push_back(key);

        while let Some(node) = self.queue.pop_front() {
            if self.output_layout.contains_key(&node) {
                continue;
            }
            match node {
                AnyComputeKey::ElementWise(key) => self.visit_element_wise(graph, key),
                AnyComputeKey::PairWise(key) => self.visit_pair_wise(graph, key),
                AnyComputeKey::MatMul(key) => self.visit_mat_mul(graph, key),
                AnyComputeKey::QMatMul(key) => self.visit_q_mat_mul(graph, key),
                AnyComputeKey::Reduce(key) => self.visit_reduce(graph, key),
                AnyComputeKey::MapLayout(key) => self.visit_map_layout(graph, key),
                AnyComputeKey::Resize(key) => self.visit_resize(graph, key),
                AnyComputeKey::SliceAssign(key) => self.visit_slice_assign(graph, key),
                AnyComputeKey::Tensor(key) => self.visit_tensor(graph, key),
                AnyComputeKey::Dequantize(key) => self.visit_dequantize(graph, key),
                AnyComputeKey::IndexSelect(key) => self.visit_index_select(graph, key),
            }
        }
    }

    fn visit_element_wise(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::ElementWiseComputeNodeKey,
    ) {
        let operation = graph.element_wise.get(&key).unwrap();
        let input = operation.value;
        let Some(input_layout) = self.output_layout.get(&input) else {
            self.queue.push_back(input);
            self.queue.push_back(key.into());
            return;
        };
        let output_layout =
            TensorLayoutInfo::new(input_layout.layout().clone(), operation.function.datatype());
        self.output_layout.insert(key.into(), output_layout);
    }

    fn visit_pair_wise(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::PairWiseComputeNodeKey,
    ) {
        let operation = graph.pair_wise.get(&key).unwrap();
        let Some(first_layout) = self.output_layout.get(&operation.first) else {
            self.queue.push_back(operation.first);
            self.queue.push_back(key.into());
            return;
        };
        let Some(_) = self.output_layout.get(&operation.second) else {
            self.queue.push_back(operation.second);
            self.queue.push_back(key.into());
            return;
        };
        self.output_layout.insert(key.into(), first_layout.clone());
    }

    fn visit_mat_mul(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::MatMulComputeNodeKey,
    ) {
        let operation = graph.mat_mul.get(&key).unwrap();
        let Some(first_layout) = self.output_layout.get(&operation.first) else {
            self.queue.push_back(operation.first);
            self.queue.push_back(key.into());
            return;
        };
        let Some(_) = self.output_layout.get(&operation.second) else {
            self.queue.push_back(operation.second);
            self.queue.push_back(key.into());
            return;
        };
        let output_shape = &operation.out_shape;
        let output_layout = Layout::contiguous(output_shape);
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_q_mat_mul(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::QMatMulComputeNodeKey,
    ) {
        let operation = graph.q_mat_mul.get(&key).unwrap();
        let Some(first_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key.into());
            return;
        };
        let output_layout = Layout::contiguous(&operation.out_shape);
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_reduce(&mut self, graph: &super::ComputeGraphNodes, key: super::ReduceComputeNodeKey) {
        let operation = graph.reduce.get(&key).unwrap();
        let dim = operation.axis;
        let Some(input_layout) = self.output_layout.get(&operation.value) else {
            self.queue.push_back(operation.value);
            self.queue.push_back(key.into());
            return;
        };
        let new_shape = input_layout
            .layout()
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect::<Vec<_>>();
        let new_layout = Layout::contiguous(&new_shape);
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_map_layout(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::MapLayoutComputeNodeKey,
    ) {
        let operation = graph.map_layout.get(&key).unwrap();
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key.into());
            return;
        };
        let new_layout = operation.map_layout(input_layout.layout());
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_resize(&mut self, graph: &super::ComputeGraphNodes, key: super::ResizeComputeNodeKey) {
        let operation = graph.resize.get(&key).unwrap();
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key.into());
            return;
        };
        let new_layout = Layout::contiguous(&operation.new_shape);
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_slice_assign(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::SliceAssignComputeNodeKey,
    ) {
        let operation = graph.slice_assign.get(&key).unwrap();
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key.into());
            return;
        };
        let Some(_) = self.output_layout.get(&operation.value) else {
            self.queue.push_back(operation.value);
            self.queue.push_back(key.into());
            return;
        };
        self.output_layout.insert(key.into(), input_layout.clone());
    }

    fn visit_tensor(&mut self, graph: &super::ComputeGraphNodes, key: super::TensorComputeNodeKey) {
        let operation = graph.tensor.get(&key).unwrap();
        let info = operation.info();
        self.output_layout.insert(key.into(), info.clone());
    }

    fn visit_dequantize(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::DequantizeComputeKey,
    ) {
        let operation = graph.dequantize.get(&key).unwrap();
        let matrix = &operation.matrix;
        let new_layout = Layout::contiguous(matrix.shape());
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(new_layout, operation.datatype),
        );
    }

    fn visit_index_select(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::IndexSelectComputeNodeKey,
    ) {
        let operation = graph.index_select.get(&key).unwrap();
        let Some(indexes_shape) = self.output_layout.get(&operation.indexes) else {
            self.queue.push_back(operation.indexes);
            self.queue.push_back(key.into());
            return;
        };
        let Some(input_shape) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key.into());
            return;
        };
        let shape = IndexSelectOperation::output_shape(
            operation.dimension,
            indexes_shape.shape(),
            input_shape.shape(),
        );
        let new_layout = Layout::contiguous(&shape);
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(new_layout, operation.datatype),
        );
    }
}
