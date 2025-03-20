use std::collections::HashMap;

use crate::{Layout, TensorLayoutInfo};

use super::{
    AnyComputeKey,
    visit::{
        VisitComputeGraph, visit_element_wise, visit_mat_mul, visit_pair_wise, visit_reduce,
        visit_resize, visit_map_layout, visit_slice_assign, visit_tensor,
    },
};

#[derive(Default)]
pub struct LayoutPass {
    pub(crate) output_layout: HashMap<AnyComputeKey, TensorLayoutInfo>,
}

impl VisitComputeGraph for LayoutPass {
    fn visit_element_wise(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::ElementWiseComputeNodeKey,
    ) {
        visit_element_wise(self, graph, key);
        let operation = graph.element_wise.get(&key).unwrap();
        let input = operation.value;
        let input_layout = self.output_layout.get(&input).unwrap();
        let output_layout =
            TensorLayoutInfo::new(input_layout.layout().clone(), operation.function.datatype());
        self.output_layout.insert(key.into(), output_layout);
    }

    fn visit_pair_wise(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::PairWiseComputeNodeKey,
    ) {
        visit_pair_wise(self, graph, key);
        let operation = graph.pair_wise.get(&key).unwrap();
        let first = operation.first;
        let first_layout = self.output_layout.get(&first).unwrap();
        self.output_layout.insert(key.into(), first_layout.clone());
    }

    fn visit_mat_mul(
        &mut self,
        graph: &super::ComputeGraphNodes,
        key: super::MatMulComputeNodeKey,
    ) {
        visit_mat_mul(self, graph, key);
        let operation = graph.mat_mul.get(&key).unwrap();
        let first = operation.first;
        let first_layout = self.output_layout.get(&first).unwrap();
        let second_layout = self.output_layout.get(&operation.second).unwrap();
        let first_shape = first_layout.layout().shape();
        let second_shape = second_layout.layout().shape();
        let output_shape = [first_shape[1], second_shape[0]];
        let output_layout = Layout::contiguous(&output_shape);
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_reduce(&mut self, graph: &super::ComputeGraphNodes, key: super::ReduceComputeNodeKey) {
        visit_reduce(self, graph, key);
        let operation = graph.reduce.get(&key).unwrap();
        let input = operation.value;
        let dim = operation.axis;
        let input_layout = self.output_layout.get(&input).unwrap();
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
        visit_map_layout(self, graph, key);
        let operation = graph.map_layout.get(&key).unwrap();
        let input = operation.input;
        let input_layout = self.output_layout.get(&input).unwrap();
        let new_layout = operation.map_layout(input_layout.layout());
        self.output_layout.insert(
            key.into(),
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_resize(&mut self, graph: &super::ComputeGraphNodes, key: super::ResizeComputeNodeKey) {
        visit_resize(self, graph, key);
        let operation = graph.resize.get(&key).unwrap();
        let input = operation.input;
        let input_layout = self.output_layout.get(&input).unwrap();
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
        visit_slice_assign(self, graph, key);
        let operation = graph.slice_assign.get(&key).unwrap();
        let input = operation.input;
        let input_layout = self.output_layout.get(&input).unwrap();
        self.output_layout.insert(key.into(), input_layout.clone());
    }

    fn visit_tensor(&mut self, graph: &super::ComputeGraphNodes, key: super::TensorComputeNodeKey) {
        visit_tensor(self, graph, key);
        let operation = graph.tensor.get(&key).unwrap();
        let info = operation.info();
        self.output_layout.insert(key.into(), info.clone());
    }
}
