use rustc_hash::FxHashMap;

use crate::{Layout, TensorLayoutInfo, index_select::IndexSelectOperation};

use super::{NodeIndex, ComputeGraphNodeVariant, queue::ComputeQueue};

#[derive(Default)]
pub(crate) struct LayoutPass {
    queue: ComputeQueue,
    pub(crate) output_layout: FxHashMap<NodeIndex, TensorLayoutInfo>,
}

impl LayoutPass {
    pub fn visit(&mut self, graph: &super::ComputeGraphInner, key: NodeIndex) {
        self.queue.push_back(key);

        while let Some(node) = self.queue.pop_front() {
            if self.output_layout.contains_key(&node) {
                continue;
            }
            if let Some(resolved) = graph.cached_results.get(&node) {
                self.output_layout.insert(node, resolved.info().clone());
                continue;
            }
            let node_data = graph.nodes.nodes.node_weight(node).expect("Node not found");
            match &node_data.variant {
                ComputeGraphNodeVariant::ElementWise(_) => self.visit_element_wise(graph, node),
                ComputeGraphNodeVariant::PairWise(_) => self.visit_pair_wise(graph, node),
                ComputeGraphNodeVariant::MatMul(_) => self.visit_mat_mul(graph, node),
                ComputeGraphNodeVariant::QMatMul(_) => self.visit_q_mat_mul(graph, node),
                ComputeGraphNodeVariant::Reduce(_) => self.visit_reduce(graph, node),
                ComputeGraphNodeVariant::MapLayout(_) => self.visit_map_layout(graph, node),
                ComputeGraphNodeVariant::Resize(_) => self.visit_resize(graph, node),
                ComputeGraphNodeVariant::SliceAssign(_) => self.visit_slice_assign(graph, node),
                ComputeGraphNodeVariant::Tensor(_) => self.visit_tensor(graph, node),
                ComputeGraphNodeVariant::Dequantize(_) => self.visit_dequantize(graph, node),
                ComputeGraphNodeVariant::IndexSelect(_) => self.visit_index_select(graph, node),
                ComputeGraphNodeVariant::Custom(_) => self.visit_custom(graph, node),
            }
        }
    }

    fn visit_element_wise(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::ElementWise(op) => op,
            _ => panic!("Expected ElementWise node"),
        };
        let input = operation.value;
        let Some(input_layout) = self.output_layout.get(&input) else {
            self.queue.push_back(input);
            self.queue.push_back(key);
            return;
        };
        let output_layout = TensorLayoutInfo::new(
            input_layout.layout().clone(),
            operation.functions.out_datatype(),
        );
        self.output_layout.insert(key, output_layout);
    }

    fn visit_pair_wise(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::PairWise(op) => op,
            _ => panic!("Expected PairWise node"),
        };
        let Some(first_layout) = self.output_layout.get(&operation.first) else {
            self.queue.push_back(operation.first);
            self.queue.push_back(key);
            return;
        };
        let Some(_) = self.output_layout.get(&operation.second) else {
            self.queue.push_back(operation.second);
            self.queue.push_back(key);
            return;
        };
        self.output_layout.insert(key, first_layout.clone());
    }

    fn visit_mat_mul(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::MatMul(op) => op,
            _ => panic!("Expected MatMul node"),
        };
        let Some(first_layout) = self.output_layout.get(&operation.first) else {
            self.queue.push_back(operation.first);
            self.queue.push_back(key);
            return;
        };
        let Some(_) = self.output_layout.get(&operation.second) else {
            self.queue.push_back(operation.second);
            self.queue.push_back(key);
            return;
        };
        let output_shape = &operation.out_shape;
        let output_layout = Layout::contiguous(output_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_q_mat_mul(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::QMatMul(op) => op,
            _ => panic!("Expected QMatMul node"),
        };
        let Some(first_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let output_layout = Layout::contiguous(&operation.out_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_reduce(&mut self, graph: &super::ComputeGraphInner, key: NodeIndex) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Reduce(op) => op,
            _ => panic!("Expected Reduce node"),
        };
        let dim = operation.axis;
        let Some(input_layout) = self.output_layout.get(&operation.value) else {
            self.queue.push_back(operation.value);
            self.queue.push_back(key);
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
            key,
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_map_layout(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::MapLayout(op) => op,
            _ => panic!("Expected MapLayout node"),
        };
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let new_layout = operation.map_layout(input_layout.layout());
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_resize(&mut self, graph: &super::ComputeGraphInner, key: NodeIndex) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Resize(op) => op,
            _ => panic!("Expected Resize node"),
        };
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let new_layout = Layout::contiguous(&operation.new_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_slice_assign(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::SliceAssign(op) => op,
            _ => panic!("Expected SliceAssign node"),
        };
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let Some(_) = self.output_layout.get(&operation.value) else {
            self.queue.push_back(operation.value);
            self.queue.push_back(key);
            return;
        };
        self.output_layout.insert(key, input_layout.clone());
    }

    fn visit_tensor(&mut self, graph: &super::ComputeGraphInner, key: NodeIndex) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Tensor(data) => data,
            _ => panic!("Expected Tensor node"),
        };
        let info = operation.info();
        self.output_layout.insert(key, info.clone());
    }

    fn visit_dequantize(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Dequantize(op) => op,
            _ => panic!("Expected Dequantize node"),
        };
        let matrix = &operation.matrix;
        let new_layout = Layout::contiguous(matrix.shape());
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, operation.datatype),
        );
    }

    fn visit_index_select(
        &mut self,
        graph: &super::ComputeGraphInner,
        key: NodeIndex,
    ) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::IndexSelect(op) => op,
            _ => panic!("Expected IndexSelect node"),
        };
        let Some(indexes_shape) = self.output_layout.get(&operation.indexes) else {
            self.queue.push_back(operation.indexes);
            self.queue.push_back(key);
            return;
        };
        let Some(input_shape) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let shape = IndexSelectOperation::calc_output_shape(
            operation.dimension,
            indexes_shape.shape(),
            input_shape.shape(),
        );
        let new_layout = Layout::contiguous(&shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, operation.datatype),
        );
    }

    fn visit_custom(&mut self, graph: &super::ComputeGraphInner, key: NodeIndex) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Custom(op) => op,
            _ => panic!("Expected Custom node"),
        };
        let mut dependencies = Vec::new();
        operation.visit_dependencies(&mut |dep| {
            dependencies.push(dep);
        });

        for dependency in dependencies {
            if !self.output_layout.contains_key(&dependency) {
                self.queue.push_back(dependency);
                self.queue.push_back(key);
                return;
            }
        }
        self.output_layout
            .insert(key, operation.output_layout(&self.output_layout));
    }
}