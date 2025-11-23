use rustc_hash::FxHashMap;

use crate::{Layout, TensorLayoutInfo, index_select::IndexSelectOperation};

use super::{ComputeGraphNodeVariant, NodeIndex, queue::ComputeQueue};

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
            let node_data = graph.nodes.nodes.node_weight(node).expect("Node not found");
            if let Some(resolved) = &node_data.cached {
                self.output_layout.insert(node, resolved.info().clone());
                continue;
            }
            match &node_data.variant {
                ComputeGraphNodeVariant::ElementWise(op) => self.visit_element_wise(node, op),
                ComputeGraphNodeVariant::PairWise(op) => self.visit_pair_wise(node, op),
                ComputeGraphNodeVariant::MatMul(op) => self.visit_mat_mul(node, op),
                ComputeGraphNodeVariant::QMatMul(op) => self.visit_q_mat_mul(node, op),
                ComputeGraphNodeVariant::Reduce(op) => self.visit_reduce(node, op),
                ComputeGraphNodeVariant::MapLayout(op) => self.visit_map_layout(node, op),
                ComputeGraphNodeVariant::Resize(op) => self.visit_resize(node, op),
                ComputeGraphNodeVariant::SliceAssign(op) => self.visit_slice_assign(node, op),
                ComputeGraphNodeVariant::Tensor(op) => self.visit_tensor(node, op),
                ComputeGraphNodeVariant::Dequantize(op) => self.visit_dequantize(node, op),
                ComputeGraphNodeVariant::IndexSelect(op) => self.visit_index_select(node, op),
                ComputeGraphNodeVariant::Custom(op) => self.visit_custom(node, op),
            }
        }
    }

    fn visit_element_wise(&mut self, key: NodeIndex, operation: &crate::ElementWiseOperation) {
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

    fn visit_pair_wise(&mut self, key: NodeIndex, operation: &crate::PairWiseOperation) {
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

    fn visit_mat_mul(&mut self, key: NodeIndex, operation: &crate::MatMulOperation) {
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
        key: NodeIndex,
        operation: &crate::quantized::matmul::QMatMulOperation,
    ) {
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

    fn visit_reduce(&mut self, key: NodeIndex, operation: &crate::ReduceOperation) {
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
        key: NodeIndex,
        operation: &crate::map_layout::MapLayoutOperation,
    ) {
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

    fn visit_resize(&mut self, key: NodeIndex, operation: &crate::resize::ResizeOperation) {
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
        key: NodeIndex,
        operation: &crate::slice_assign::SliceAssignOperation,
    ) {
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

    fn visit_tensor(&mut self, key: NodeIndex, operation: &crate::tensor::TensorData) {
        let info = operation.info();
        self.output_layout.insert(key, info.clone());
    }

    fn visit_dequantize(
        &mut self,
        key: NodeIndex,
        operation: &crate::dequantize::DequantizeOperation,
    ) {
        let matrix = &operation.matrix;
        let new_layout = Layout::contiguous(matrix.shape());
        self.output_layout
            .insert(key, TensorLayoutInfo::new(new_layout, operation.datatype));
    }

    fn visit_index_select(
        &mut self,
        key: NodeIndex,
        operation: &crate::index_select::IndexSelectOperation,
    ) {
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
        self.output_layout
            .insert(key, TensorLayoutInfo::new(new_layout, operation.datatype));
    }

    fn visit_custom(
        &mut self,
        key: NodeIndex,
        operation: &std::sync::Arc<dyn crate::mir::operation::Operation + Send + Sync>,
    ) {
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
