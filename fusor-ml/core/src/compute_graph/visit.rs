use super::{
    AnyComputeKey, ComputeGraphInner, ElementWiseComputeNodeKey, MapLayoutComputeNodeKey,
    MatMulComputeNodeKey, PairWiseComputeNodeKey, ReduceComputeNodeKey, ResizeComputeNodeKey,
    SliceAssignComputeNodeKey, TensorComputeNodeKey,
};

pub(crate) trait VisitComputeGraph: Sized {
    fn visit(&mut self, graph: &ComputeGraphInner, key: AnyComputeKey) {
        match key {
            AnyComputeKey::ElementWiseComputeNodeKey(element_wise_compute_node_key) => {
                self.visit_element_wise(graph, element_wise_compute_node_key)
            }
            AnyComputeKey::PairWiseComputeNodeKey(pair_wise_compute_node_key) => {
                self.visit_pair_wise(graph, pair_wise_compute_node_key)
            }
            AnyComputeKey::MatMulComputeNodeKey(mat_mul_compute_node_key) => {
                self.visit_mat_mul(graph, mat_mul_compute_node_key);
            }
            AnyComputeKey::ReduceComputeNodeKey(reduce_compute_node_key) => {
                self.visit_reduce(graph, reduce_compute_node_key);
            }
            AnyComputeKey::MapLayoutComputeNodeKey(slice_compute_node_key) => {
                self.visit_slice(graph, slice_compute_node_key);
            }
            AnyComputeKey::ResizeComputeNodeKey(resize_compute_node_key) => {
                self.visit_resize(graph, resize_compute_node_key);
            }
            AnyComputeKey::SliceAssignComputeNodeKey(slice_assign_compute_node_key) => {
                self.visit_slice_assign(graph, slice_assign_compute_node_key);
            }
            AnyComputeKey::TensorComputeNodeKey(tensor_compute_node_key) => {
                self.visit_tensor(graph, tensor_compute_node_key);
            }
        }
    }

    fn visit_element_wise(&mut self, graph: &ComputeGraphInner, key: ElementWiseComputeNodeKey) {
        visit_element_wise(self, graph, key);
    }

    fn visit_pair_wise(&mut self, graph: &ComputeGraphInner, key: PairWiseComputeNodeKey) {
        visit_pair_wise(self, graph, key);
    }

    fn visit_mat_mul(&mut self, graph: &ComputeGraphInner, key: MatMulComputeNodeKey) {
        visit_mat_mul(self, graph, key);
    }

    fn visit_reduce(&mut self, graph: &ComputeGraphInner, key: ReduceComputeNodeKey) {
        visit_reduce(self, graph, key);
    }

    fn visit_slice(&mut self, graph: &ComputeGraphInner, key: MapLayoutComputeNodeKey) {
        visit_slice(self, graph, key);
    }

    fn visit_resize(&mut self, graph: &ComputeGraphInner, key: ResizeComputeNodeKey) {
        visit_resize(self, graph, key);
    }

    fn visit_slice_assign(&mut self, graph: &ComputeGraphInner, key: SliceAssignComputeNodeKey) {
        visit_slice_assign(self, graph, key);
    }

    fn visit_tensor(&mut self, graph: &ComputeGraphInner, key: TensorComputeNodeKey) {
        visit_tensor(self, graph, key);
    }
}

pub(crate) fn visit_element_wise(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: ElementWiseComputeNodeKey,
) {
    let operation = graph.element_wise.get(&key).unwrap();
    let input = operation.value;
    visitor.visit(graph, input);
}

pub(crate) fn visit_pair_wise(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: PairWiseComputeNodeKey,
) {
    let operation = graph.pair_wise.get(&key).unwrap();
    let first = operation.first;
    let second = operation.second;
    visitor.visit(graph, first);
    visitor.visit(graph, second);
}

pub(crate) fn visit_mat_mul(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: MatMulComputeNodeKey,
) {
    let operation = graph.mat_mul.get(&key).unwrap();
    let first = operation.first;
    let second = operation.second;
    visitor.visit(graph, first);
    visitor.visit(graph, second);
}

pub(crate) fn visit_reduce(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: ReduceComputeNodeKey,
) {
    let operation = graph.reduce.get(&key).unwrap();
    let value = operation.value;
    visitor.visit(graph, value);
}

pub(crate) fn visit_slice(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: MapLayoutComputeNodeKey,
) {
    let operation = graph.map_layout.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
}

pub(crate) fn visit_resize(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: ResizeComputeNodeKey,
) {
    let operation = graph.resize.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
}

pub(crate) fn visit_slice_assign(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphInner,
    key: SliceAssignComputeNodeKey,
) {
    let operation = graph.slice_assign.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
    let value = operation.value;
    visitor.visit(graph, value);
}

pub(crate) fn visit_tensor(
    _: &mut impl VisitComputeGraph,
    _: &ComputeGraphInner,
    _: TensorComputeNodeKey,
) {
}
