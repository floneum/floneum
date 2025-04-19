use super::{
    AnyComputeKey, ComputeGraphNodes, DequantizeComputeKey, ElementWiseComputeNodeKey,
    IndexSelectComputeNodeKey, MapLayoutComputeNodeKey, MatMulComputeNodeKey,
    PairWiseComputeNodeKey, QMatMulComputeNodeKey, ReduceComputeNodeKey, ResizeComputeNodeKey,
    SliceAssignComputeNodeKey, TensorComputeNodeKey,
};

pub(crate) trait VisitComputeGraph: Sized {
    fn visit(&mut self, graph: &ComputeGraphNodes, key: AnyComputeKey) {
        visit(self, graph, key);
    }

    fn visit_element_wise(&mut self, graph: &ComputeGraphNodes, key: ElementWiseComputeNodeKey) {
        visit_element_wise(self, graph, key);
    }

    fn visit_pair_wise(&mut self, graph: &ComputeGraphNodes, key: PairWiseComputeNodeKey) {
        visit_pair_wise(self, graph, key);
    }

    fn visit_mat_mul(&mut self, graph: &ComputeGraphNodes, key: MatMulComputeNodeKey) {
        visit_mat_mul(self, graph, key);
    }

    fn visit_q_mat_mul(&mut self, graph: &ComputeGraphNodes, key: QMatMulComputeNodeKey) {
        visit_q_mat_mul(self, graph, key);
    }

    fn visit_reduce(&mut self, graph: &ComputeGraphNodes, key: ReduceComputeNodeKey) {
        visit_reduce(self, graph, key);
    }

    fn visit_map_layout(&mut self, graph: &ComputeGraphNodes, key: MapLayoutComputeNodeKey) {
        visit_map_layout(self, graph, key);
    }

    fn visit_resize(&mut self, graph: &ComputeGraphNodes, key: ResizeComputeNodeKey) {
        visit_resize(self, graph, key);
    }

    fn visit_slice_assign(&mut self, graph: &ComputeGraphNodes, key: SliceAssignComputeNodeKey) {
        visit_slice_assign(self, graph, key);
    }

    fn visit_index_select(&mut self, graph: &ComputeGraphNodes, key: IndexSelectComputeNodeKey) {
        visit_index_select(self, graph, key);
    }

    fn visit_tensor(&mut self, graph: &ComputeGraphNodes, key: TensorComputeNodeKey) {
        visit_tensor(self, graph, key);
    }

    fn visit_dequantize(&mut self, graph: &ComputeGraphNodes, key: DequantizeComputeKey) {
        visit_dequantize(self, graph, key);
    }
}

pub(crate) fn visit(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: AnyComputeKey,
) {
    match key {
        AnyComputeKey::ElementWise(element_wise_compute_node_key) => {
            visitor.visit_element_wise(graph, element_wise_compute_node_key)
        }
        AnyComputeKey::PairWise(pair_wise_compute_node_key) => {
            visitor.visit_pair_wise(graph, pair_wise_compute_node_key)
        }
        AnyComputeKey::MatMul(mat_mul_compute_node_key) => {
            visitor.visit_mat_mul(graph, mat_mul_compute_node_key);
        }
        AnyComputeKey::QMatMul(mat_mul_compute_node_key) => {
            visitor.visit_q_mat_mul(graph, mat_mul_compute_node_key);
        }
        AnyComputeKey::Reduce(reduce_compute_node_key) => {
            visitor.visit_reduce(graph, reduce_compute_node_key);
        }
        AnyComputeKey::MapLayout(slice_compute_node_key) => {
            visitor.visit_map_layout(graph, slice_compute_node_key);
        }
        AnyComputeKey::Resize(resize_compute_node_key) => {
            visitor.visit_resize(graph, resize_compute_node_key);
        }
        AnyComputeKey::SliceAssign(slice_assign_compute_node_key) => {
            visitor.visit_slice_assign(graph, slice_assign_compute_node_key);
        }
        AnyComputeKey::IndexSelect(index_select_compute_node_key) => {
            visitor.visit_index_select(graph, index_select_compute_node_key);
        }
        AnyComputeKey::Tensor(tensor_compute_node_key) => {
            visitor.visit_tensor(graph, tensor_compute_node_key);
        }
        AnyComputeKey::Dequantize(dequantize_compute_node_key) => {
            visitor.visit_dequantize(graph, dequantize_compute_node_key);
        }
    }
}

pub(crate) fn visit_element_wise(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: ElementWiseComputeNodeKey,
) {
    let operation = graph.element_wise.get(&key).unwrap();
    let input = operation.value;
    visitor.visit(graph, input);
}

pub(crate) fn visit_pair_wise(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
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
    graph: &ComputeGraphNodes,
    key: MatMulComputeNodeKey,
) {
    let operation = graph.mat_mul.get(&key).unwrap();
    let first = operation.first;
    let second = operation.second;
    visitor.visit(graph, first);
    visitor.visit(graph, second);
}

pub(crate) fn visit_q_mat_mul(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: QMatMulComputeNodeKey,
) {
    let operation = graph.q_mat_mul.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
}

pub(crate) fn visit_reduce(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: ReduceComputeNodeKey,
) {
    let operation = graph.reduce.get(&key).unwrap();
    let value = operation.value;
    visitor.visit(graph, value);
}

pub(crate) fn visit_map_layout(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: MapLayoutComputeNodeKey,
) {
    let operation = graph.map_layout.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
}

pub(crate) fn visit_resize(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: ResizeComputeNodeKey,
) {
    let operation = graph.resize.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
}

pub(crate) fn visit_slice_assign(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: SliceAssignComputeNodeKey,
) {
    let operation = graph.slice_assign.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
    let value = operation.value;
    visitor.visit(graph, value);
}

pub(crate) fn visit_index_select(
    visitor: &mut impl VisitComputeGraph,
    graph: &ComputeGraphNodes,
    key: IndexSelectComputeNodeKey,
) {
    let operation = graph.index_select.get(&key).unwrap();
    let input = operation.input;
    visitor.visit(graph, input);
    let index = operation.indexes;
    visitor.visit(graph, index);
}

pub(crate) fn visit_tensor(
    _: &mut impl VisitComputeGraph,
    _: &ComputeGraphNodes,
    _: TensorComputeNodeKey,
) {
}

pub(crate) fn visit_dequantize(
    _: &mut impl VisitComputeGraph,
    _: &ComputeGraphNodes,
    _: DequantizeComputeKey,
) {
}
