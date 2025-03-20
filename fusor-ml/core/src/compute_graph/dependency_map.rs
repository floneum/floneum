use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    AnyComputeKey, ComputeGraphNodes, ElementWiseComputeNodeKey, MapLayoutComputeNodeKey,
    MatMulComputeNodeKey, PairWiseComputeNodeKey, QMatMulComputeNodeKey, ReduceComputeNodeKey,
    ResizeComputeNodeKey, SliceAssignComputeNodeKey,
};

#[derive(Default)]
pub(crate) struct DependencyMap {
    // The items that are dependent on the key
    pub(crate) dependant_map: FxHashMap<AnyComputeKey, FxHashSet<AnyComputeKey>>,
    pub(crate) reference_count: FxHashMap<AnyComputeKey, usize>,
}

impl DependencyMap {
    pub(crate) fn add_dependents(&mut self, key: AnyComputeKey, graph: &ComputeGraphNodes) {
        visit_dependencies(graph, key, |dependent| {
            self.dependant_map
                .entry(dependent)
                .or_default()
                .insert(key);
        });
    }
}

pub(crate) fn visit_dependencies(
    graph: &ComputeGraphNodes,
    key: AnyComputeKey,
    f: impl FnMut(AnyComputeKey),
) {
    fn add_dependents_generic<const N: usize, T: Into<AnyComputeKey> + Copy>(
        graph: &ComputeGraphNodes,
        key: T,
        get_dependencies: fn(&ComputeGraphNodes, T) -> [AnyComputeKey; N],
        mut f: impl FnMut(AnyComputeKey),
    ) {
        let dependencies = get_dependencies(graph, key);
        for dependency in dependencies {
            f(dependency);
        }
    }

    match key {
        AnyComputeKey::ElementWise(element_wise_compute_node_key) => add_dependents_generic(
            graph,
            element_wise_compute_node_key,
            element_wise_dependencies,
            f,
        ),
        AnyComputeKey::PairWise(pair_wise_compute_node_key) => {
            add_dependents_generic(graph, pair_wise_compute_node_key, pair_wise_dependencies, f)
        }
        AnyComputeKey::MatMul(mat_mul_compute_node_key) => {
            add_dependents_generic(graph, mat_mul_compute_node_key, mat_mul_dependencies, f)
        }
        AnyComputeKey::Reduce(reduce_compute_node_key) => {
            add_dependents_generic(graph, reduce_compute_node_key, reduce_dependencies, f)
        }
        AnyComputeKey::MapLayout(map_layout_compute_node_key) => add_dependents_generic(
            graph,
            map_layout_compute_node_key,
            map_layout_dependencies,
            f,
        ),
        AnyComputeKey::Resize(resize_compute_node_key) => {
            add_dependents_generic(graph, resize_compute_node_key, resize_dependencies, f)
        }
        AnyComputeKey::SliceAssign(slice_assign_compute_node_key) => add_dependents_generic(
            graph,
            slice_assign_compute_node_key,
            slice_assign_dependencies,
            f,
        ),
        AnyComputeKey::QMatMul(q_mat_mul_compute_node_key) => {
            add_dependents_generic(graph, q_mat_mul_compute_node_key, q_mat_mul_dependencies, f)
        }
        AnyComputeKey::Tensor(_) => {}
    }
}

pub(crate) fn element_wise_dependencies(
    graph: &ComputeGraphNodes,
    key: ElementWiseComputeNodeKey,
) -> [AnyComputeKey; 1] {
    let operation = graph.element_wise.get(&key).unwrap();
    let value = operation.value;
    [value]
}

pub(crate) fn pair_wise_dependencies(
    graph: &ComputeGraphNodes,
    key: PairWiseComputeNodeKey,
) -> [AnyComputeKey; 2] {
    let operation = graph.pair_wise.get(&key).unwrap();
    let first = operation.first;
    let second = operation.second;
    [first, second]
}

pub(crate) fn mat_mul_dependencies(
    graph: &ComputeGraphNodes,
    key: MatMulComputeNodeKey,
) -> [AnyComputeKey; 2] {
    let operation = graph.mat_mul.get(&key).unwrap();
    let first = operation.first;
    let second = operation.second;

    [first, second]
}

pub(crate) fn reduce_dependencies(
    graph: &ComputeGraphNodes,
    key: ReduceComputeNodeKey,
) -> [AnyComputeKey; 1] {
    let operation = graph.reduce.get(&key).unwrap();
    let value = operation.value;
    [value]
}

pub(crate) fn q_mat_mul_dependencies(
    graph: &ComputeGraphNodes,
    key: QMatMulComputeNodeKey,
) -> [AnyComputeKey; 1] {
    let operation = graph.q_mat_mul.get(&key).unwrap();
    let input = operation.input;
    [input]
}

pub(crate) fn resize_dependencies(
    graph: &ComputeGraphNodes,
    key: ResizeComputeNodeKey,
) -> [AnyComputeKey; 1] {
    let operation = graph.resize.get(&key).unwrap();
    let input = operation.input;
    [input]
}

pub(crate) fn map_layout_dependencies(
    graph: &ComputeGraphNodes,
    key: MapLayoutComputeNodeKey,
) -> [AnyComputeKey; 1] {
    let operation = graph.map_layout.get(&key).unwrap();
    let input = operation.input;
    [input]
}

pub(crate) fn slice_assign_dependencies(
    graph: &ComputeGraphNodes,
    key: SliceAssignComputeNodeKey,
) -> [AnyComputeKey; 2] {
    let operation = graph.slice_assign.get(&key).unwrap();
    let input = operation.input;
    let value = operation.value;
    [input, value]
}
