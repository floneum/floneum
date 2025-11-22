use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;
pub(crate) use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use resolve::Resolver;
use rustc_hash::FxHashMap;
use tabbycat::Graph;
use wgpu::CommandEncoderDescriptor;

mod layout_pass;
mod queue;
mod resolve;
mod visualize;

use crate::{
    DataTypeEnum, Device, ElementWiseOperation, MatMulOperation, PairWiseOperation, QMatrix,
    ReduceOperation, dequantize::DequantizeOperation, index_select::IndexSelectOperation,
    map_layout::MapLayoutOperation, mir::operation::Operation, quantized::matmul::QMatMulOperation,
    resize::ResizeOperation, slice_assign::SliceAssignOperation, tensor::TensorData,
    visit_tiled::MaybeQData,
};

#[derive(Clone)]
pub(crate) struct ComputeGraph {
    inner: Arc<ArcSwap<Arc<RwLock<ComputeGraphInner>>>>,
}

impl ComputeGraph {
    pub(crate) fn new(device: Device) -> Self {
        let inner_arc = Arc::new(RwLock::new(ComputeGraphInner::new(device)));
        Self {
            inner: Arc::new(ArcSwap::from_pointee(inner_arc)),
        }
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let inner_arc = self.inner.load_full();
        let mut inner = inner_arc.write();
        let result = f(&mut inner);
        #[cfg(feature = "extra_assertions")]
        {
            inner.verify_integrity()
        }
        result
    }

    pub(crate) fn create_element_wise(&self, function: ElementWiseOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::ElementWise(function),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_pair_wise(&self, function: PairWiseOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::PairWise(function),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_mat_mul(&self, function: MatMulOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::MatMul(function),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_q_mat_mul(&self, function: QMatMulOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::QMatMul(function),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_reduce(&self, function: ReduceOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::Reduce(function),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_map_layout(&self, op: MapLayoutOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::MapLayout(op),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_resize(&self, op: ResizeOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::Resize(op),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_slice_assign(&self, op: SliceAssignOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::SliceAssign(op),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_index_select(&self, op: IndexSelectOperation) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::IndexSelect(op),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_tensor(&self, info: TensorData) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::Tensor(info),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn dequantize(&self, matrix: QMatrix, ty: DataTypeEnum) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::Dequantize(DequantizeOperation::new(matrix, ty)),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn create_custom(&self, operation: Arc<dyn Operation + Send + Sync>) -> NodeIndex {
        self.with_mut(|inner| {
            let node = ComputeGraphNode {
                variant: ComputeGraphNodeVariant::Custom(operation),
            };
            let id = inner.nodes.nodes.add_node(node);
            inner.add_reference(id);
            id
        })
    }

    pub(crate) fn resolve(&self, key: NodeIndex, device: &Device) -> TensorData {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("ComputeGraph Encoder"),
            });
        let data = self.with_mut(|inner| {
            let mut resolver = Resolver::new(inner, key, &mut encoder);
            resolver.run(inner)
        });
        device.wgpu_queue().submit(Some(encoder.finish()));
        // Reset the written flag on all buffers
        device.reset_initialized_buffers();

        // Flush the cache to a file
        if let (Some(pipeline_cache), Some(cache_file)) =
            (device.wgpu_cache(), device.wgpu_cache_file())
        {
            let data = pipeline_cache.get_data();
            if let Some(data) = data {
                let temp_file = cache_file.with_extension("temp");
                std::fs::write(&temp_file, &data).unwrap();
                std::fs::rename(&temp_file, cache_file).unwrap();
            }
        }

        data
    }

    pub(crate) fn graphvis(&self, root: NodeIndex) -> Graph {
        self.with_mut(|inner| inner.graphvis(root))
    }

    pub(crate) fn add_reference(&self, key: NodeIndex) {
        self.with_mut(|inner| inner.add_reference(key));
    }

    pub(crate) fn remove_reference(&self, key: NodeIndex) {
        self.with_mut(|inner| inner.remove_reference(key));
    }
}

#[derive(Default)]
pub(crate) struct ComputeGraphNodes {
    pub(crate) nodes: StableGraph<ComputeGraphNode, ()>,
}

impl ComputeGraphNodes {
}

pub(crate) struct ComputeGraphNode {
    variant: ComputeGraphNodeVariant,
}

#[derive(Clone)]
pub(crate) enum ComputeGraphNodeVariant {
    ElementWise(ElementWiseOperation),
    PairWise(PairWiseOperation),
    SliceAssign(SliceAssignOperation),
    Resize(ResizeOperation),
    MapLayout(MapLayoutOperation),
    Dequantize(DequantizeOperation),
    MatMul(MatMulOperation),
    QMatMul(QMatMulOperation),
    Tensor(TensorData),
    Reduce(ReduceOperation),
    IndexSelect(IndexSelectOperation),
    Custom(Arc<dyn Operation + Send + Sync>),
}

pub(crate) struct ComputeGraphInner {
    pub(crate) device: Device,
    pub(crate) nodes: ComputeGraphNodes,
    pub(crate) cached_results: FxHashMap<NodeIndex, TensorData>,
    reference_count: FxHashMap<NodeIndex, usize>,
}

impl ComputeGraphInner {
    fn new(device: Device) -> Self {
        Self {
            device,
            nodes: ComputeGraphNodes::default(),
            cached_results: FxHashMap::default(),
            reference_count: FxHashMap::default(),
        }
    }

    fn add_reference(&mut self, key: NodeIndex) {
        match self.reference_count.get_mut(&key) {
            Some(count) => {
                *count += 1;
            }
            None => {
                self.reference_count.insert(key, 1);
                // Add edges from dependencies to this node
                self.add_dependency_edges(key);
            }
        }
    }

    fn add_dependency_edges(&mut self, key: NodeIndex) {
        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dep| {
            dependencies.push(dep);
        });
        for dep in dependencies {
            self.nodes.nodes.add_edge(dep, key, ());
        }
    }

    fn visit_dependencies(&self, key: NodeIndex, f: &mut dyn FnMut(NodeIndex)) {
        if let Some(node) = self.nodes.nodes.node_weight(key) {
            match &node.variant {
                ComputeGraphNodeVariant::ElementWise(op) => f(op.value),
                ComputeGraphNodeVariant::PairWise(op) => {
                    f(op.first);
                    f(op.second);
                }
                ComputeGraphNodeVariant::MatMul(op) => {
                    f(op.first);
                    f(op.second);
                }
                ComputeGraphNodeVariant::QMatMul(op) => {
                    f(op.input);
                }
                ComputeGraphNodeVariant::Reduce(op) => f(op.value),
                ComputeGraphNodeVariant::MapLayout(op) => f(op.input),
                ComputeGraphNodeVariant::Resize(op) => f(op.input),
                ComputeGraphNodeVariant::SliceAssign(op) => {
                    f(op.input);
                    f(op.value);
                }
                ComputeGraphNodeVariant::IndexSelect(op) => {
                    f(op.input);
                    f(op.indexes);
                }
                ComputeGraphNodeVariant::Dequantize(_) => {}
                ComputeGraphNodeVariant::Tensor(_) => {}
                ComputeGraphNodeVariant::Custom(op) => {
                    op.visit_dependencies(f);
                }
            }
        }
    }

    fn remove_reference(&mut self, key: NodeIndex) {
        if let Some(count) = self.reference_count.get_mut(&key)
            && *count > 0
        {
            *count -= 1;
            // Remove the node if it is dead
            self.check_life(key);
        }
    }

    fn check_life(&mut self, key: NodeIndex) {
        if let Some(count) = self.reference_count.get(&key) {
            if *count > 0 {
                // The node still has references, so it is alive
                return;
            }
        } else {
            // The node is already dead
            return;
        }

        // Check if any of the nodes that depend on this key are alive
        let dependents: Vec<_> = self.nodes.nodes
            .neighbors_directed(key, petgraph::Direction::Outgoing)
            .collect();

        for dependant in dependents {
            // If the dependant still exists and it hasn't been computed yet
            // keep this node alive
            let alive = self.reference_count.contains_key(&dependant);
            let computed = self.cached_results.contains_key(&dependant);
            if alive && !computed {
                return;
            }
        }

        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dependency| {
            dependencies.push(dependency);
        });

        // If no other nodes depend on this key and it has zero references, it is dead
        // remove it from the graph
        self.remove_key(key);

        // Then check if any nodes it depends on are alive
        for dependency in dependencies {
            self.check_life(dependency);
        }
    }

    fn remove_key(&mut self, key: NodeIndex) {
        // Remove the cached result if it exists
        self.cached_results.remove(&key);
        self.reference_count.remove(&key);
        // Remove the node from the graph (this also removes all edges)
        self.nodes.nodes.remove_node(key);
    }

    pub(crate) fn get_result_or_qmatrix(&self, key: NodeIndex) -> Option<MaybeQData> {
        // Check if this is a Dequantize node
        if let Some(node) = self.nodes.nodes.node_weight(key)
            && let ComputeGraphNodeVariant::Dequantize(op) = &node.variant
        {
            return Some(op.matrix.clone().into());
        }
        // Otherwise, get from cached results
        self.cached_results.get(&key).map(|t| t.clone().into())
    }

    pub(crate) fn get_result(&self, key: NodeIndex) -> Option<TensorData> {
        self.cached_results.get(&key).cloned()
    }

    #[cfg(feature = "extra_assertions")]
    fn contains_key(&self, key: NodeIndex) -> bool {
        self.cached_results.contains_key(&key) || self.nodes.nodes.contains_node(key)
    }

    #[cfg(feature = "extra_assertions")]
    fn verify_integrity(&self) {
        // Check that all node references exist in the graph
        for key in self.reference_count.keys() {
            assert!(
                self.contains_key(*key),
                "{key:?} does not exist in the reference map"
            );
        }

        // Check that all edges point to existing nodes
        for key in self.nodes.nodes.node_indices() {
            for neighbor in self.nodes.nodes.neighbors(key) {
                assert!(
                    self.nodes.nodes.contains_node(neighbor),
                    "edge points to non-existent node {neighbor:?}"
                );
            }
        }

        // Check that all dependencies of non-cached nodes exist
        for key in self.nodes.nodes.node_indices() {
            if self.cached_results.contains_key(&key) {
                continue;
            }
            self.visit_dependencies(key, &mut |dependency| {
                assert!(
                    self.contains_key(dependency),
                    "dependency {dependency:?} of {key:?} does not exist"
                );
            });
        }
    }
}
