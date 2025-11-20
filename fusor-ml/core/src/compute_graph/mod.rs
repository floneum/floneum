use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak, atomic::AtomicUsize};

use dependency_map::{DependencyMap, visit_dependencies};
use parking_lot::RwLock;
use petgraph::csr::DefaultIx;
use petgraph::graph::NodeIndex;
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
    inner: Arc<RwLock<ComputeGraphInner>>,
}

impl ComputeGraph {
    pub(crate) fn new(device: Device) -> Self {
        let myself = Self {
            inner: Arc::new(RwLock::new(ComputeGraphInner::new(device))),
        };

        let weak = Arc::downgrade(&myself.inner);
        myself.with_mut(|inner| {
            inner.pointed_to_by.push(weak);
        });
        myself
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let mut inner = self.inner.write();
        let result = f(&mut inner);
        #[cfg(feature = "extra_assertions")]
        {
            inner.verify_integrity()
        }
        result
    }

    pub(crate) fn create_element_wise(&self, function: ElementWiseOperation) -> DefaultIx {
        let id = ElementWiseComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.element_wise.insert(id, function);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_pair_wise(&self, function: PairWiseOperation) -> DefaultIx {
        let id = PairWiseComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.pair_wise.insert(id, function);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_mat_mul(&self, function: MatMulOperation) -> MatMulComputeNodeKey {
        let id = MatMulComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.mat_mul.insert(id, function);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_q_mat_mul(&self, function: QMatMulOperation) -> QMatMulComputeNodeKey {
        let id = QMatMulComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.q_mat_mul.insert(id, function);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_reduce(&self, function: ReduceOperation) -> ReduceComputeNodeKey {
        let id = ReduceComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.reduce.insert(id, function);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_map_layout(&self, op: MapLayoutOperation) -> MapLayoutComputeNodeKey {
        let id = MapLayoutComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.map_layout.insert(id, op);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_resize(&self, op: ResizeOperation) -> ResizeComputeNodeKey {
        let id = ResizeComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.resize.insert(id, op);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_slice_assign(
        &self,
        op: SliceAssignOperation,
    ) -> SliceAssignComputeNodeKey {
        let id = SliceAssignComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.slice_assign.insert(id, op);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_index_select(
        &self,
        op: IndexSelectOperation,
    ) -> IndexSelectComputeNodeKey {
        let id = IndexSelectComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.index_select.insert(id, op);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_tensor(&self, info: TensorData) -> TensorComputeNodeKey {
        let id = TensorComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.tensor.insert(id, info);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn dequantize(&self, matrix: QMatrix, ty: DataTypeEnum) -> DequantizeComputeKey {
        let id = DequantizeComputeKey::new();
        self.with_mut(|inner| {
            inner
                .nodes
                .dequantize
                .insert(id, DequantizeOperation::new(matrix, ty));
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_custom(
        &self,
        operation: Arc<dyn Operation + Send + Sync>,
    ) -> CustomComputeKey {
        let id = CustomComputeKey::new();
        self.with_mut(|inner| {
            inner.nodes.custom.insert(id, operation);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn resolve(&self, key: NodeIndex, device: &Device) -> TensorData {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("ComputeGraph Encoder"),
            });
        let data = self.with_mut(|inner| Resolver::new(inner, key, &mut encoder).run());
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

    pub(crate) fn graphvis(&self, key: NodeIndex) -> Graph {
        self.with_mut(|inner| inner.graphvis(key))
    }
}

#[derive(Default)]
pub(crate) struct ComputeGraphNodes {
    pub(crate) nodes: StableGraph<ComputeGraphNode, ()>,
}

impl ComputeGraphNodes {
    fn insert_node(&mut self, node: ComputeGraphNode) -> NodeIndex {
        self.nodes.add_node(node)
    }
}

pub(crate) struct ComputeGraphNode {
    cached: Option<TensorData>,
    variant: ComputeGraphNodeVariant,
}

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
}

pub(crate) struct ComputeGraphInner {
    pub(crate) device: Device,
    pub(crate) nodes: ComputeGraphNodes,
}

impl ComputeGraphInner {
    fn new(device: Device) -> Self {
        Self {
            device,
            nodes: ComputeGraphNodes::default(),
        }
    }

    fn add_reference(&mut self, key: NodeIndex) {
        match self.dependency_map.reference_count.get_mut(&key) {
            Some(count) => {
                *count += 1;
            }
            None => {
                self.dependency_map.reference_count.insert(key, 1);
                self.dependency_map.add_dependents(key, &self.nodes);
            }
        }
    }

    fn remove_reference(&mut self, key: NodeIndex) {
        if let Some(count) = self.dependency_map.reference_count.get_mut(&key) {
            *count -= 1;
            // Remove the node if it is dead
            self.check_life(key);
        }
    }

    fn check_life(&mut self, key: NodeIndex) {
        if let Some(count) = self.dependency_map.reference_count.get(&key) {
            if *count > 0 {
                // The node still has references, so it is alive
                return;
            }
        } else {
            // The node is already dead
            return;
        }
        // Check if any of the nodes that depend on this key are alive
        if let Some(dependents) = self.dependency_map.dependant_map.get(&key) {
            for dependant in dependents {
                // If the dependant still exists and it hasn't been computed yet
                // keep this node alive
                let alive = self.dependency_map.reference_count.contains_key(dependant);
                let computed = self.cached_results.contains_key(dependant);
                if alive && !computed {
                    return;
                }
            }
        }

        let mut dependencies = Vec::new();
        visit_dependencies(&self.nodes, key, |dependency| {
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
        visit_dependencies(&self.nodes, key, |dependency| {
            if let Some(map) = self.dependency_map.dependant_map.get_mut(&dependency) {
                map.remove(&key);
            }
        });
        self.nodes.nodes.remove_node(key);
    }

    pub(crate) fn get_result_or_qmatrix(&self, key: NodeIndex) -> Option<MaybeQData> {
        let result = if let NodeIndex::Dequantize(key) = key {
            let tensor = &self.nodes.dequantize.get(&key)?;
            tensor.matrix.clone().into()
        } else {
            self.cached_results.get(&key)?.clone().into()
        };
        Some(result)
    }

    pub(crate) fn get_result(&self, key: NodeIndex) -> Option<TensorData> {
        self.cached_results.get(&key).cloned()
    }

    #[cfg(feature = "extra_assertions")]
    fn contains_key(&self, key: NodeIndex) -> bool {
        if self.cached_results.contains_key(&key) {
            return true;
        }
        match key {
            NodeIndex::ElementWise(key) => self.nodes.element_wise.contains_key(&key),
            NodeIndex::PairWise(key) => self.nodes.pair_wise.contains_key(&key),
            NodeIndex::MatMul(key) => self.nodes.mat_mul.contains_key(&key),
            NodeIndex::Reduce(key) => self.nodes.reduce.contains_key(&key),
            NodeIndex::MapLayout(key) => self.nodes.map_layout.contains_key(&key),
            NodeIndex::Resize(key) => self.nodes.resize.contains_key(&key),
            NodeIndex::SliceAssign(key) => self.nodes.slice_assign.contains_key(&key),
            NodeIndex::IndexSelect(key) => self.nodes.index_select.contains_key(&key),
            NodeIndex::Tensor(key) => self.nodes.tensor.contains_key(&key),
            NodeIndex::QMatMul(key) => self.nodes.q_mat_mul.contains_key(&key),
            NodeIndex::Dequantize(key) => self.nodes.dequantize.contains_key(&key),
            NodeIndex::Custom(key) => self.nodes.custom.contains_key(&key),
        }
    }

    #[cfg(feature = "extra_assertions")]
    fn verify_integrity(&self) {
        // Check that all node references exist in the graph
        for key in self.dependency_map.reference_count.keys() {
            assert!(
                self.contains_key(*key),
                "{key:?} does not exist in the reference map"
            );
        }
        for (key, dependants) in self.dependency_map.dependant_map.iter() {
            assert!(
                self.contains_key(*key),
                "{key:?} is in the dependant map, but it doesn't exist"
            );
            for dependant in dependants {
                assert!(
                    self.contains_key(*dependant),
                    "the dependant {dependant:?} of {key:?} does not exist"
                );
            }
        }

        let keys = self
            .nodes
            .element_wise
            .keys()
            .copied()
            .map(NodeIndex::from)
            .chain(self.nodes.pair_wise.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.mat_mul.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.reduce.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.map_layout.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.resize.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.slice_assign.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.index_select.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.tensor.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.q_mat_mul.keys().copied().map(NodeIndex::from))
            .chain(self.nodes.dequantize.keys().copied().map(NodeIndex::from));

        for key in keys {
            if self.cached_results.contains_key(&key) {
                continue;
            }
            visit_dependencies(&self.nodes, key, |dependency| {
                assert!(self.contains_key(dependency));
            });
        }
    }
}
