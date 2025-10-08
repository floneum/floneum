use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak, atomic::AtomicUsize};

use arc_swap::{ArcSwap, ArcSwapAny};
use dependency_map::{DependencyMap, visit_dependencies};
use parking_lot::RwLock;
use resolve::Resolver;
use rustc_hash::FxHashMap;
use tabbycat::Graph;
use wgpu::CommandEncoderDescriptor;

mod dependency_map;
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ElementWiseComputeNodeKey(usize);
impl ElementWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PairWiseComputeNodeKey(usize);
impl PairWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MatMulComputeNodeKey(usize);
impl MatMulComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ReduceComputeNodeKey(usize);
impl ReduceComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MapLayoutComputeNodeKey(usize);
impl MapLayoutComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ResizeComputeNodeKey(usize);
impl ResizeComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SliceAssignComputeNodeKey(usize);
impl SliceAssignComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct IndexSelectComputeNodeKey(usize);
impl IndexSelectComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct TensorComputeNodeKey(usize);
impl TensorComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DequantizeComputeKey(usize);
impl DequantizeComputeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CustomComputeKey(usize);
impl CustomComputeKey {
    pub(crate) fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct QMatMulComputeNodeKey(usize);
impl QMatMulComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AnyComputeKey {
    ElementWise(ElementWiseComputeNodeKey),
    PairWise(PairWiseComputeNodeKey),
    MatMul(MatMulComputeNodeKey),
    Reduce(ReduceComputeNodeKey),
    MapLayout(MapLayoutComputeNodeKey),
    Resize(ResizeComputeNodeKey),
    SliceAssign(SliceAssignComputeNodeKey),
    IndexSelect(IndexSelectComputeNodeKey),
    Tensor(TensorComputeNodeKey),
    QMatMul(QMatMulComputeNodeKey),
    Dequantize(DequantizeComputeKey),
    Custom(CustomComputeKey),
}

impl From<ElementWiseComputeNodeKey> for AnyComputeKey {
    fn from(value: ElementWiseComputeNodeKey) -> Self {
        Self::ElementWise(value)
    }
}

impl From<PairWiseComputeNodeKey> for AnyComputeKey {
    fn from(value: PairWiseComputeNodeKey) -> Self {
        Self::PairWise(value)
    }
}

impl From<MatMulComputeNodeKey> for AnyComputeKey {
    fn from(value: MatMulComputeNodeKey) -> Self {
        Self::MatMul(value)
    }
}

impl From<ReduceComputeNodeKey> for AnyComputeKey {
    fn from(value: ReduceComputeNodeKey) -> Self {
        Self::Reduce(value)
    }
}

impl From<TensorComputeNodeKey> for AnyComputeKey {
    fn from(value: TensorComputeNodeKey) -> Self {
        Self::Tensor(value)
    }
}

impl From<MapLayoutComputeNodeKey> for AnyComputeKey {
    fn from(value: MapLayoutComputeNodeKey) -> Self {
        Self::MapLayout(value)
    }
}

impl From<ResizeComputeNodeKey> for AnyComputeKey {
    fn from(value: ResizeComputeNodeKey) -> Self {
        Self::Resize(value)
    }
}

impl From<SliceAssignComputeNodeKey> for AnyComputeKey {
    fn from(value: SliceAssignComputeNodeKey) -> Self {
        Self::SliceAssign(value)
    }
}

impl From<IndexSelectComputeNodeKey> for AnyComputeKey {
    fn from(value: IndexSelectComputeNodeKey) -> Self {
        Self::IndexSelect(value)
    }
}

impl From<QMatMulComputeNodeKey> for AnyComputeKey {
    fn from(value: QMatMulComputeNodeKey) -> Self {
        Self::QMatMul(value)
    }
}

impl From<DequantizeComputeKey> for AnyComputeKey {
    fn from(value: DequantizeComputeKey) -> Self {
        Self::Dequantize(value)
    }
}

impl From<CustomComputeKey> for AnyComputeKey {
    fn from(value: CustomComputeKey) -> Self {
        Self::Custom(value)
    }
}

#[derive(Clone)]
pub(crate) struct ComputeGraph {
    inner: Arc<ArcSwap<RwLock<ComputeGraphInner>>>,
}

impl ComputeGraph {
    pub(crate) fn new(device: Device) -> Self {
        let myself = Self {
            inner: Arc::new(ArcSwap::new(Arc::new(RwLock::new(ComputeGraphInner::new(
                device,
            ))))),
        };

        let weak = Arc::downgrade(&myself.inner);
        myself.with_mut(|inner| {
            inner.pointed_to_by.push(weak);
        });
        myself
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let write = self.inner.load();
        let mut inner = write.write();
        let result = f(&mut inner);
        #[cfg(feature = "extra_assertions")]
        {
            inner.verify_integrity()
        }
        result
    }

    pub(crate) fn merge(&self, other: &Self) {
        if Arc::ptr_eq(&self.inner.load(), &other.inner.load()) {
            return;
        }
        self.with_mut(|inner| {
            other.with_mut(|other_inner| {
                inner.nodes.merge(&mut other_inner.nodes);

                inner
                    .cached_results
                    .extend(other_inner.cached_results.drain());
                inner.dependency_map.merge(&mut other_inner.dependency_map);
            })
        });

        other.point_to(self);
    }

    fn point_to(&self, target: &Self) {
        {
            if Arc::ptr_eq(&self.inner.load(), &target.inner.load()) {
                return;
            }
        }

        let pointed_to_by = self.with_mut(|inner| std::mem::take(&mut inner.pointed_to_by));

        for pointed_to in pointed_to_by {
            if let Some(pointed_to) = pointed_to.upgrade() {
                if Arc::ptr_eq(&pointed_to, &target.inner) {
                    continue;
                }
                let pointed_to = Self { inner: pointed_to };
                pointed_to.point_to(target);
            }
        }

        target.with_mut(|inner| {
            inner.pointed_to_by.push(Arc::downgrade(&self.inner));
        });
        let target = target.inner.load_full();
        self.inner.store(target);
    }

    pub(crate) fn create_element_wise(
        &self,
        function: ElementWiseOperation,
    ) -> ElementWiseComputeNodeKey {
        let id = ElementWiseComputeNodeKey::new();
        self.with_mut(|inner| {
            inner.nodes.element_wise.insert(id, function);
            inner.add_reference(id.into());
        });
        id
    }

    pub(crate) fn create_pair_wise(&self, function: PairWiseOperation) -> PairWiseComputeNodeKey {
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

    pub(crate) fn resolve(&self, key: AnyComputeKey, device: &Device) -> TensorData {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("ComputeGraph Encoder"),
            });
        let data = self.with_mut(|inner| Resolver::new(inner, key, &mut encoder).run());
        device.wgpu_queue().submit(Some(encoder.finish()));

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

    pub(crate) fn graphvis(&self, key: AnyComputeKey) -> Graph {
        self.with_mut(|inner| inner.graphvis(key))
    }

    pub(crate) fn remove_reference(&self, key: AnyComputeKey) {
        self.with_mut(|inner| {
            inner.remove_reference(key);
        });
    }

    pub(crate) fn add_reference(&self, key: AnyComputeKey) {
        self.with_mut(|inner| {
            inner.add_reference(key);
        });
    }

    pub(crate) fn graph_hash(&self, key: AnyComputeKey) -> u64 {
        self.with_mut(|inner| inner.graph_hash(key))
    }
}

#[derive(Default)]
pub(crate) struct ComputeGraphNodes {
    pub(crate) element_wise: FxHashMap<ElementWiseComputeNodeKey, ElementWiseOperation>,
    pub(crate) pair_wise: FxHashMap<PairWiseComputeNodeKey, PairWiseOperation>,
    pub(crate) mat_mul: FxHashMap<MatMulComputeNodeKey, MatMulOperation>,
    pub(crate) reduce: FxHashMap<ReduceComputeNodeKey, ReduceOperation>,
    pub(crate) map_layout: FxHashMap<MapLayoutComputeNodeKey, MapLayoutOperation>,
    pub(crate) resize: FxHashMap<ResizeComputeNodeKey, ResizeOperation>,
    pub(crate) slice_assign: FxHashMap<SliceAssignComputeNodeKey, SliceAssignOperation>,
    pub(crate) index_select: FxHashMap<IndexSelectComputeNodeKey, IndexSelectOperation>,
    pub(crate) tensor: FxHashMap<TensorComputeNodeKey, TensorData>,
    pub(crate) q_mat_mul: FxHashMap<QMatMulComputeNodeKey, QMatMulOperation>,
    pub(crate) dequantize: FxHashMap<DequantizeComputeKey, DequantizeOperation>,
    pub(crate) custom: FxHashMap<CustomComputeKey, Arc<dyn Operation + Send + Sync>>,
}

impl ComputeGraphNodes {
    pub(crate) fn merge(&mut self, other: &mut Self) {
        self.element_wise.extend(other.element_wise.drain());
        self.pair_wise.extend(other.pair_wise.drain());
        self.mat_mul.extend(other.mat_mul.drain());
        self.reduce.extend(other.reduce.drain());
        self.map_layout.extend(other.map_layout.drain());
        self.resize.extend(other.resize.drain());
        self.slice_assign.extend(other.slice_assign.drain());
        self.index_select.extend(other.index_select.drain());
        self.tensor.extend(other.tensor.drain());
        self.q_mat_mul.extend(other.q_mat_mul.drain());
        self.dequantize.extend(other.dequantize.drain());
        self.custom.extend(other.custom.drain());
    }
}

pub(crate) struct ComputeGraphInner {
    pub(crate) device: Device,
    pub(crate) nodes: ComputeGraphNodes,

    pub(crate) cached_results: FxHashMap<AnyComputeKey, TensorData>,

    dependency_map: DependencyMap,

    pointed_to_by: Vec<Weak<ArcSwapAny<Arc<RwLock<ComputeGraphInner>>>>>,
}

impl ComputeGraphInner {
    fn new(device: Device) -> Self {
        Self {
            device,
            nodes: ComputeGraphNodes::default(),
            cached_results: Default::default(),
            dependency_map: DependencyMap::default(),
            pointed_to_by: Vec::new(),
        }
    }

    fn add_reference(&mut self, key: AnyComputeKey) {
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

    fn remove_reference(&mut self, key: AnyComputeKey) {
        if let Some(count) = self.dependency_map.reference_count.get_mut(&key) {
            *count -= 1;
            // Remove the node if it is dead
            self.check_life(key);
        }
    }

    fn check_life(&mut self, key: AnyComputeKey) {
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

    fn remove_key(&mut self, key: AnyComputeKey) {
        // Remove the cached result if it exists
        visit_dependencies(&self.nodes, key, |dependency| {
            if let Some(map) = self.dependency_map.dependant_map.get_mut(&dependency) {
                map.remove(&key);
            }
        });
        self.dependency_map.dependant_map.remove(&key);
        self.cached_results.remove(&key);
        self.dependency_map.reference_count.remove(&key);
        match key {
            AnyComputeKey::ElementWise(element_wise_compute_node_key) => {
                self.nodes
                    .element_wise
                    .remove(&element_wise_compute_node_key);
            }
            AnyComputeKey::PairWise(pair_wise_compute_node_key) => {
                self.nodes.pair_wise.remove(&pair_wise_compute_node_key);
            }
            AnyComputeKey::MatMul(mat_mul_compute_node_key) => {
                self.nodes.mat_mul.remove(&mat_mul_compute_node_key);
            }
            AnyComputeKey::Reduce(reduce_compute_node_key) => {
                self.nodes.reduce.remove(&reduce_compute_node_key);
            }
            AnyComputeKey::MapLayout(map_layout_compute_node_key) => {
                self.nodes.map_layout.remove(&map_layout_compute_node_key);
            }
            AnyComputeKey::Resize(resize_compute_node_key) => {
                self.nodes.resize.remove(&resize_compute_node_key);
            }
            AnyComputeKey::SliceAssign(slice_assign_compute_node_key) => {
                self.nodes
                    .slice_assign
                    .remove(&slice_assign_compute_node_key);
            }
            AnyComputeKey::IndexSelect(index_select_compute_node_key) => {
                self.nodes
                    .index_select
                    .remove(&index_select_compute_node_key);
            }
            AnyComputeKey::QMatMul(q_mat_mul_compute_node_key) => {
                self.nodes.q_mat_mul.remove(&q_mat_mul_compute_node_key);
            }
            AnyComputeKey::Tensor(tensor_compute_node_key) => {
                self.nodes.tensor.remove(&tensor_compute_node_key);
            }
            AnyComputeKey::Dequantize(dequantize_compute_key) => {
                self.nodes.dequantize.remove(&dequantize_compute_key);
            }
            AnyComputeKey::Custom(custom_compute_key) => {
                self.nodes.custom.remove(&custom_compute_key);
            }
        }
    }

    pub(crate) fn get_result_or_qmatrix(&self, key: AnyComputeKey) -> Option<MaybeQData> {
        let result = if let AnyComputeKey::Dequantize(key) = key {
            let tensor = &self.nodes.dequantize.get(&key)?;
            tensor.matrix.clone().into()
        } else {
            self.cached_results.get(&key)?.clone().into()
        };
        Some(result)
    }

    pub(crate) fn get_result(&self, key: AnyComputeKey) -> Option<TensorData> {
        self.cached_results.get(&key).cloned()
    }

    #[cfg(feature = "extra_assertions")]
    fn contains_key(&self, key: AnyComputeKey) -> bool {
        if self.cached_results.contains_key(&key) {
            return true;
        }
        match key {
            AnyComputeKey::ElementWise(key) => self.nodes.element_wise.contains_key(&key),
            AnyComputeKey::PairWise(key) => self.nodes.pair_wise.contains_key(&key),
            AnyComputeKey::MatMul(key) => self.nodes.mat_mul.contains_key(&key),
            AnyComputeKey::Reduce(key) => self.nodes.reduce.contains_key(&key),
            AnyComputeKey::MapLayout(key) => self.nodes.map_layout.contains_key(&key),
            AnyComputeKey::Resize(key) => self.nodes.resize.contains_key(&key),
            AnyComputeKey::SliceAssign(key) => self.nodes.slice_assign.contains_key(&key),
            AnyComputeKey::IndexSelect(key) => self.nodes.index_select.contains_key(&key),
            AnyComputeKey::Tensor(key) => self.nodes.tensor.contains_key(&key),
            AnyComputeKey::QMatMul(key) => self.nodes.q_mat_mul.contains_key(&key),
            AnyComputeKey::Dequantize(key) => self.nodes.dequantize.contains_key(&key),
            AnyComputeKey::Custom(key) => self.nodes.custom.contains_key(&key),
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
            .map(AnyComputeKey::from)
            .chain(
                self.nodes
                    .pair_wise
                    .keys()
                    .copied()
                    .map(AnyComputeKey::from),
            )
            .chain(self.nodes.mat_mul.keys().copied().map(AnyComputeKey::from))
            .chain(self.nodes.reduce.keys().copied().map(AnyComputeKey::from))
            .chain(
                self.nodes
                    .map_layout
                    .keys()
                    .copied()
                    .map(AnyComputeKey::from),
            )
            .chain(self.nodes.resize.keys().copied().map(AnyComputeKey::from))
            .chain(
                self.nodes
                    .slice_assign
                    .keys()
                    .copied()
                    .map(AnyComputeKey::from),
            )
            .chain(
                self.nodes
                    .index_select
                    .keys()
                    .copied()
                    .map(AnyComputeKey::from),
            )
            .chain(self.nodes.tensor.keys().copied().map(AnyComputeKey::from))
            .chain(
                self.nodes
                    .q_mat_mul
                    .keys()
                    .copied()
                    .map(AnyComputeKey::from),
            )
            .chain(
                self.nodes
                    .dequantize
                    .keys()
                    .copied()
                    .map(AnyComputeKey::from),
            );

        for key in keys {
            if self.cached_results.contains_key(&key) {
                continue;
            }
            visit_dependencies(&self.nodes, key, |dependency| {
                assert!(self.contains_key(dependency));
            });
        }
    }

    pub(crate) fn graph_hash(&self, key: AnyComputeKey) -> u64 {
        let mut cache = FxHashMap::default();
        self.compute_node_hash(key, &mut cache)
    }

    fn compute_node_hash(
        &self,
        key: AnyComputeKey,
        cache: &mut FxHashMap<AnyComputeKey, u64>,
    ) -> u64 {
        if let Some(&hash) = cache.get(&key) {
            return hash;
        }

        let mut hasher = DefaultHasher::new();

        let discriminant = std::mem::discriminant(&key);
        discriminant.hash(&mut hasher);
        match key {
            AnyComputeKey::Tensor(tensor_key) => {
                if let Some(tensor) = self.nodes.tensor.get(&tensor_key) {
                    tensor.info().datatype().hash(&mut hasher);
                    tensor.info().shape().hash(&mut hasher);
                }
            }
            AnyComputeKey::ElementWise(ew_key) => {
                if let Some(op) = self.nodes.element_wise.get(&ew_key) {
                    let dep_hash = self.compute_node_hash(op.value, cache);
                    dep_hash.hash(&mut hasher);
                    op.shape.hash(&mut hasher);
                    // Hash element-wise functions
                    for func in op.functions.iter() {
                        func.datatype.hash(&mut hasher);
                        func.operation.hash(&mut hasher);
                        if let Some(name) = &func.name {
                            name.hash(&mut hasher);
                        }
                    }
                }
            }
            AnyComputeKey::PairWise(pw_key) => {
                if let Some(op) = self.nodes.pair_wise.get(&pw_key) {
                    let first_hash = self.compute_node_hash(op.first, cache);
                    let second_hash = self.compute_node_hash(op.second, cache);
                    first_hash.hash(&mut hasher);
                    second_hash.hash(&mut hasher);
                    op.function.datatype.hash(&mut hasher);
                    op.function.operation.hash(&mut hasher);
                    if let Some(name) = &op.function.name {
                        name.hash(&mut hasher);
                    }
                }
            }
            AnyComputeKey::MatMul(mm_key) => {
                if let Some(op) = self.nodes.mat_mul.get(&mm_key) {
                    let first_hash = self.compute_node_hash(op.first, cache);
                    let second_hash = self.compute_node_hash(op.second, cache);
                    first_hash.hash(&mut hasher);
                    second_hash.hash(&mut hasher);
                    op.datatype.hash(&mut hasher);
                    op.first_shape.hash(&mut hasher);
                    op.second_shape.hash(&mut hasher);
                    op.out_shape.hash(&mut hasher);
                }
            }
            AnyComputeKey::QMatMul(qmm_key) => {
                if let Some(op) = self.nodes.q_mat_mul.get(&qmm_key) {
                    let input_hash = self.compute_node_hash(op.input, cache);
                    input_hash.hash(&mut hasher);
                    op.input_datatype.hash(&mut hasher);
                    op.in_shape.hash(&mut hasher);
                    op.out_shape.hash(&mut hasher);
                    // Hash quantized matrix properties
                    op.matrix.shape().hash(&mut hasher);
                    op.matrix.datatype().hash(&mut hasher);
                }
            }
            AnyComputeKey::Reduce(reduce_key) => {
                if let Some(op) = self.nodes.reduce.get(&reduce_key) {
                    let dep_hash = self.compute_node_hash(op.value, cache);
                    dep_hash.hash(&mut hasher);
                    op.axis.hash(&mut hasher);
                    op.shape.hash(&mut hasher);
                    op.function.datatype.hash(&mut hasher);
                    op.function.operation.hash(&mut hasher);
                    op.function.initial_value.hash(&mut hasher);
                    if let Some(name) = &op.function.name {
                        name.hash(&mut hasher);
                    }
                }
            }
            AnyComputeKey::MapLayout(ml_key) => {
                if let Some(op) = self.nodes.map_layout.get(&ml_key) {
                    let dep_hash = self.compute_node_hash(op.input, cache);
                    dep_hash.hash(&mut hasher);
                }
            }
            AnyComputeKey::Resize(resize_key) => {
                if let Some(op) = self.nodes.resize.get(&resize_key) {
                    let dep_hash = self.compute_node_hash(op.input, cache);
                    dep_hash.hash(&mut hasher);
                    op.current_shape.hash(&mut hasher);
                    op.new_shape.hash(&mut hasher);
                    op.fill_shape.hash(&mut hasher);
                }
            }
            AnyComputeKey::SliceAssign(sa_key) => {
                if let Some(op) = self.nodes.slice_assign.get(&sa_key) {
                    let input_hash = self.compute_node_hash(op.input, cache);
                    let value_hash = self.compute_node_hash(op.value, cache);
                    input_hash.hash(&mut hasher);
                    value_hash.hash(&mut hasher);
                    for slice in op.slices.iter() {
                        slice.start.hash(&mut hasher);
                        slice.end.hash(&mut hasher);
                    }
                }
            }
            AnyComputeKey::IndexSelect(is_key) => {
                if let Some(op) = self.nodes.index_select.get(&is_key) {
                    let input_hash = self.compute_node_hash(op.input, cache);
                    let indexes_hash = self.compute_node_hash(op.indexes, cache);
                    input_hash.hash(&mut hasher);
                    indexes_hash.hash(&mut hasher);
                    op.datatype.hash(&mut hasher);
                    op.dimension.hash(&mut hasher);
                    op.value_shape.hash(&mut hasher);
                    op.indexes_shape.hash(&mut hasher);
                }
            }
            AnyComputeKey::Dequantize(dq_key) => {
                if let Some(op) = self.nodes.dequantize.get(&dq_key) {
                    op.datatype.hash(&mut hasher);
                    op.matrix.shape().hash(&mut hasher);
                    op.matrix.datatype().hash(&mut hasher);
                }
            }
            AnyComputeKey::Custom(_) => {
                // Custom operations don't have a stable hash representation
            }
        }

        let hash = hasher.finish();
        cache.insert(key, hash);
        hash
    }
}
