use std::{
    collections::HashMap,
    sync::{Arc, RwLock, atomic::AtomicUsize},
};

use arc_swap::ArcSwap;
use tabbycat::Graph;

mod layout_pass;
mod resolve;
mod visit;
mod visualize;

use crate::{
    map_layout::MapLayoutOperation, quantized::QMatMulOperation, resize::ResizeOperation, slice_assign::SliceAssignOperation, tensor::TensorData, Device, ElementWiseOperation, MatMulOperation, PairWiseOperation, PerformanceQueries, QueryResults, ReduceOperation
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
pub(crate) struct TensorComputeNodeKey(usize);
impl TensorComputeNodeKey {
    fn new() -> Self {
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
    Tensor(TensorComputeNodeKey),
    QMatMul(QMatMulComputeNodeKey),
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

impl From<QMatMulComputeNodeKey> for AnyComputeKey {
    fn from(value: QMatMulComputeNodeKey) -> Self {
        Self::QMatMul(value)
    }
}

#[derive(Clone, Default)]
pub(crate) struct ComputeGraph {
    inner: Arc<ArcSwap<RwLock<ComputeGraphInner>>>,
}

impl ComputeGraph {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let write = self.inner.load();
        let mut inner = write.write().unwrap();
        f(&mut inner)
    }

    pub(crate) fn merge(&self, other: &Self) {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return;
        }
        self.with_mut(|inner| {
            other.with_mut(|other_inner| {
                inner.element_wise.extend(other_inner.element_wise.drain());
                inner.pair_wise.extend(other_inner.pair_wise.drain());
                inner.mat_mul.extend(other_inner.mat_mul.drain());
                inner.reduce.extend(other_inner.reduce.drain());
                inner.map_layout.extend(other_inner.map_layout.drain());
                inner.resize.extend(other_inner.resize.drain());
                inner.slice_assign.extend(other_inner.slice_assign.drain());
                inner.tensor.extend(other_inner.tensor.drain());
                inner.q_mat_mul.extend(other_inner.q_mat_mul.drain());
                inner.timing_information.extend(other_inner.timing_information.drain());
            })
        });
        other.inner.store(self.inner.load_full());
    }

    pub(crate) fn create_element_wise(
        &self,
        function: ElementWiseOperation,
    ) -> ElementWiseComputeNodeKey {
        let id = ElementWiseComputeNodeKey::new();
        self.with_mut(|inner| inner.element_wise.insert(id, function));
        id
    }

    pub(crate) fn create_pair_wise(&self, function: PairWiseOperation) -> PairWiseComputeNodeKey {
        let id = PairWiseComputeNodeKey::new();
        self.with_mut(|inner| inner.pair_wise.insert(id, function));
        id
    }

    pub(crate) fn create_mat_mul(&self, function: MatMulOperation) -> MatMulComputeNodeKey {
        let id = MatMulComputeNodeKey::new();
        self.with_mut(|inner| inner.mat_mul.insert(id, function));
        id
    }

    pub(crate) fn create_q_mat_mul(&self, function: QMatMulOperation) -> QMatMulComputeNodeKey {
        let id = QMatMulComputeNodeKey::new();
        self.with_mut(|inner| inner.q_mat_mul.insert(id, function));
        id
    }

    pub(crate) fn create_reduce(&self, function: ReduceOperation) -> ReduceComputeNodeKey {
        let id = ReduceComputeNodeKey::new();
        self.with_mut(|inner| inner.reduce.insert(id, function));
        id
    }

    pub(crate) fn create_map_layout(&self, op: MapLayoutOperation) -> MapLayoutComputeNodeKey {
        let id = MapLayoutComputeNodeKey::new();
        self.with_mut(|inner| inner.map_layout.insert(id, op));
        id
    }

    pub(crate) fn create_resize(&self, op: ResizeOperation) -> ResizeComputeNodeKey {
        let id = ResizeComputeNodeKey::new();
        self.with_mut(|inner| inner.resize.insert(id, op));
        id
    }

    pub(crate) fn create_slice_assign(
        &self,
        op: SliceAssignOperation,
    ) -> SliceAssignComputeNodeKey {
        let id = SliceAssignComputeNodeKey::new();
        self.with_mut(|inner| inner.slice_assign.insert(id, op));
        id
    }

    pub(crate) fn create_tensor(&self, info: TensorData) -> TensorComputeNodeKey {
        let id = TensorComputeNodeKey::new();
        self.with_mut(|inner| inner.tensor.insert(id, info));
        id
    }

    pub(crate) fn resolve(&self, key: AnyComputeKey, device: &Device) -> TensorData {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&Default::default());
        let data = self.with_mut(|inner| inner.resolve(key, &mut encoder));
        device.wgpu_queue().submit(Some(encoder.finish()));
        data
    }

    pub(crate) fn graphvis(&self, key: AnyComputeKey) -> Graph {
        self.with_mut(|inner| inner.graphvis(key))
    }

    #[allow(clippy::await_holding_lock)]
    pub(crate) async fn all_timing_information(&self) -> Vec<QueryResults> {
        let myself = self.inner.load();
        let myself = myself.read().unwrap();
        let mut output = Vec::new();
        for timing_information in myself.timing_information.values() {
            output.push(timing_information.wait_for_results().await);
        }
        output
    }
}

#[derive(Default)]
struct ComputeGraphInner {
    element_wise: HashMap<ElementWiseComputeNodeKey, ElementWiseOperation>,
    pair_wise: HashMap<PairWiseComputeNodeKey, PairWiseOperation>,
    mat_mul: HashMap<MatMulComputeNodeKey, MatMulOperation>,
    reduce: HashMap<ReduceComputeNodeKey, ReduceOperation>,
    map_layout: HashMap<MapLayoutComputeNodeKey, MapLayoutOperation>,
    resize: HashMap<ResizeComputeNodeKey, ResizeOperation>,
    slice_assign: HashMap<SliceAssignComputeNodeKey, SliceAssignOperation>,
    tensor: HashMap<TensorComputeNodeKey, TensorData>,
    q_mat_mul: HashMap<QMatMulComputeNodeKey, QMatMulOperation>,
    timing_information: HashMap<AnyComputeKey, PerformanceQueries>,
}
