use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, Div, DivAssign, Index, Mul, MulAssign, Range, Sub, SubAssign},
    sync::Arc,
};

use bytemuck::{AnyBitPattern, NoUninit};
use tabbycat::Graph;
use wgpu::{
    BufferDescriptor, COPY_BUFFER_ALIGNMENT,
    util::{DeviceExt, DownloadBuffer},
};

use crate::{
    Device, ElementWiseOperation, MatMulOperation, MatMulParams, PairWiseFunction,
    PairWiseOperation, ReduceFunction, ReduceOperation,
    compute_graph::{AnyComputeKey, ComputeGraph},
    index_select::IndexSelectOperation,
    layout::Layout,
    map_layout::MapLayoutOperation,
    mir::operation::Operation,
    quantized::{QMatrix, matmul::QMatMulOperation},
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
};

pub trait DataType:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + PartialOrd
    + NoUninit
    + AnyBitPattern
    + Debug
    + Display
{
    const WGSL_TYPE: DataTypeEnum;

    fn zero() -> Self;
    fn one() -> Self;
}

pub trait FloatDataType: DataType {
    fn from_f32(value: f32) -> Self;

    fn is_finite(&self) -> bool;
}

impl DataType for f32 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::F32;

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }
}

impl FloatDataType for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
}

impl DataType for half::f16 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::F16;

    fn zero() -> Self {
        half::f16::from_f32(0.)
    }

    fn one() -> Self {
        half::f16::from_f32(1.)
    }
}

impl FloatDataType for half::f16 {
    fn from_f32(value: f32) -> Self {
        half::f16::from_f32(value)
    }

    fn is_finite(&self) -> bool {
        half::f16::is_finite(*self)
    }
}

impl DataType for u32 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::U32;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataTypeEnum {
    F32,
    F16,
    U32,
}

impl DataTypeEnum {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataTypeEnum::F32 => "f32",
            DataTypeEnum::F16 => "f16",
            DataTypeEnum::U32 => "u32",
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            DataTypeEnum::F32 => size_of::<f32>(),
            DataTypeEnum::F16 => size_of::<half::f16>(),
            DataTypeEnum::U32 => size_of::<u32>(),
        }
    }
}

impl Display for DataTypeEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorLayoutInfo {
    layout: Layout,
    datatype: DataTypeEnum,
}

impl Display for TensorLayoutInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.layout.shape(), self.datatype)
    }
}

impl TensorLayoutInfo {
    pub(crate) fn new(layout: Layout, datatype: DataTypeEnum) -> Self {
        Self { layout, datatype }
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorInfo {
    shape: Box<[usize]>,
    datatype: DataTypeEnum,
}

impl Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.shape, self.datatype)
    }
}

impl TensorInfo {
    pub(crate) fn new(shape: Box<[usize]>, datatype: DataTypeEnum) -> Self {
        Self { shape, datatype }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

pub(crate) struct LazyTensorData {
    device: Device,
    info: TensorInfo,
    graph: ComputeGraph,
    key: AnyComputeKey,
}

impl Clone for LazyTensorData {
    fn clone(&self) -> Self {
        self.graph.add_reference(self.key);
        Self {
            device: self.device.clone(),
            info: self.info.clone(),
            graph: self.graph.clone(),
            key: self.key,
        }
    }
}

impl Drop for LazyTensorData {
    fn drop(&mut self) {
        self.graph.remove_reference(self.key);
    }
}

impl LazyTensorData {
    pub(crate) fn new(data: TensorData) -> Self {
        let device = data.device.clone();
        let graph = ComputeGraph::new(device.clone());
        let info = data.info.clone();
        let key = graph.create_tensor(data);

        Self {
            device,
            info: TensorInfo::new(info.shape().into(), info.datatype()),
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn from_parts(
        device: Device,
        graph: ComputeGraph,
        info: TensorInfo,
        key: AnyComputeKey,
    ) -> Self {
        Self {
            device,
            info,
            graph,
            key,
        }
    }

    pub(crate) fn custom(&self, custom: Arc<dyn Operation + Send + Sync>) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let info = self.info.clone();
        let key = graph.create_custom(custom);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn element_wise(&self, function: ElementWiseOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.datatype = function.functions.out_datatype();
        let key = graph.create_element_wise(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn pair_wise(&self, function: PairWiseOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let info = self.info.clone();
        let key = graph.create_pair_wise(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn mat_mul(&self, function: MatMulOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = graph.create_mat_mul(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn q_mat_mul(&self, function: QMatMulOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = graph.create_q_mat_mul(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn reduce(&self, function: ReduceOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let mut info = self.info.clone();
        let dim = function.axis;
        let new_shape: Box<[usize]> = self
            .info
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect();
        info = TensorInfo::new(new_shape, info.datatype());
        let key = graph.create_reduce(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn map_layout(&self, op: MapLayoutOperation) -> Self {
        let device = self.device.clone();
        let info = TensorInfo::new((op.map_size)(self.info.shape()), self.info.datatype());
        let graph = self.graph.clone();
        let key = self.graph.create_map_layout(op);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn resize(&self, op: ResizeOperation) -> Self {
        let device = self.device.clone();
        let info = TensorInfo::new(op.new_shape.clone(), self.info.datatype());
        let graph = self.graph.clone();
        let key = self.graph.create_resize(op);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn slice_assign(&self, op: SliceAssignOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let graph = self.graph.clone();
        let key = self.graph.create_slice_assign(op);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn index_select(&self, op: IndexSelectOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = op.output_shape();
        let graph = self.graph.clone();
        let key = self.graph.create_index_select(op);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn materialize(&self) -> TensorData {
        self.graph.resolve(self.key, &self.device)
    }

    pub fn graphvis(&self) -> Graph {
        self.graph.graphvis(self.key)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TensorData {
    device: Device,
    buffer: Arc<wgpu::Buffer>,
    info: TensorLayoutInfo,
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        self.info == other.info && self.buffer == other.buffer
    }
}

impl TensorData {
    pub(crate) fn new_from_buffer(
        device: &Device,
        buffer: impl Into<Arc<wgpu::Buffer>>,
        size: &[usize],
        datatype: DataTypeEnum,
    ) -> Self {
        let layout = Layout::contiguous(size);
        Self::new_from_parts(device, buffer, layout, datatype)
    }

    pub(crate) fn new_from_parts(
        device: &Device,
        buffer: impl Into<Arc<wgpu::Buffer>>,
        layout: Layout,
        datatype: DataTypeEnum,
    ) -> Self {
        Self {
            device: device.clone(),
            buffer: buffer.into(),
            info: TensorLayoutInfo::new(layout, datatype),
        }
    }

    pub(crate) fn new_for_shape(device: &Device, shape: &[usize], datatype: DataTypeEnum) -> Self {
        let size =
            padded_tensor_size((datatype.element_size() * shape.iter().product::<usize>()) as u64);
        let buffer = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tensor Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self::new_from_buffer(device, buffer, shape, datatype)
    }

    pub(crate) fn new_splat<D: DataType>(device: &Device, shape: &[usize], data: D) -> Self {
        let datatype = D::WGSL_TYPE;
        let buffer = device
            .wgpu_device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Splat Tensor Buffer"),
                contents: bytemuck::bytes_of(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let strides = (0..shape.len()).map(|_| 0).collect();
        let layout = Layout::from_parts(0, shape.into(), strides);
        Self::new_from_parts(device, buffer, layout, datatype)
    }

    fn new_inner<'a, D: DataType, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: &[usize],
    ) -> Self {
        // MODIFIED from: https://github.com/gfx-rs/wgpu/blob/d8833d079833c62b4fd00325d0ba08ec0c8bc309/wgpu/src/util/device.rs#L38
        fn create_aligned_buffer(
            element_size: u64,
            shape: &[usize],
            device: &Device,
        ) -> (wgpu::Buffer, u64) {
            let size = element_size * shape.iter().copied().product::<usize>() as u64;

            let padded_size = padded_tensor_size(size);

            let wgt_descriptor = BufferDescriptor {
                label: Some("Tensor Buffer"),
                size: padded_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            };

            let buffer = device.wgpu_device().create_buffer(&wgt_descriptor);
            (buffer, size)
        }
        let (buffer, unpadded_size) = create_aligned_buffer(size_of::<D>() as u64, shape, device);

        buffer.slice(..).get_mapped_range_mut()[..unpadded_size as usize]
            .iter_mut()
            .zip(data.flat_map(bytemuck::bytes_of))
            .for_each(|(dst, src)| *dst = *src);
        buffer.unmap();

        Self::new_from_buffer(device, buffer, shape, D::WGSL_TYPE)
    }

    pub fn slice(&self, ranges: &[Range<usize>]) -> Self {
        let layout = self.info.layout.slice(ranges);
        Self {
            device: self.device.clone(),
            buffer: self.buffer.clone(),
            info: TensorLayoutInfo::new(layout, self.info.datatype),
        }
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.info.layout
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.info.datatype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    pub(crate) fn info(&self) -> &TensorLayoutInfo {
        &self.info
    }

    /// Check if this is the only reference to the buffer
    pub(crate) fn owned(&self) -> bool {
        std::sync::Arc::strong_count(&self.buffer) == 1
    }
}

pub struct Tensor<const R: usize, D> {
    data: LazyTensorData,
    datatype: PhantomData<D>,
}

impl<const R: usize, D: DataType> Display for Tensor<R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {:?}", self.datatype(), self.shape())
    }
}

impl<const R: usize, D: DataType> Debug for Tensor<R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({} x {:?})", self.datatype(), self.shape())
    }
}

impl<const R: usize, D: DataType> From<TensorData> for Tensor<R, D> {
    fn from(value: TensorData) -> Self {
        Self {
            data: LazyTensorData::new(value),
            datatype: PhantomData,
        }
    }
}

impl<const R: usize, D> Clone for Tensor<R, D> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            datatype: PhantomData,
        }
    }
}

pub trait IntoTensor<const R: usize, D> {
    fn into_tensor(self, device: &Device) -> Tensor<R, D>;
}

impl<D: DataType> IntoTensor<0, D> for () {
    fn into_tensor(self, device: &Device) -> Tensor<0, D> {
        let iter = std::iter::empty();
        Tensor::new_inner(device, iter, [])
    }
}

impl<'a, I, D: DataType> IntoTensor<1, D> for I
where
    I: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<1, D> {
        let iter = self.into_iter();
        let size = iter.len();
        Tensor::new_inner(device, iter, [size])
    }
}

impl<'a, I, I2, D: DataType> IntoTensor<2, D> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<2, D> {
        let mut iter = self.into_iter().map(IntoIterator::into_iter).peekable();
        let size = iter.len();
        let second_size = iter.peek().map(ExactSizeIterator::len).unwrap_or_default();
        let iter = iter.flat_map(|i| {
            let size = i.len();
            if size != second_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {second_size}, but another inner iterator size was {size}");
            }
            i
        });
        Tensor::new_inner(device, iter, [size, second_size])
    }
}

impl<'a, I, I2, I3, D: DataType> IntoTensor<3, D> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<3, D> {
        let mut iter = self
            .into_iter()
            .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
            .peekable();
        let mut shape = [iter.len(), 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek() {
                let size = iter.len();
                shape[2] = size;
            }
        }

        let iter = iter.flat_map(|i| {
            let size = i.len();
            let required_size = shape[1];
            if size != required_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}");
            }
            i.flat_map(|i| {
                let size = i.len();
                let required_size = shape[2];
                if size != required_size {
                    panic!("expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}");
                }
                i
            })
        });

        Tensor::new_inner(device, iter, shape)
    }
}

impl<'a, I, I2, I3, I4, D: DataType> IntoTensor<4, D> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<4, D> {
        let mut iter = self
            .into_iter()
            .map(|i| {
                i.into_iter()
                    .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
                    .peekable()
            })
            .peekable();
        let mut shape = [iter.len(), 0, 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek_mut() {
                let size = iter.len();
                shape[2] = size;
                if let Some(iter) = iter.peek() {
                    let size = iter.len();
                    shape[3] = size;
                }
            }
        }

        let iter = iter.flat_map(|i| {
            let size = i.len();
            let required_size = shape[1];
            if size != required_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}");
            }
            i.flat_map(|i| {
                let size = i.len();
                let required_size = shape[2];
                if size != required_size {
                    panic!("expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}");
                }
                i.flat_map(|i| {
                    let size = i.len();
                    let required_size = shape[3];
                    if size != required_size {
                        panic!("expected a rectangular matrix. The first inner inner inner iterator size was {required_size}, but another inner inner inner iterator size was {size}");
                    }
                    i
                })
            })
        });

        Tensor::new_inner(device, iter, shape)
    }
}

impl<D: DataType, const R: usize> Tensor<R, D> {
    pub fn new(device: &Device, data: impl IntoTensor<R, D>) -> Self {
        data.into_tensor(device)
    }

    pub fn splat(device: &Device, value: D, shape: [usize; R]) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_splat(
            device, &shape, value,
        )))
    }

    pub(crate) fn from_parts(data: LazyTensorData) -> Self {
        debug_assert_eq!(D::WGSL_TYPE, data.info.datatype());
        Self {
            data,
            datatype: PhantomData,
        }
    }

    fn new_inner<'a, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: [usize; R],
    ) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_inner(
            device, data, &shape,
        )))
    }

    async fn as_slice_from_tensor_data(
        tensor: &TensorData,
    ) -> Result<TensorSlice<R, D>, wgpu::BufferAsyncError> {
        let buffer = tensor.buffer();
        let (sender, receiver) = futures_channel::oneshot::channel();
        DownloadBuffer::read_buffer(
            tensor.device.wgpu_device(),
            tensor.device.wgpu_queue(),
            &buffer.slice(..),
            move |result| {
                _ = sender.send(result);
            },
        );
        let downloaded = receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        Ok(TensorSlice::new(downloaded, tensor.layout().clone()))
    }

    #[track_caller]
    pub fn materialize(&self) -> impl Future<Output = ()> + 'static {
        #[allow(unused)]
        let data = self.data.materialize();
        #[cfg(feature = "extra_assertions")]
        let caller = std::panic::Location::caller();
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.device().wgpu_queue().on_submitted_work_done(|| {
            _ = sender.send(());
        });
        async move {
            let _ = receiver.await;
            #[cfg(feature = "extra_assertions")]
            {
                let mut contains_non_finite = false;
                if D::WGSL_TYPE == DataTypeEnum::F32 {
                    let data: TensorSlice<R, f32> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                } else if D::WGSL_TYPE == DataTypeEnum::F16 {
                    let data: TensorSlice<R, half::f16> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                }

                if contains_non_finite {
                    tracing::warn!(
                        "Tensor materialized at {} contains non-finite values. This may lead to unexpected behavior.",
                        caller
                    );
                }
            }
        }
    }

    pub async fn as_slice(&self) -> Result<TensorSlice<R, D>, wgpu::BufferAsyncError> {
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let tensor = self.data.materialize();
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Materialized tensor in {:?}", start_time.elapsed());
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let out = Self::as_slice_from_tensor_data(&tensor).await;
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Downloaded tensor in {:?}", start_time.elapsed());
        out
    }

    pub fn debug_assert_real(self) -> Self
    where
        D: FloatDataType,
    {
        #[cfg(debug_assertions)]
        {
            use pollster::FutureExt as _;
            let as_slice = self.as_slice().block_on().unwrap();
            for item in as_slice.as_slice() {
                assert!(item.is_finite(), "Tensor contains non-finite value: {item}");
            }
        }
        self
    }

    pub(crate) fn element_wise<D2: DataType>(
        &self,
        function: ElementWiseOperation,
    ) -> Tensor<R, D2> {
        Tensor::from_parts(self.data.element_wise(function))
    }

    pub(crate) fn pair_wise(&self, other: &Self, function: PairWiseFunction) -> Self {
        // If the two tensors are the same, we can lower this to a cheaper element wise operation
        if self.data.key == other.data.key {
            return self.element_wise(ElementWiseOperation::new(
                self.datatype(),
                self.key(),
                function.lower_to_element_wise(),
                self.shape().as_slice(),
            ));
        }

        self.data.graph.merge(&other.data.graph);
        assert_eq!(self.shape(), other.shape());
        let operation =
            PairWiseOperation::new(function, self.data.key, other.data.key, self.shape());
        Self::from_parts(self.data.pair_wise(operation))
    }

    pub(crate) fn add_mat_mul(&self, other: &Self, parameters: Option<MatMulParams>) -> Self {
        self.data.graph.merge(&other.data.graph);
        let operation = MatMulOperation::new(
            self.datatype(),
            self.data.key,
            other.data.key,
            self.shape(),
            other.shape(),
            parameters,
        );

        Self::from_parts(self.data.mat_mul(operation))
    }

    pub(crate) fn add_q_mat_mul(&self, other: &QMatrix) -> Self {
        let operation =
            QMatMulOperation::new(self.datatype(), self.shape(), self.data.key, other.clone());

        Self::from_parts(self.data.q_mat_mul(operation))
    }

    #[cfg(test)]
    pub(crate) fn add_q_mat_mul_with_chunked_config(
        &self,
        other: &QMatrix,
        config: crate::quantized::matmul::ChunkedSgemmConfig,
    ) -> Self {
        let operation =
            QMatMulOperation::new(self.datatype(), self.shape(), self.data.key, other.clone())
                .with_chunked_config(config);

        Self::from_parts(self.data.q_mat_mul(operation))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn add_q_mat_mul_with_general_config(
        &self,
        other: &QMatrix,
        config: crate::quantized::matmul::GeneralSgemmConfig,
    ) -> Self {
        let operation =
            QMatMulOperation::new(self.datatype(), self.shape(), self.data.key, other.clone())
                .with_general_config(config);

        Self::from_parts(self.data.q_mat_mul(operation))
    }

    pub(crate) fn add_resize<const R2: usize>(&self, op: ResizeOperation) -> Tensor<R2, D> {
        Tensor {
            data: self.data.resize(op),
            datatype: PhantomData,
        }
    }

    pub(crate) fn add_slice_assign(&self, other: &Self, slices: [Range<usize>; R]) -> Self {
        self.data.graph.merge(&other.data.graph);
        let op = SliceAssignOperation::new(self.data.key, other.data.key, slices.into());
        Self::from_parts(self.data.slice_assign(op))
    }

    pub(crate) fn add_index_select(&self, dimension: usize, indexes: &Tensor<1, u32>) -> Self {
        self.data.graph.merge(&indexes.data.graph);
        let op = IndexSelectOperation::new(
            self.data.key,
            indexes.data.key,
            self.datatype(),
            dimension,
            self.shape(),
            indexes.shape(),
        );
        Self::from_parts(self.data.index_select(op))
    }

    pub(crate) fn reduce<const OUT: usize>(
        &self,
        function: ReduceFunction,
        dim: usize,
    ) -> Tensor<OUT, D> {
        Tensor {
            data: self.data.reduce(ReduceOperation::new(
                self.data.key,
                function,
                dim,
                self.shape(),
            )),
            datatype: PhantomData,
        }
    }

    pub(crate) fn add_map_layout<const R2: usize>(&self, op: MapLayoutOperation) -> Tensor<R2, D> {
        Tensor::from_parts(self.data.map_layout(op))
    }

    pub(crate) fn key(&self) -> AnyComputeKey {
        self.data.key
    }

    pub fn shape(&self) -> &[usize; R] {
        let shape = self.data.info.shape();
        match shape.try_into() {
            Ok(shape) => shape,
            Err(_) => {
                panic!("Internal error. Expected a tensor of rank {R}, found shape: {shape:?}")
            }
        }
    }

    pub fn rank(&self) -> usize {
        self.data.info.rank()
    }

    pub fn datatype(&self) -> DataTypeEnum {
        self.data.info.datatype()
    }

    pub fn device(&self) -> &Device {
        &self.data.device
    }

    pub fn graphvis(&self) -> Graph {
        self.data.graphvis()
    }

    pub(crate) fn data(&self) -> &LazyTensorData {
        &self.data
    }

    pub(crate) fn graph(&self) -> &ComputeGraph {
        &self.data.graph
    }

    /// Returns a hash of the compute graph. The hash is sensitive to the structure of the
    /// compute graph
    pub fn graph_hash(&self) -> u64 {
        self.data.graph.graph_hash(self.data.key)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor_slice() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let slice = tensor.slice([0..2, 0..1]);
    let as_slice = slice.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice.get([0, 1]), None);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice.get([1, 1]), None);
    assert_eq!(as_slice.get([2, 0]), None);
    assert_eq!(as_slice.get([2, 1]), None);
}

pub struct TensorSlice<const R: usize, D> {
    buffer: DownloadBuffer,
    layout: Layout,
    datatype: PhantomData<D>,
}

impl<D: DataType + Debug> Debug for TensorSlice<0, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get([]).fmt(f)
    }
}

impl<D: DataType + Debug> Debug for TensorSlice<1, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| self.get([i]).unwrap())
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: DataType + Debug> Debug for TensorSlice<2, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| self.get([i, j]).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: DataType + Debug> Debug for TensorSlice<3, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| {
                        (0..shape[2])
                            .map(|k| self.get([i, j, k]).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: DataType + Debug> Debug for TensorSlice<4, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| {
                        (0..shape[2])
                            .map(|k| {
                                (0..shape[3])
                                    .map(|l| self.get([i, j, k, l]).unwrap())
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<const R: usize, D: DataType + PartialEq> PartialEq for TensorSlice<R, D> {
    fn eq(&self, other: &Self) -> bool {
        let self_shape = self.layout.shape();
        let other_shape = other.layout.shape();
        if self_shape != other_shape {
            return false;
        }
        if R == 0 {
            return true;
        }
        let mut matches = true;
        self.visit_indexes(|index| {
            matches &= self.get(index) == other.get(index);
        });
        matches
    }
}

impl<const R: usize, D: DataType> TensorSlice<R, D> {
    fn visit_indexes(&self, mut visitor: impl FnMut([usize; R])) {
        let self_shape = self.layout.shape();
        let mut index = [0; R];
        loop {
            index[0] += 1;
            for i in 0..R {
                if index[i] >= self_shape[i] {
                    index[i] = 0;
                    if i == R - 1 {
                        return;
                    }
                    index[i + 1] += 1;
                } else {
                    break;
                }
            }
            visitor(index);
        }
    }

    #[cfg(feature = "extra_assertions")]
    fn visit_items(&self, mut visitor: impl FnMut(&D)) {
        self.visit_indexes(|index| {
            visitor(self.get(index).unwrap());
        });
    }
}

impl<'a, D: DataType> PartialEq<&'a [D]> for TensorSlice<1, D> {
    fn eq(&self, other: &&'a [D]) -> bool {
        self.as_slice() == *other
    }
}

impl<const N: usize, D: DataType> PartialEq<[D; N]> for TensorSlice<1, D> {
    fn eq(&self, other: &[D; N]) -> bool {
        self.as_slice() == *other
    }
}

impl<D: DataType> PartialEq<TensorSlice<1, D>> for &[D] {
    fn eq(&self, other: &TensorSlice<1, D>) -> bool {
        *self == other.as_slice()
    }
}

impl<const N: usize, D: DataType> PartialEq<TensorSlice<1, D>> for &[D; N] {
    fn eq(&self, other: &TensorSlice<1, D>) -> bool {
        *self == other.as_slice()
    }
}

pub(crate) fn padded_tensor_size(size: u64) -> u64 {
    // Valid vulkan usage is
    // 1. buffer size must be a multiple of COPY_BUFFER_ALIGNMENT.
    // 2. buffer size must be greater than 0.
    // Therefore we round the value up to the nearest multiple, and ensure it's at least COPY_BUFFER_ALIGNMENT.
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;

    ((size + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor_compare() {
    let device = Device::new().await.unwrap();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
        [[7., 8.], [7., 8.]],
        [[9., 10.], [9., 10.]],
        [[11., 12.], [11., 12.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let slice = tensor.slice([0..2, 0..1, 0..1]);
    let as_slice = slice.as_slice().await.unwrap();
    assert_eq!(as_slice, as_slice);

    let other_slice = tensor.slice([0..1, 0..1, 0..1]);
    let other_as_slice = other_slice.as_slice().await.unwrap();
    assert!(as_slice != other_as_slice);

    let other_slice = tensor.slice([1..3, 0..1, 0..1]);
    let other_as_slice = other_slice.as_slice().await.unwrap();
    assert!(as_slice != other_as_slice);
}

impl<D: DataType, const R: usize> TensorSlice<R, D> {
    fn new(buffer: DownloadBuffer, layout: Layout) -> Self {
        Self {
            buffer,
            layout,
            datatype: PhantomData,
        }
    }

    fn as_slice(&self) -> &[D] {
        bytemuck::cast_slice(&self.buffer.deref()[self.layout.offset() * size_of::<D>()..])
    }
}

impl<D: DataType, const R: usize> TensorSlice<R, D> {
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    fn get(&self, index: [usize; R]) -> Option<&D> {
        let mut index_sum = 0;
        let layout = &self.layout;
        for ((index_component, &stride), &size) in
            index.into_iter().zip(layout.strides()).zip(layout.shape())
        {
            if index_component >= size {
                return None;
            }
            index_sum += stride * index_component;
        }

        self.as_slice().get(index_sum)
    }
}

impl<D: DataType, const R: usize> Index<[usize; R]> for TensorSlice<R, D> {
    type Output = D;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.get(index).unwrap()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let as_slice = tensor.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_graph_hash() {
    let device = Device::new().await.unwrap();

    // Create a tensor and use it in different operations
    let data = [[1., 2.], [3., 4.]];
    let tensor = Tensor::new(&device, &data);

    // Create two identical computational graphs using the same tensor
    let result1 = &tensor + &tensor; // Add tensor to itself
    let result1 = result1 * 2.0; // Multiply by 2

    let result2 = &tensor + &tensor; // Same operations
    let result2 = result2 * 2.0;

    // The hashes should be the same because the graph structure is identical
    let hash1 = result1.graph_hash();
    let hash2 = result2.graph_hash();
    assert_eq!(hash1, hash2, "Identical graphs should have the same hash");

    // Create a different graph with a different operation
    let result3 = &tensor * &tensor; // Different operation (multiply instead of add)
    let result3 = result3 * 2.0;
    let hash3 = result3.graph_hash();
    assert_ne!(
        hash1, hash3,
        "Different graphs should have different hashes"
    );

    // Create a graph with different ordering (should have different hash)
    let result4 = tensor.clone() * 2.0; // Multiply first
    let result4 = &result4 + &result4; // Then add
    let hash4 = result4.graph_hash();
    assert_ne!(
        hash1, hash4,
        "Graphs with different operation order should have different hashes"
    );

    // Test that using a different tensor with the same shape produces the same hash
    let data2 = [[5., 6.], [7., 8.]]; // Different data, same shape
    let tensor2 = Tensor::new(&device, &data2);
    let result5 = &tensor2 + &tensor2;
    let result5 = result5 * 2.0;
    let hash5 = result5.graph_hash();
    assert_eq!(
        hash1, hash5,
        "Graphs with different tensors but same shape should have the same hash"
    );

    // Test that different shapes produce different hashes
    let data3 = [[1., 2., 3.], [4., 5., 6.]];
    let tensor3 = Tensor::new(&device, &data3);
    let result6 = &tensor3 + &tensor3;
    let result6 = result6 * 2.0;
    let hash6 = result6.graph_hash();
    assert_ne!(
        hash1, hash6,
        "Graphs with different shapes should have different hashes"
    );
}
