use std::{fmt::Debug, ops::Range, sync::Arc};

use crate::{
    DataType, Layout, MaxRank, Tensor, TensorData, compute_graph::AnyComputeKey,
    mir::operation::Operation, slice_shape, slice_strides,
};

type MapSize = Arc<dyn Fn(&[usize]) -> Box<[usize]> + Send + Sync>;
type MapStride = Arc<dyn Fn(usize, &[usize]) -> (usize, Box<[usize]>) + Send + Sync>;

#[derive(Clone)]
pub(crate) struct MapLayoutOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) map_size: MapSize,
    pub(crate) map_stride: MapStride,
}

impl Debug for MapLayoutOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapLayoutOperation")
            .field("input", &self.input)
            .finish()
    }
}

impl MapLayoutOperation {
    pub fn new(
        input: AnyComputeKey,
        map_size: impl Fn(&[usize]) -> Box<[usize]> + Send + Sync + 'static,
        map_stride: impl Fn(usize, &[usize]) -> (usize, Box<[usize]>) + Send + Sync + 'static,
    ) -> Self {
        Self {
            input,
            map_size: Arc::new(map_size),
            map_stride: Arc::new(map_stride),
        }
    }

    pub fn map_tensor(&self, tensor: &TensorData) -> TensorData {
        TensorData::new_from_parts(
            tensor.device(),
            tensor.buffer().clone(),
            self.map_layout(tensor.layout()),
            tensor.datatype(),
        )
    }

    pub fn map_layout(&self, layout: &Layout) -> Layout {
        let (offset, strides) = (self.map_stride)(layout.offset(), layout.strides());
        Layout::from_parts(offset, (self.map_size)(layout.shape()), strides)
    }

    pub fn run(&self, graph: &mut crate::compute_graph::ComputeGraphInner) -> TensorData {
        let input = graph.get_result(self.input).unwrap();
        self.map_tensor(&input)
    }
}

impl Operation for MapLayoutOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        Default::default()
    }

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        [1, 1, 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.input);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        vec![nodes.get_result(self.input).unwrap().into()]
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let input = inputs[0].as_tensor().unwrap();
        self.map_tensor(input).into()
    }

    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[crate::mir::inputs::MirValue],
        _: &mut crate::mir::kernel::GenericKernel,
    ) {
    }

    fn name(&self) -> String {
        "map_layout".to_string()
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn slice(&self, slices: [Range<usize>; R]) -> Tensor<R, T> {
        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            {
                let slices = slices.clone();
                move |shape| slice_shape(&slices, shape)
            },
            move |offset, strides| slice_strides(&slices, offset, strides),
        ))
    }

    pub fn transpose(&self, first_axis: usize, second_axis: usize) -> Tensor<R, T> {
        assert!(first_axis < self.rank());
        assert!(second_axis < self.rank());
        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            move |shape| {
                let mut shape: Box<[usize]> = shape.into();
                shape.swap(first_axis, second_axis);
                shape
            },
            move |offset, strides| {
                let mut strides: Box<[usize]> = strides.into();
                strides.swap(first_axis, second_axis);
                (offset, strides)
            },
        ))
    }

    pub fn t(&self) -> Tensor<R, T> {
        const {
            assert!(
                R >= 2,
                "The tensor must have at least 2 dimensions to transpose"
            )
        };
        let last_dim = self.rank() - 1;
        let second_last_dim = self.rank() - 2;
        self.transpose(last_dim, second_last_dim)
    }

    pub fn broadcast_as<const R2: usize>(&self, out_shape: [usize; R2]) -> Tensor<R2, T> {
        const {
            assert!(
                R2 >= R,
                "The output dimension must be more than the input dimension"
            )
        };

        let current_shape = *self.shape();

        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            move |_| out_shape.into(),
            move |offset, strides| {
                let mut new_strides = [0; R2];
                let mut current_shape_iter = current_shape.into_iter().rev().peekable();
                let mut strides = strides.iter().rev();
                for (new_strides_fill, new_shape) in out_shape.into_iter().enumerate().rev() {
                    let stride = if current_shape_iter.next_if_eq(&new_shape).is_some() {
                        *strides.next().unwrap()
                    } else {
                        _ = current_shape_iter.next_if_eq(&1);
                        0
                    };
                    new_strides[new_strides_fill] = stride;
                }
                assert_eq!(current_shape_iter.len(), 0, "failed to broadcast tensor: input shape {current_shape:?} is not compatible with output shape {out_shape:?}");
                (offset, new_strides.into())
            },
        ))
    }

    pub(crate) fn broadcast_together<const R2: usize, const R3: usize>(
        first: &Tensor<R, T>,
        second: &Tensor<R2, T>,
    ) -> (Tensor<R3, T>, Tensor<R3, T>)
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        const {
            assert!(
                R3 == if R > R2 { R } else { R2 },
                "The output dimension must be the maximum of the two input dimensions"
            )
        };

        let shape = if first.rank() > second.rank() {
            std::array::from_fn(|i| first.shape()[i])
        } else if first.rank() < second.rank() {
            std::array::from_fn(|i| second.shape()[i])
        } else {
            std::array::from_fn(|i| first.shape()[i].max(second.shape()[i]))
        };
        (first.broadcast_as(shape), second.broadcast_as(shape))
    }

    pub(crate) fn broadcast_then_elementwise_op<const R2: usize, const R3: usize>(
        first: &Tensor<R, T>,
        second: &Tensor<R2, T>,
        op: impl Fn(Tensor<R3, T>, Tensor<R3, T>) -> Tensor<R3, T>,
    ) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        let (b1, b2) = Self::broadcast_together(first, second);
        assert_eq!(b1.shape(), b2.shape());
        op(b1, b2)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_transpose() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let transposed = tensor.transpose(0, 1);
    let as_slice = transposed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 3.);
    assert_eq!(as_slice[[0, 2]], 5.);
    assert_eq!(as_slice[[1, 0]], 2.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[1, 2]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_broadcast_as() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.]];
    let tensor = Tensor::new(&device, &data);
    let broadcasted = tensor.broadcast_as([2, 2, 3]);
    println!("{broadcasted:?}");
    let as_slice = broadcasted.as_slice().await.unwrap();
    println!("{as_slice:?}");
    for i in 0..2 {
        assert_eq!(as_slice[[0, 0, i]], 1.);
        assert_eq!(as_slice[[0, 1, i]], 2.);
        assert_eq!(as_slice[[1, 0, i]], 3.);
        assert_eq!(as_slice[[1, 1, i]], 4.);
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_broadcast_together_first_larger() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data1 = [[1., 2.], [3., 4.]];
    let tensor1 = Tensor::new(&device, &data1);
    let data2 = [10., 20.];
    let tensor2 = Tensor::new(&device, &data2);
    let (b1, b2) = Tensor::broadcast_together(&tensor1, &tensor2);
    println!("{b1:?}");
    println!("{b2:?}");
    let as_slice1 = b1.as_slice().await.unwrap();
    let as_slice2 = b2.as_slice().await.unwrap();
    println!("{as_slice1:?}");
    println!("{as_slice2:?}");
    assert_eq!(as_slice1[[0, 0]], 1.);
    assert_eq!(as_slice1[[0, 1]], 2.);
    assert_eq!(as_slice1[[1, 0]], 3.);
    assert_eq!(as_slice1[[1, 1]], 4.);

    assert_eq!(as_slice2[[0, 0]], 10.);
    assert_eq!(as_slice2[[0, 1]], 20.);
    assert_eq!(as_slice2[[1, 0]], 10.);
    assert_eq!(as_slice2[[1, 1]], 20.);
}

#[cfg(test)]
#[tokio::test]
async fn test_broadcast_together_second_larger() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data1 = [[1., 2.], [3., 4.]];
    let tensor1 = Tensor::new(&device, &data1);
    let data2 = [10., 20.];
    let tensor2 = Tensor::new(&device, &data2);
    let (b2, b1) = Tensor::broadcast_together(&tensor2, &tensor1);
    println!("{b1:?}");
    println!("{b2:?}");
    let as_slice1 = b1.as_slice().await.unwrap();
    let as_slice2 = b2.as_slice().await.unwrap();
    println!("{as_slice1:?}");
    println!("{as_slice2:?}");
    assert_eq!(as_slice1[[0, 0]], 1.);
    assert_eq!(as_slice1[[0, 1]], 2.);
    assert_eq!(as_slice1[[1, 0]], 3.);
    assert_eq!(as_slice1[[1, 1]], 4.);

    assert_eq!(as_slice2[[0, 0]], 10.);
    assert_eq!(as_slice2[[0, 1]], 20.);
    assert_eq!(as_slice2[[1, 0]], 10.);
    assert_eq!(as_slice2[[1, 1]], 20.);
}

#[cfg(test)]
#[tokio::test]
async fn test_broadcast_together_same_size() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data1 = [[1.], [2.]];
    let tensor1 = Tensor::new(&device, &data1);
    let data2 = [[10., 20.]];
    let tensor2 = Tensor::new(&device, &data2);
    let (b2, b1) = Tensor::broadcast_together(&tensor2, &tensor1);
    println!("{b1:?}");
    println!("{b2:?}");
    let as_slice1 = b1.as_slice().await.unwrap();
    let as_slice2 = b2.as_slice().await.unwrap();
    println!("{as_slice1:?}");
    println!("{as_slice2:?}");
    assert_eq!(as_slice1[[0, 0]], 1.);
    assert_eq!(as_slice1[[0, 1]], 1.);
    assert_eq!(as_slice1[[1, 0]], 2.);
    assert_eq!(as_slice1[[1, 1]], 2.);

    assert_eq!(as_slice2[[0, 0]], 10.);
    assert_eq!(as_slice2[[0, 1]], 20.);
    assert_eq!(as_slice2[[1, 0]], 10.);
    assert_eq!(as_slice2[[1, 1]], 20.);
}
