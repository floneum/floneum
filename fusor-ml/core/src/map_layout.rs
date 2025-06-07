use std::{fmt::Debug, ops::Range, sync::Arc};

use crate::{
    DataType, Layout, Tensor, TensorData, compute_graph::AnyComputeKey, mir::operation::Operation,
    slice_shape, slice_strides,
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

    pub fn broadcast<const R2: usize>(&self, out_shape: [usize; R2]) -> Tensor<R2, T> {
        const {
            assert!(
                R2 == R + 1,
                "The output dimension must be one more than the input dimension"
            )
        };

        let new_dim = self
            .shape()
            .iter()
            .zip(out_shape.iter())
            .take_while(|(a, b)| a == b)
            .count();
        assert_eq!(self.shape()[..new_dim], out_shape[..new_dim]);
        assert_eq!(self.shape()[new_dim..], out_shape[new_dim + 1..]);

        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            move |_| out_shape.into(),
            move |offset, strides| {
                let mut new_strides = [0; R2];
                let mut new_strides_fill = 0;
                for (i, stride) in strides.iter().enumerate() {
                    if i == new_dim {
                        new_strides[new_strides_fill] = 0;
                        new_strides_fill += 1;
                    }
                    new_strides[new_strides_fill] = *stride;
                    new_strides_fill += 1;
                }
                (offset, new_strides.into())
            },
        ))
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
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 3.);
    assert_eq!(as_slice[[0, 2]], 5.);
    assert_eq!(as_slice[[1, 0]], 2.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[1, 2]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_broadcast() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.]];
    let tensor = Tensor::new(&device, &data);
    let broadcasted = tensor.broadcast([2, 2, 3]);
    let as_slice = broadcasted.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    for i in 0..2 {
        assert_eq!(as_slice[[0, 0, i]], 1.);
        assert_eq!(as_slice[[0, 1, i]], 2.);
        assert_eq!(as_slice[[1, 0, i]], 3.);
        assert_eq!(as_slice[[1, 1, i]], 4.);
    }
}
