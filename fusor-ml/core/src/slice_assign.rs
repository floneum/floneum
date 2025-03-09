use std::{ops::Range, sync::OnceLock};

use wgpu::CommandEncoder;

use crate::{
    PerformanceQueries, TILE_SIZE, Tensor, TensorData, compute_graph::AnyComputeKey,
    visit_tiled::VisitTiledKernel,
};

pub(crate) struct SliceAssignOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) value: AnyComputeKey,
    pub(crate) slices: Box<[Range<usize>]>,
}

impl SliceAssignOperation {
    pub fn new(input: AnyComputeKey, value: AnyComputeKey, slices: Box<[Range<usize>]>) -> Self {
        Self {
            input,
            value,
            slices,
        }
    }
}

pub(crate) struct UntypedSliceAssignKernel {
    slices: Box<[Range<usize>]>,
    sparse_kernel: OnceLock<VisitTiledKernel>,
}

impl UntypedSliceAssignKernel {
    pub(crate) fn new(slices: &[Range<usize>]) -> Self {
        Self {
            slices: slices.into(),
            sparse_kernel: OnceLock::new(),
        }
    }

    pub fn run_with_query(
        &self,
        target: &TensorData,
        value: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let rank = target.layout().rank();
        let datatype = target.datatype();

        let create_kernel = || {
            let datatypes = vec![datatype; 2];

            VisitTiledKernel::new(
                rank as u32,
                TILE_SIZE,
                false,
                datatypes,
                |_, indexes, tensors| {
                    let target_index = &indexes[0];
                    let value_index = &indexes[1];
                    let target_tensor = &tensors[0];
                    let value_tensor = &tensors[1];
                    format!("{target_tensor}[{target_index}] = {value_tensor}[{value_index}];")
                },
            )
        };
        let kernel = self.sparse_kernel.get_or_init(create_kernel);

        let sliced = target.slice(&self.slices);
        assert_eq!(sliced.layout().shape(), value.layout().shape());
        let tensors = vec![&sliced, value];
        kernel.run_with_query(tensors, query, command_encoder);
        target.clone()
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        self.add_slice_assign(value, slices)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_slice_assign() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let value_tensor = Tensor::new(&device, &[[10., 20.], [30., 40.]]);
    let tensor = tensor.slice_assign([0..2, 0..2], &value_tensor);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0]], 10.);
    assert_eq!(as_slice[[0, 1]], 20.);
    assert_eq!(as_slice[[1, 0]], 30.);
    assert_eq!(as_slice[[1, 1]], 40.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}
