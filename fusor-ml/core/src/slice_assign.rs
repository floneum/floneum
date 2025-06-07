use std::ops::Range;

use crate::{
    TILE_SIZE, Tensor,
    compute_graph::{AnyComputeKey, ComputeGraphInner},
    mir::operation::Operation,
    visit_tiled::{
        MaybeQData, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

#[derive(Clone, Debug)]
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

    pub fn rank(&self) -> usize {
        self.slices.len()
    }
}

impl Operation for SliceAssignOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(self.rank() as _)
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let inputs: Box<[_]> = inputs
            .iter()
            .map(|input| {
                let tensor: MaybeQData = input.clone().try_into().unwrap();
                tensor
            })
            .collect();
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &inputs)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.value);
        f(self.input);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<crate::mir::inputs::MirValue> {
        let input = nodes.cached_results.get(&self.input).unwrap();
        let input = input.slice(&self.slices);
        let value = nodes.get_result_or_qmatrix(self.value).unwrap();

        vec![input.into(), value.into()]
    }

    fn build_kernel(
        &self,
        nodes: &ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        kernel: &mut crate::mir::kernel::GenericKernel,
    ) -> crate::mir::inputs::MirValue {
        let input: MaybeQData = inputs[0].clone().try_into().unwrap();
        let value: MaybeQData = inputs[1].clone().try_into().unwrap();
        assert_eq!(input.layout().shape(), value.layout().shape());
        let rank = input.layout().rank();
        let datatype = input.datatype();

        let datatypes = vec![datatype.into(); 2];

        build_visit_tiled_kernel(
            rank as u32,
            TILE_SIZE,
            datatypes,
            |_, indexes, tensors, values| {
                let target_index = &indexes[0];
                let target_tensor = &tensors[0];
                let value = &values[1];
                format!("{target_tensor}[{target_index}] = {value};")
            },
            kernel,
        );

        nodes.get_result_or_qmatrix(self.input).unwrap().into()
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
