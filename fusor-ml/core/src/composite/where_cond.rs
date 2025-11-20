use std::sync::Arc;

use crate::{
    DataType, Device, Tensor, TensorData,
    compute_graph::NodeIndex,
    layout::TILE_SIZE,
    mir::{
        inputs::MirValue, kernel::GenericKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    tensor::DataTypeEnum,
    visit_tiled::{
        MaybeQData, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn where_cond<D2>(self, on_true: &Tensor<R, D2>, on_false: &Tensor<R, D2>) -> Tensor<R, D2>
    where
        D2: DataType,
    {
        // Merge all compute graphs to ensure all tensors are in the same graph
        self.graph().merge(on_true.graph());
        self.graph().merge(on_false.graph());

        let operation = WhereCondOperation::new(
            self.key(),
            on_true.key(),
            on_false.key(),
            self.datatype(),
            on_true.datatype(),
            self.shape(),
        );
        let data = on_true.data();

        Tensor::from_parts(data.custom(Arc::new(operation)))
    }
}

#[derive(Debug, Clone)]
struct WhereCondOperation {
    pub(crate) condition: NodeIndex,
    pub(crate) on_true: NodeIndex,
    pub(crate) on_false: NodeIndex,
    pub(crate) condition_datatype: DataTypeEnum,
    pub(crate) output_datatype: DataTypeEnum,
    pub(crate) shape: Box<[usize]>,
}

impl WhereCondOperation {
    pub fn new(
        condition: NodeIndex,
        on_true: NodeIndex,
        on_false: NodeIndex,
        condition_datatype: DataTypeEnum,
        output_datatype: DataTypeEnum,
        shape: &[usize],
    ) -> Self {
        Self {
            condition,
            on_true,
            on_false,
            condition_datatype,
            output_datatype,
            shape: shape.into(),
        }
    }

    fn rank(&self) -> u32 {
        self.shape.len() as _
    }

    fn kernel(
        &self,
        device: &Device,
        _workgroup_shape: &WorkgroupShape,
        kernel: &mut GenericKernel,
    ) {
        let datatypes = vec![
            self.condition_datatype.into(),
            self.output_datatype.into(),
            self.output_datatype.into(),
            self.output_datatype.into(),
        ];

        build_visit_tiled_kernel(
            device,
            &self.shape,
            TILE_SIZE,
            datatypes,
            |_kernel, indexes, tensors, values| {
                let condition_value = &values[0];
                let on_true_value = &values[1];
                let on_false_value = &values[2];
                let output_index = &indexes[3];
                let out_tensor = &tensors[3];

                let condition_datatype = self.condition_datatype;

                format!(
                    "{out_tensor}[{output_index}] = select({on_false_value}, {on_true_value}, {condition_value} != {condition_datatype}(0));"
                )
            },
            kernel,
        );
    }
}

impl Operation for WhereCondOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.shape, device)
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

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.condition);
        f(self.on_true);
        f(self.on_false);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let condition = nodes.cached_results.get(&self.condition).unwrap();
        let on_true = nodes.cached_results.get(&self.on_true).unwrap();
        let on_false = nodes.cached_results.get(&self.on_false).unwrap();

        let output_tensor = TensorData::new_for_shape(
            condition.device(),
            condition.layout().shape(),
            self.output_datatype,
        );

        vec![
            condition.clone().into(),
            on_true.clone().into(),
            on_false.clone().into(),
            output_tensor.into(),
        ]
    }

    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        self.kernel(&graph.device, workgroup_shape, kernel);
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs[3].clone()
    }

    fn name(&self) -> String {
        format!("where_cond_{}_{}", self.rank(), self.output_datatype)
    }

    fn output_layout(
        &self,
        layouts: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let on_true = layouts.get(&self.on_true).unwrap();
        on_true.clone()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_where_cond() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = Tensor::arange(&device, 0., 10.);
    let even = Tensor::arange(&device, 0, 10) % 2;
    let zero = Tensor::splat(&device, 0., *data.shape());

    let data_where_even = even.where_cond(&data, &zero);

    let result = data_where_even.as_slice().await.unwrap();
    println!("result: {result:?}");

    assert_eq!(result[[0]], 0.);
    assert_eq!(result[[1]], 1.);
    assert_eq!(result[[2]], 0.);
    assert_eq!(result[[3]], 3.);
    assert_eq!(result[[4]], 0.);
    assert_eq!(result[[5]], 5.);
    assert_eq!(result[[6]], 0.);
    assert_eq!(result[[7]], 7.);
    assert_eq!(result[[8]], 0.);
    assert_eq!(result[[9]], 9.);
}
