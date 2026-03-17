use std::{fmt::Write, sync::Arc};

use crate::{
    DataTypeEnum, FloatDataType, Layout, LazyTensorData, Tensor, TensorData,
    TensorInfo,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    visit_tiled::distribute_workgroups,
};

const BLOCKSIZE: u32 = 256;

#[derive(Debug, Clone)]
struct GatherLastOperation {
    values: NodeIndex,
    indexes: NodeIndex,
    rows: usize,
    width: usize,
    datatype: DataTypeEnum,
}

impl Operation for GatherLastOperation {
    fn workgroup_shape_constraints(&self, _: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(BLOCKSIZE));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, _: &WorkgroupShape, _: &[MirValue]) -> [u32; 3] {
        distribute_workgroups((self.rows as u32).div_ceil(BLOCKSIZE))
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.values);
        f(self.indexes);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let values = nodes.get_cached_result(self.values).unwrap().clone();
        let indexes = nodes.get_cached_result(self.indexes).unwrap().clone();
        let output = TensorData::new_for_shape(&nodes.device(), &[self.rows], self.datatype);
        vec![values.into(), indexes.into(), output.into()]
    }

    fn output(&self, _: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs[2].clone()
    }

    fn build_kernel(
        &self,
        _: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let values = kernel.add_tensor_input(2, false, self.datatype);
        let indexes = kernel.add_tensor_input(1, false, DataTypeEnum::U32);
        let output = kernel.add_tensor_input(1, true, self.datatype);
        let workgroup_local_index = kernel.workgroup_local_index();
        let linearized_workgroup = workgroup_shape.linearized_workgroup_index(kernel);

        writeln!(
            kernel,
            "let row = ({linearized_workgroup}) * {BLOCKSIZE}u + {workgroup_local_index};"
        )
        .unwrap();
        writeln!(kernel, "if row < {} {{", output.shape_binding(0)).unwrap();
        write!(kernel, "let index_offset = ").unwrap();
        indexes.strided_index(kernel, ["row"]);
        writeln!(kernel, ";").unwrap();
        writeln!(kernel, "let column = {indexes}[index_offset];").unwrap();
        write!(kernel, "let value_offset = ").unwrap();
        values.strided_index(kernel, ["row", "column"]);
        writeln!(kernel, ";").unwrap();
        write!(kernel, "let output_offset = ").unwrap();
        output.strided_index(kernel, ["row"]);
        writeln!(kernel, ";").unwrap();
        writeln!(kernel, "{output}[output_offset] = {values}[value_offset];").unwrap();
        writeln!(kernel, "}}").unwrap();
    }

    fn name(&self) -> String {
        format!("gather_last_{}x{}", self.rows, self.width)
    }

    fn output_layout(
        &self,
        _: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        crate::TensorLayoutInfo::new(Layout::contiguous(&[self.rows]), self.datatype)
    }
}

impl<D: FloatDataType> Tensor<2, D> {
    pub fn gather_last(&self, indexes: &Tensor<1, u32>) -> Tensor<1, D> {
        assert_eq!(
            self.shape()[0],
            indexes.shape()[0],
            "gather_last expects one index per row"
        );

        let rows = self.shape()[0];
        let width = self.shape()[1];
        let operation = GatherLastOperation {
            values: self.key(),
            indexes: indexes.key(),
            rows,
            width,
            datatype: self.datatype(),
        };
        let device = self.device().clone();
        let key = device.compute_graph().create_custom(Arc::new(operation));
        Tensor::from_parts(LazyTensorData::from_parts(
            device,
            TensorInfo::new(vec![rows].into_boxed_slice(), self.datatype()),
            key,
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_gather_last() {
    use crate::Device;

    let device = Device::test_instance();
    let values = Tensor::new(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let indexes = Tensor::new(&device, &[2u32, 0]);
    let gathered = values.gather_last(&indexes);

    let output = gathered.as_slice().await.unwrap();
    assert_eq!(output[[0]], 3.0);
    assert_eq!(output[[1]], 4.0);
}

