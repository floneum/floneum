use crate::{
    Device, QMatrix,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{QMatMulOperation, sgemm::general::general_sgemm},
};

mod general;

#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
) {
    general_sgemm(
        op,
        generic_kernel,
        workgroup_size,
        input_a,
        input_b,
        output,
        _n_size,
        _m_size,
        k_size,
    )
}

pub(crate) fn dispatch_size(
    workgroup_shape: &WorkgroupShape,
    _matrix: &QMatrix,
    n: u32,
    m: u32,
) -> [u32; 3] {
    [
        n.div_ceil(workgroup_shape.x()),
        m.div_ceil(workgroup_shape.y()),
        1,
    ]
}

pub(crate) fn workgroup_shape_constraints(
    _matrix: &QMatrix,
    _device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();

    constraints.add_constraint(0, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}
