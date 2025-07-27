use fusor_gguf::GgmlType;

use crate::{
    Device, QMatrix,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{
        QMatMulOperation,
        sgemm::general::general_sgemm,
        sgemv::{
            SGEMV_CHUNK_SIZE, q_8_0::Q_8_0_SGEMV_CHUNK_SIZE, q_n::Q_N_SGEMV_CHUNK_SIZE,
            q4k::Q4K_SGEMV_CHUNK_SIZE, q6k::Q6K_SGEMV_CHUNK_SIZE,
        },
    },
};

mod general;

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
    match op.matrix.datatype {
        _ => general_sgemm(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
    }
}

pub(crate) fn dispatch_size(
    workgroup_shape: &WorkgroupShape,
    matrix: &QMatrix,
    n: u32,
    m: u32,
) -> [u32; 3] {
    [
        (n as u32).div_ceil(workgroup_shape.x()),
        (m as u32).div_ceil(workgroup_shape.y()),
        1,
    ]
}

pub(crate) fn workgroup_shape_constraints(
    matrix: &QMatrix,
    device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();

    constraints.add_constraint(0, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}
