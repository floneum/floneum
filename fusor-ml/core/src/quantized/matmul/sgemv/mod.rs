use fusor_gguf::GgmlType;

use crate::{
    Device,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{
        QMatMulOperation,
        sgemv::{
            general::general_sgemv, q_8_0::q_8_0_sgemv, q_n::q_n_sgemv, q4k::q4k_sgemv,
            q6k::q6k_sgemv,
        },
    },
};
use crate::{
    QMatrix,
    quantized::matmul::sgemv::{
        q_8_0::Q_8_0_SGEMV_CHUNK_SIZE, q_n::Q_N_SGEMV_CHUNK_SIZE, q4k::Q4K_SGEMV_CHUNK_SIZE,
        q6k::Q6K_SGEMV_CHUNK_SIZE,
    },
};

mod general;
pub mod q4k;
pub mod q6k;
pub mod q_8_0;
pub mod q_n;

pub(crate) const SGEMV_CHUNK_SIZE: u32 = 2; // This is the size of the chunk each thread will process at a time
pub(crate) const SGEMV_VECTOR_SIZE: u32 = 4; // This is the size of the chunk we will dot at a time

#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemv(
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
    graph: &crate::compute_graph::ComputeGraphInner,
) {
    match op.matrix.datatype {
        GgmlType::Q6K => q6k_sgemv(
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
        GgmlType::Q4K => q4k_sgemv(
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
        GgmlType::Q4_0 | GgmlType::Q5_0 => q_n_sgemv(
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
        GgmlType::Q8_0 => q_8_0_sgemv(
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
        _ => general_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
            graph,
        ),
    }
}

pub(crate) fn dispatch_size(matrix: &QMatrix, n: u32, _m: u32, batch_size: u32) -> [u32; 3] {
    if matrix.datatype == GgmlType::Q6K {
        return [n.div_ceil(Q6K_SGEMV_CHUNK_SIZE * 2), 1, batch_size];
    }
    if matrix.datatype == GgmlType::Q4K {
        return [n.div_ceil(Q4K_SGEMV_CHUNK_SIZE * 2), 1, batch_size];
    }
    if matches!(matrix.datatype, GgmlType::Q4_0 | GgmlType::Q5_0) {
        return [n.div_ceil(Q_N_SGEMV_CHUNK_SIZE * 2), 1, batch_size];
    }
    if matches!(matrix.datatype, GgmlType::Q8_0) {
        return [n.div_ceil(Q_8_0_SGEMV_CHUNK_SIZE * 2), 1, batch_size];
    }
    [n.div_ceil(SGEMV_CHUNK_SIZE * 2), 1, batch_size]
}

pub(crate) fn workgroup_shape_constraints(
    _: &QMatrix,
    device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    let limits = device.limits();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::more_than_or_equals(limits.min_subgroup_size),
    );
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::less_than_or_equals(limits.max_subgroup_size),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}
