use fusor_gguf::GgmlType;

use crate::{
    Device, QMatrix,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::{Constraint, WorkgroupShape},
    },
    quantized::matmul::{
        QMatMulOperation,
        sgemm::{
            chunked::{SGEMM_SUBGROUP_THERADS_PER_BLOCK, chunked_sgemm},
            general::general_sgemm,
        },
    },
};

mod chunked;
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
    _: &crate::compute_graph::ComputeGraphInner,
) {
    match input_b.datatype {
        GgmlType::Q6K => {
            chunked_sgemm(
                op,
                generic_kernel,
                workgroup_size,
                input_a,
                input_b,
                output,
                _n_size,
                _m_size,
                k_size,
            );
        }
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
    batch_size: u32,
) -> [u32; 3] {
    match matrix.datatype() {
        GgmlType::Q6K => [
            m.div_ceil(workgroup_shape.y() * 4),
            n.div_ceil(workgroup_shape.x() * 4),
            batch_size.div_ceil(workgroup_shape.z()),
        ],
        _ => [
            n.div_ceil(workgroup_shape.x()),
            m.div_ceil(workgroup_shape.y()),
            batch_size.div_ceil(workgroup_shape.z()),
        ],
    }
}

pub(crate) fn workgroup_shape_constraints(
    matrix: &QMatrix,
    _device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    if matrix.datatype() == GgmlType::Q6K {
        let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
        constraints.add_constraint(0, Constraint::equals(4));
        constraints.add_constraint(1, Constraint::equals(4));
        constraints.add_constraint(2, Constraint::equals(1));
        return constraints;
    }
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    let second_dim = matrix.shape()[1];
    let second_dim_block_width = second_dim.div_ceil(matrix.datatype().block_size());
    constraints.add_constraint(
        1,
        Constraint::less_than_or_equals(second_dim_block_width as _),
    );
    constraints.add_constraint(2, Constraint::equals(1));
    constraints
}
