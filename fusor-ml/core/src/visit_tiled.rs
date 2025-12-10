use std::fmt::{Display, Write};

use fusor_gguf::GgmlType;

use crate::{
    DataTypeEnum, Device, Layout, QMatrix, TensorData, dequantize_block, dequantize_mat4x4_block,
    mir::{
        inputs::{MirValue, QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
    },
};

#[derive(Clone, Debug)]
pub(crate) enum MaybeQData {
    Tensor(TensorData),
    QMatrix(QMatrix),
}

impl MaybeQData {
    pub(crate) fn device(&self) -> &crate::Device {
        match self {
            MaybeQData::Tensor(tensor) => tensor.device(),
            MaybeQData::QMatrix(qmatrix) => qmatrix.device(),
        }
    }

    pub(crate) fn layout(&self) -> Layout {
        match self {
            MaybeQData::Tensor(tensor) => tensor.layout().clone(),
            MaybeQData::QMatrix(qmatrix) => Layout::contiguous(qmatrix.shape()),
        }
    }

    pub(crate) fn datatype(&self) -> VisitTiledInputType {
        match self {
            MaybeQData::Tensor(tensor) => VisitTiledInputType::Dequantized(tensor.datatype()),
            MaybeQData::QMatrix(qmatrix) => VisitTiledInputType::Quantized(qmatrix.datatype()),
        }
    }

    pub(crate) fn owned(&self) -> bool {
        match self {
            MaybeQData::Tensor(tensor) => tensor.owned(),
            MaybeQData::QMatrix(_) => false,
        }
    }
}

impl From<TensorData> for MaybeQData {
    fn from(tensor: TensorData) -> Self {
        Self::Tensor(tensor)
    }
}

impl From<&TensorData> for MaybeQData {
    fn from(tensor: &TensorData) -> Self {
        Self::Tensor(tensor.clone())
    }
}

impl From<QMatrix> for MaybeQData {
    fn from(qmatrix: QMatrix) -> Self {
        Self::QMatrix(qmatrix)
    }
}

impl From<&QMatrix> for MaybeQData {
    fn from(qmatrix: &QMatrix) -> Self {
        Self::QMatrix(qmatrix.clone())
    }
}

impl From<MaybeQData> for MirValue {
    fn from(val: MaybeQData) -> Self {
        match val {
            MaybeQData::Tensor(tensor) => MirValue::Tensor(tensor.clone()),
            MaybeQData::QMatrix(qmatrix) => MirValue::QMatrix(qmatrix.clone()),
        }
    }
}

impl TryFrom<MirValue> for MaybeQData {
    type Error = ();

    fn try_from(value: MirValue) -> Result<Self, Self::Error> {
        match value {
            MirValue::Tensor(tensor) => Ok(MaybeQData::Tensor(tensor)),
            MirValue::QMatrix(qmatrix) => Ok(MaybeQData::QMatrix(qmatrix)),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum VisitTiledInputType {
    Quantized(GgmlType),
    Dequantized(DataTypeEnum),
}

impl From<DataTypeEnum> for VisitTiledInputType {
    fn from(ty: DataTypeEnum) -> Self {
        Self::Dequantized(ty)
    }
}

impl From<GgmlType> for VisitTiledInputType {
    fn from(ty: GgmlType) -> Self {
        Self::Quantized(ty)
    }
}

pub(crate) enum MaybeQTensorInput {
    Tensor(TensorInput),
    QTensor(QMatrixInput),
}

impl Display for MaybeQTensorInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaybeQTensorInput::Tensor(tensor) => tensor.fmt(f),
            MaybeQTensorInput::QTensor(tensor) => tensor.fmt(f),
        }
    }
}

pub(crate) fn build_visit_tiled_kernel(
    device: &Device,
    shape: &[usize],
    tile_size: u32,
    datatypes: Vec<VisitTiledInputType>,
    modify_data: impl FnMut(&mut GenericKernel, &[String], &[MaybeQTensorInput], &[String]) -> String,
    kernel: &mut GenericKernel,
) {
    build_tiled_map_kernel(device, shape, tile_size, &datatypes, kernel, modify_data);
}

fn build_tiled_map_kernel(
    _device: &Device,
    shape: &[usize],
    tile_size: u32,
    datatypes: &[VisitTiledInputType],
    kernel: &mut GenericKernel,
    mut modify_data: impl FnMut(
        &mut GenericKernel,
        &[String],
        &[MaybeQTensorInput],
        &[String],
    ) -> String,
) {
    let rank = shape.len() as u32;
    let tensors = datatypes
        .iter()
        .map(|ty| match ty {
            VisitTiledInputType::Quantized(ty) => {
                MaybeQTensorInput::QTensor(kernel.add_q_matrix_input(rank, *ty))
            }
            VisitTiledInputType::Dequantized(ty) => {
                MaybeQTensorInput::Tensor(kernel.add_tensor_input(rank, true, *ty))
            }
        })
        .collect::<Vec<_>>();

    fn first_tensor_input(tensors: &[MaybeQTensorInput]) -> &TensorInput {
        tensors
            .iter()
            .find_map(|i| match i {
                MaybeQTensorInput::Tensor(tensor) => Some(tensor),
                MaybeQTensorInput::QTensor(_) => None,
            })
            .unwrap()
    }

    let quantized_block = datatypes
        .iter()
        .enumerate()
        .find_map(|(i, tensor)| match tensor {
            VisitTiledInputType::Quantized(tensor) => Some((
                tensor,
                match &tensors[i] {
                    MaybeQTensorInput::Tensor(_) => panic!("Expected a qtensor"),
                    MaybeQTensorInput::QTensor(tensor) => tensor,
                },
            )),
            _ => None,
        });

    let first = first_tensor_input(&tensors);

    // Compute flat global thread index
    let num_workgroups = kernel.num_workgroups();
    let workgroup_id = kernel.workgroup_index();
    let local_id = kernel.workgroup_local_index();

    writeln!(kernel, "// Compute flat global thread index").unwrap();
    writeln!(kernel, "let workgroup_flat_id = {workgroup_id}.x + {workgroup_id}.y * {num_workgroups}.x + {workgroup_id}.z * {num_workgroups}.x * {num_workgroups}.y;").unwrap();
    writeln!(
        kernel,
        "let global_thread_id = workgroup_flat_id * BLOCKSIZE + {local_id};"
    )
    .unwrap();
    writeln!(kernel).unwrap();

    if let Some((quantized_type, quantized_input)) = quantized_block {
        let block_size = quantized_type.block_size() as u32;

        writeln!(
            kernel,
            "// Process {} elements per thread (quantized)",
            tile_size
        )
        .unwrap();
        writeln!(kernel, "var tile_offset = 0u;").unwrap();
        writeln!(
            kernel,
            "while (tile_offset < {}u) {{",
            tile_size
        )
        .unwrap();
        writeln!(
            kernel,
            "let flat_index = global_thread_id * {}u + tile_offset;",
            tile_size
        )
        .unwrap();
        writeln!(kernel).unwrap();

        // Convert flat index to multi-dimensional indices
        writeln!(kernel, "// Convert flat index to multi-dimensional indices").unwrap();
        writeln!(kernel, "var remaining_index = flat_index;").unwrap();

        // Generate strides for all dimensions (row-major order)
        for i in (0..rank).rev() {
            let shape_binding = first.shape_binding(i);
            if i == rank - 1 {
                // Last dimension: need to account for block size in quantized case
                writeln!(kernel, "let dim_{i} = remaining_index % {shape_binding};").unwrap();
                writeln!(
                    kernel,
                    "remaining_index = remaining_index / {shape_binding};"
                )
                .unwrap();
            } else {
                writeln!(kernel, "let dim_{i} = remaining_index % {shape_binding};").unwrap();
                writeln!(
                    kernel,
                    "remaining_index = remaining_index / {shape_binding};"
                )
                .unwrap();
            }
        }
        writeln!(kernel).unwrap();

        // For quantized data, we need to handle block alignment for the last dimension
        writeln!(
            kernel,
            "// Compute block-aligned indices for quantized data"
        )
        .unwrap();
        writeln!(
            kernel,
            "let block_dim_{} = dim_{} / {};",
            rank - 1,
            rank - 1,
            block_size
        )
        .unwrap();
        writeln!(
            kernel,
            "let block_offset = dim_{} % {};",
            rank - 1,
            block_size
        )
        .unwrap();

        // Load the quantized block
        write!(kernel, "let chunk = &{quantized_input}[").unwrap();
        quantized_input.strided_index(
            kernel,
            (0..rank - 1)
                .map(|i| format!("dim_{i}"))
                .chain(std::iter::once(format!("block_dim_{}", rank - 1))),
        );
        writeln!(kernel, "];").unwrap();
        
        writeln!(kernel, "let sub_block_index = block_offset / 16u;").unwrap();
        let handled = dequantize_mat4x4_block(
            kernel,
            *quantized_type,
            "sub_block_index",
            "chunk".to_string(),
            DataTypeEnum::F32,
            |val, kernel| {
                writeln!(kernel, "let sub_block_offset = block_offset % 16u;").unwrap();
                writeln!(
                    kernel,
                    "let items_limit = min(16u - sub_block_offset, {}u - tile_offset);",
                    tile_size
                )
                .unwrap();
                writeln!(kernel, "for (var i = 0u; i < items_limit; i++) {{").unwrap();
                writeln!(kernel, "let local_idx = sub_block_offset + i;").unwrap();
                writeln!(kernel, "let col = local_idx / 4u;").unwrap();
                writeln!(kernel, "let row = local_idx % 4u;").unwrap();
                writeln!(kernel, "let val = {val}[col][row];").unwrap();

                // Recalculate dimensions for the current element
                for i in 0..rank {
                    if i == rank - 1 {
                        writeln!(kernel, "let current_dim_{i} = dim_{i} + i;").unwrap();
                    } else {
                        writeln!(kernel, "let current_dim_{i} = dim_{i};").unwrap();
                    }
                }

                first_tensor_input(&tensors).check_bounds(
                    kernel,
                    (0..rank).map(|i| format!("current_dim_{i}")),
                    |kernel| {
                        let mut values = Vec::new();
                        for (index, tensor) in tensors.iter().enumerate() {
                            match tensor {
                                MaybeQTensorInput::Tensor(tensor) => {
                                    writeln!(kernel, "let index_{index} = ",).unwrap();
                                    tensor.strided_index(
                                        kernel,
                                        (0..rank).map(|i| format!("current_dim_{i}")),
                                    );
                                    writeln!(kernel, ";").unwrap();
                                    values.push(format!("{tensor}[index_{index}]"));
                                }
                                MaybeQTensorInput::QTensor(_) => {
                                    values.push("val".to_string());
                                }
                            }
                        }
                        let indexes = (0..datatypes.len())
                            .map(|i| format!("index_{i}"))
                            .collect::<Vec<_>>();

                        let modify_data = modify_data(kernel, &indexes, &tensors, &values);
                        writeln!(kernel, "{modify_data}").unwrap();
                    },
                );
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "tile_offset += items_limit;").unwrap();
            },
        );

        assert!(handled);

        writeln!(kernel, "}}").unwrap();
    } else {
        writeln!(kernel, "// Process {} elements per thread", tile_size).unwrap();
        writeln!(
            kernel,
            "for (var tile_offset = 0u; tile_offset < {}u; tile_offset++) {{",
            tile_size
        )
        .unwrap();
        writeln!(
            kernel,
            "let flat_index = global_thread_id * {}u + tile_offset;",
            tile_size
        )
        .unwrap();
        writeln!(kernel).unwrap();

        // Convert flat index to multi-dimensional indices
        writeln!(kernel, "// Convert flat index to multi-dimensional indices").unwrap();
        writeln!(kernel, "var remaining_index = flat_index;").unwrap();

        // Generate indices for all dimensions (row-major order)
        for i in (0..rank).rev() {
            let shape_binding = first.shape_binding(i);
            writeln!(kernel, "let dim_{i} = remaining_index % {shape_binding};").unwrap();
            if i > 0 {
                writeln!(
                    kernel,
                    "remaining_index = remaining_index / {shape_binding};"
                )
                .unwrap();
            }
        }
        writeln!(kernel).unwrap();

        // Bounds check and data access
        first_tensor_input(&tensors).check_bounds(
            kernel,
            (0..rank).map(|i| format!("dim_{i}")),
            |kernel| {
                for (index, tensor) in tensors.iter().enumerate() {
                    writeln!(kernel, "let index_{index} = ",).unwrap();
                    match tensor {
                        MaybeQTensorInput::Tensor(tensor) => {
                            tensor.strided_index(kernel, (0..rank).map(|i| format!("dim_{i}")))
                        }
                        MaybeQTensorInput::QTensor(_) => unreachable!(),
                    }
                    writeln!(kernel, ";").unwrap();
                }
                let indexes = (0..datatypes.len())
                    .map(|i| format!("index_{i}"))
                    .collect::<Vec<_>>();
                let values = tensors
                    .iter()
                    .enumerate()
                    .map(|(i, tensor)| format!("{tensor}[index_{i}]"))
                    .collect::<Vec<_>>();

                let modify_data = modify_data(kernel, &indexes, &tensors, &values);
                writeln!(kernel, "{modify_data}").unwrap();
            },
        );

        writeln!(kernel, "}}").unwrap();
    }
}

pub(crate) fn titled_map_workgroup_size_constraints(
    _shape: &[usize],
    device: &crate::Device,
) -> WorkgroupShapeConstraints {
    let mut constraints = WorkgroupShapeConstraints::new();

    // For flattened dispatch, we can use all three dimensions of the workgroup
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::more_than_or_equals(
            device.limits().min_subgroup_size,
        ),
    );

    constraints
}

pub(crate) fn titled_map_dispatch_size(
    tile_size: u32,
    workgroup_shape: WorkgroupShape,
    shape: &[usize],
) -> [u32; 3] {
    // Calculate total number of elements
    let total_elements: u64 = shape.iter().map(|&x| x as u64).product();

    // Calculate total number of tiles needed (each thread processes tile_size elements)
    let total_tiles = total_elements.div_ceil(tile_size as u64) as u32;

    // Calculate total workgroups needed
    let workgroup_volume = workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z();
    let total_workgroups = total_tiles.div_ceil(workgroup_volume);

    // Distribute workgroups across x, y, z dimensions
    // Try to keep it as flat as possible for better occupancy
    let max_x = 65535u32; // WebGPU limit
    let max_y = 65535u32;

    let workgroup_size_x = total_workgroups.min(max_x);
    let remaining = total_workgroups.div_ceil(workgroup_size_x);
    let workgroup_size_y = remaining.min(max_y);
    let workgroup_size_z = total_workgroups
        .div_ceil(workgroup_size_x * workgroup_size_y)
        .max(1);

    [workgroup_size_x, workgroup_size_y, workgroup_size_z]
}
