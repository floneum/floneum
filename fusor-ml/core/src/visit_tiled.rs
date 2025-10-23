use std::fmt::{Display, Write};

use fusor_gguf::GgmlType;

use crate::{
    DataTypeEnum, Layout, QMatrix, TensorData, dequantize_block,
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
    shape: &[usize],
    tile_size: u32,
    datatypes: Vec<VisitTiledInputType>,
    modify_data: impl FnMut(&mut GenericKernel, &[String], &[MaybeQTensorInput], &[String]) -> String,
    kernel: &mut GenericKernel,
) {
    build_tiled_map_kernel(shape, tile_size, &datatypes, kernel, modify_data);
}

fn build_tiled_map_kernel(
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
    let global_id = kernel.global_id();
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
    let mut global_indexes = Vec::new();
    for index in ["x", "y"].iter().take(rank as usize) {
        global_indexes.push(format!("{global_id}.{index}"));
    }
    // Handle the z dimension if the rank is greater than 2. We only need to handle
    // the dimensions that are greater than 1.
    let shape_dims_after_2_greater_than_1: Box<[_]> = shape
        .iter()
        .enumerate()
        .skip(2)
        .map(|(dim, shape)| (*shape > 1).then_some(dim))
        .collect();
    // If there is only one interesting dimension, just set that to the z index directly.
    let interesting_extra_dims = shape_dims_after_2_greater_than_1
        .iter()
        .filter(|&x| x.is_some())
        .count();
    let less_than_two_interesting_extra_dims = interesting_extra_dims < 2;

    let scaled_tile_size = scaled_tile_size(shape, tile_size);

    if less_than_two_interesting_extra_dims {
        for index in &shape_dims_after_2_greater_than_1 {
            if index.is_some() {
                global_indexes.push(format!("{global_id}.z"));
            } else {
                global_indexes.push("0u".to_string());
            }
        }
    } else {
        writeln!(kernel, "var remaining_z = {global_id}.z;").unwrap();
        for index in (0..rank).skip(2) {
            let size = first.shape_binding(index);
            writeln!(kernel, "let z_{index} = remaining_z % {size};").unwrap();
            writeln!(kernel, "remaining_z = remaining_z / {size};").unwrap();
            global_indexes.push(format!("z_{index}"));
        }
    }

    if let Some((quantized_type, quantized_input)) = quantized_block {
        for (i, index) in global_indexes.iter().enumerate() {
            let chunk_size = if i == rank as usize - 1 {
                quantized_type.block_size() as u32
            } else {
                scaled_tile_size
            };
            writeln!(kernel, "let tile_index_{i} = {index} * {chunk_size};").unwrap();
        }
        writeln!(kernel, "\n").unwrap();

        for i in 0..rank - 1 {
            writeln!(kernel, "for (var local_index_{i} = 0u; local_index_{i} < {scaled_tile_size}; local_index_{i}++) {{").unwrap();
            writeln!(
                kernel,
                "let merged_index_{i} = tile_index_{i} + local_index_{i};"
            )
            .unwrap();
        }

        write!(kernel, "let chunk_index = ").unwrap();
        quantized_input.strided_index(
            kernel,
            (0..rank - 1)
                .map(|i| format!("merged_index_{i}"))
                .chain(std::iter::once(format!("tile_index_{}", rank - 1))),
        );
        writeln!(kernel, ";").unwrap();
        writeln!(kernel, "let chunk = {quantized_input}[chunk_index];").unwrap();

        dequantize_block(
            kernel,
            *quantized_type,
            "chunk".to_string(),
            DataTypeEnum::F32,
            |i, data, kernel| {
                writeln!(
                    kernel,
                    "let merged_index_{} = tile_index_{} + {i};",
                    rank - 1,
                    rank - 1,
                )
                .unwrap();
                first_tensor_input(&tensors).check_bounds(
                    kernel,
                    (0..rank).rev().map(|i| format!("merged_index_{i}")),
                    |kernel| {
                        let mut values = Vec::new();
                        for (index, tensor) in tensors.iter().enumerate() {
                            match tensor {
                                MaybeQTensorInput::Tensor(tensor) => {
                                    writeln!(kernel, "let index_{index} = ",).unwrap();
                                    tensor.strided_index(
                                        kernel,
                                        (0..).map(|i| format!("merged_index_{i}")),
                                    );
                                    writeln!(kernel, ";").unwrap();
                                    values.push(format!("{tensor}[index_{index}]"));
                                }
                                MaybeQTensorInput::QTensor(_) => {
                                    values.push(data.clone());
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
            },
        );

        for _ in 0..rank - 1 {
            writeln!(kernel, "}}").unwrap();
        }
    } else {
        let subgroup_size = kernel.subgroup_size();
        let subgroup_local_id = kernel.subgroup_local_index();
        for (i, index) in global_indexes.iter().enumerate() {
            if i == 0 {
                writeln!(
                    kernel,
                    "let tile_index_{i} = {subgroup_local_id} + (({index} / {subgroup_size}) * {subgroup_size}) * {scaled_tile_size};"
                )
                .unwrap();
            } else {
                writeln!(kernel, "let tile_index_{i} = {index} * {scaled_tile_size};").unwrap();
            }
        }
        writeln!(kernel, "\n").unwrap();

        for (i, shape) in shape.iter().enumerate() {
            if *shape > 1 {
                let local_tile_size = scaled_tile_size.min(*shape as _);
                writeln!(kernel, "for (var local_index_{i} = 0u; local_index_{i} < {local_tile_size}; local_index_{i}++) {{").unwrap();
                if i == 0 {
                    writeln!(
                        kernel,
                        "let merged_index_{i} = tile_index_{i} + local_index_{i} * {subgroup_size};"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        kernel,
                        "let merged_index_{i} = tile_index_{i} + local_index_{i};"
                    )
                    .unwrap();
                }
            } else {
                writeln!(kernel, "let merged_index_{i} = tile_index_{i};").unwrap();
            }
        }

        first_tensor_input(&tensors).check_bounds(
            kernel,
            (0..).map(|i| format!("merged_index_{i}")),
            |kernel| {
                for (index, tensor) in tensors.iter().enumerate() {
                    writeln!(kernel, "let index_{index} = ",).unwrap();
                    match tensor {
                        MaybeQTensorInput::Tensor(tensor) => {
                            tensor.strided_index(kernel, (0..).map(|i| format!("merged_index_{i}")))
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

        for shape in shape {
            if *shape > 1 {
                writeln!(kernel, "}}").unwrap();
            }
        }
    }
}

fn scaled_tile_size(shape: &[usize], tile_size: u32) -> u32 {
    let effective_dims = shape.iter().filter(|&&x| x > 1).count().max(1);
    // scaled_tile_size ^ effective_dims ~ tile_size
    ((tile_size as f64).powf(1.0 / effective_dims as f64).round() as u32).max(1)
}

pub(crate) fn titled_map_workgroup_size_constraints(
    shape: &[usize],
    device: &crate::Device,
) -> WorkgroupShapeConstraints {
    let mut constraints = WorkgroupShapeConstraints::new();
    let effective_rank = shape.iter().filter(|&&x| x > 1).count() as u32;
    for i in (0..effective_rank as usize).take(3) {
        if i == 0 {
            constraints.add_constraint(
                i,
                crate::mir::workgroup_shape::Constraint::more_than_or_equals(
                    device.limits().min_subgroup_size,
                ),
            );
        }
        constraints.add_constraint(i, crate::mir::workgroup_shape::Constraint::LessThan(256));
    }
    for i in effective_rank as usize..3 {
        constraints.add_constraint(i, crate::mir::workgroup_shape::Constraint::Equals(1));
    }
    constraints
}

pub(crate) fn titled_map_dispatch_size<'a>(
    tile_size: u32,
    workgroup_shape: WorkgroupShape,
    tensors: impl IntoIterator<Item = &'a MaybeQData>,
) -> [u32; 3] {
    let mut tensors = tensors.into_iter();
    let layout = tensors.next().unwrap().layout();
    let shape = layout.shape();
    let scaled_tile_size = scaled_tile_size(shape, tile_size);
    let workgroup_size_x = shape
        .first()
        .map(|x| (*x as u32).div_ceil(scaled_tile_size * workgroup_shape.x()))
        .unwrap_or(1);
    let workgroup_size_y = shape
        .get(1)
        .map(|x| (*x as u32).div_ceil(scaled_tile_size * workgroup_shape.y()))
        .unwrap_or(1);
    let workgroup_size_z = shape
        .iter()
        .skip(2)
        .map(|x| (*x as u32).div_ceil(scaled_tile_size))
        .product::<u32>()
        .div_ceil(workgroup_shape.z())
        .max(1);
    [workgroup_size_x, workgroup_size_y, workgroup_size_z]
}
