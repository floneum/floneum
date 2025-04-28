use std::fmt::{Display, Write};

use fusor_gguf::GgmlType;
use wgpu::CommandEncoder;

use crate::QueryItem;
use crate::{
    DataTypeEnum, Layout, QMatrix, TensorData, dequantize_block,
    mir::inputs::{KernelInputValue, QMatrixInput, TensorInput},
    mir::kernel::GenericKernel,
};

#[derive(Clone)]
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

    pub(crate) fn dequantized_datatype(&self) -> DataTypeEnum {
        match self {
            MaybeQData::Tensor(tensor) => tensor.datatype(),
            MaybeQData::QMatrix(_) => DataTypeEnum::F32,
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

impl From<MaybeQData> for KernelInputValue {
    fn from(val: MaybeQData) -> Self {
        match val {
            MaybeQData::Tensor(tensor) => KernelInputValue::Tensor(tensor.clone()),
            MaybeQData::QMatrix(qmatrix) => KernelInputValue::QMatrix(qmatrix.clone()),
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

pub(crate) struct VisitTiledKernel {
    rank: u32,
    contiguous: bool,
    tile_size: u32,
    kernel: GenericKernel,
}

impl VisitTiledKernel {
    pub(crate) fn new(
        rank: u32,
        tile_size: u32,
        contiguous: bool,
        datatypes: Vec<VisitTiledInputType>,
        modify_data: impl FnMut(
            &mut GenericKernel,
            &[String],
            &[MaybeQTensorInput],
            &[String],
        ) -> String,
    ) -> Self {
        let mut kernel = GenericKernel::new();
        let kernel_text = Self::build_tiled_map_kernel(
            rank,
            tile_size,
            contiguous,
            datatypes,
            &mut kernel,
            modify_data,
        );
        kernel.set_body(kernel_text);
        let blocksize = Self::blocksize_raw(contiguous, rank);
        let workgroup_size = if contiguous {
            [blocksize, 1, 1]
        } else {
            std::array::from_fn(|i| if rank as usize > i { blocksize } else { 1 })
        };
        kernel.set_workgroup_size(workgroup_size);
        Self {
            rank,
            contiguous,
            kernel,
            tile_size,
        }
    }

    fn blocksize_raw(contiguous: bool, rank: u32) -> u32 {
        if contiguous {
            256
        } else {
            // max_blocksize^R = 256
            (256f64.powf(1. / rank as f64)).floor() as u32
        }
    }

    fn blocksize(&self) -> u32 {
        Self::blocksize_raw(self.contiguous, self.rank)
    }

    fn build_tiled_map_kernel(
        rank: u32,
        tile_size: u32,
        contiguous: bool,
        datatypes: Vec<VisitTiledInputType>,
        kernel: &mut GenericKernel,
        mut modify_data: impl FnMut(
            &mut GenericKernel,
            &[String],
            &[MaybeQTensorInput],
            &[String],
        ) -> String,
    ) -> String {
        let mut kernel_body = String::new();
        let global_id = kernel.global_id();
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
        if rank > 2 {
            writeln!(&mut kernel_body, "var remaining_z = {global_id}.z;").unwrap();
        }
        for index in (0..rank).skip(2) {
            let size = first.shape_binding(index);
            writeln!(&mut kernel_body, "let z_{index} = remaining_z % {size};").unwrap();
            writeln!(&mut kernel_body, "remaining_z = remaining_z / {size};").unwrap();
            global_indexes.push(format!("z_{index}"));
        }

        if let Some((quantized_type, quantized_input)) = quantized_block {
            for i in 0..rank as usize {
                let index = &global_indexes[i];
                let chunk_size = if i == rank as usize - 1 {
                    quantized_type.block_size() as u32
                } else {
                    tile_size
                };
                writeln!(
                    &mut kernel_body,
                    "let tile_index_{i} = {index} * {chunk_size};"
                )
                .unwrap();
            }
            writeln!(&mut kernel_body, "\n").unwrap();

            for i in 0..rank - 1 {
                writeln!(&mut kernel_body, "for (var local_index_{i} = 0u; local_index_{i} < {tile_size}; local_index_{i}++) {{").unwrap();
                writeln!(
                    &mut kernel_body,
                    "let merged_index_{i} = tile_index_{i} + local_index_{i};"
                )
                .unwrap();
            }

            write!(&mut kernel_body, "let chunk_index = ").unwrap();
            quantized_input.strided_index(
                &mut kernel_body,
                (0..rank - 1)
                    .map(|i| format!("merged_index_{i}"))
                    .chain(std::iter::once(format!("tile_index_{}", rank - 1))),
            );
            writeln!(&mut kernel_body, ";").unwrap();
            writeln!(
                &mut kernel_body,
                "let chunk = {quantized_input}[chunk_index];"
            )
            .unwrap();

            dequantize_block(
                &mut kernel_body,
                *quantized_type,
                "chunk".to_string(),
                DataTypeEnum::F32,
                |i, data, kernel_body| {
                    writeln!(
                        kernel_body,
                        "let merged_index_{} = tile_index_{} + {i};",
                        rank - 1,
                        rank - 1,
                    )
                    .unwrap();
                    first_tensor_input(&tensors).check_bounds(
                        kernel_body,
                        (0..rank).rev().map(|i| format!("merged_index_{i}")),
                        |kernel_body| {
                            let mut values = Vec::new();
                            for (index, tensor) in tensors.iter().enumerate() {
                                match tensor {
                                    MaybeQTensorInput::Tensor(tensor) => {
                                        writeln!(kernel_body, "let index_{index} = ",).unwrap();
                                        tensor.strided_index(
                                            kernel_body,
                                            (0..).map(|i| format!("merged_index_{i}")),
                                        );
                                        writeln!(kernel_body, ";").unwrap();
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
                            writeln!(kernel_body, "{modify_data}").unwrap();
                        },
                    );
                },
            );

            for _ in 0..rank - 1 {
                writeln!(&mut kernel_body, "}}").unwrap();
            }
        } else if contiguous {
            for local_index in 0..tile_size {
                let index = format!("index_{local_index}");
                writeln!(
                    &mut kernel_body,
                    "let {index} = {global_id}.x * {tile_size} + {local_index};"
                )
                .unwrap();
                first_tensor_input(&tensors).check_bounds_contiguous(
                    &mut kernel_body,
                    index.clone(),
                    |kernel_body| {
                        let indexes = (0..datatypes.len())
                            .map(|_| index.clone())
                            .collect::<Vec<_>>();
                        let values = tensors
                            .iter()
                            .map(|tensor| format!("{tensor}[{index}]"))
                            .collect::<Vec<_>>();
                        let modify_data = modify_data(kernel, &indexes, &tensors, &values);
                        writeln!(kernel_body, "{modify_data}").unwrap();
                    },
                );
            }
        } else {
            for i in 0..rank as usize {
                let index = &global_indexes[i];
                writeln!(
                    &mut kernel_body,
                    "let tile_index_{i} = {index} * {tile_size};"
                )
                .unwrap();
            }
            writeln!(&mut kernel_body, "\n").unwrap();

            for i in 0..rank {
                writeln!(&mut kernel_body, "for (var local_index_{i} = 0u; local_index_{i} < {tile_size}; local_index_{i}++) {{").unwrap();
                writeln!(
                    &mut kernel_body,
                    "let merged_index_{i} = tile_index_{i} + local_index_{i};"
                )
                .unwrap();
            }

            first_tensor_input(&tensors).check_bounds(
                &mut kernel_body,
                (0..).map(|i| format!("merged_index_{i}")),
                |kernel_body| {
                    for (index, tensor) in tensors.iter().enumerate() {
                        writeln!(kernel_body, "let index_{index} = ",).unwrap();
                        match tensor {
                            MaybeQTensorInput::Tensor(tensor) => tensor.strided_index(
                                kernel_body,
                                (0..).map(|i| format!("merged_index_{i}")),
                            ),
                            MaybeQTensorInput::QTensor(_) => unreachable!(),
                        }

                        writeln!(kernel_body, ";").unwrap();
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
                    writeln!(kernel_body, "{modify_data}").unwrap();
                },
            );

            for _ in 0..rank {
                writeln!(&mut kernel_body, "}}").unwrap();
            }
        }

        kernel_body
    }

    pub(crate) fn run_with_query<'a>(
        &self,
        tensors: impl IntoIterator<Item = MaybeQData>,
        query: Option<&QueryItem>,
        command_encoder: &mut CommandEncoder,
    ) {
        let tensors = tensors.into_iter().collect::<Vec<_>>();
        let layout = tensors[0].layout();
        let shape = layout.shape();
        let max_blocksize = self.blocksize();
        let workgroup_dispatch_size = if self.contiguous {
            [
                shape
                    .iter()
                    .map(|x| *x as u32)
                    .product::<u32>()
                    .div_ceil(self.tile_size * max_blocksize),
                1,
                1,
            ]
        } else {
            let workgroup_size_x = shape
                .first()
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_y = shape
                .get(1)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_z = shape
                .get(2)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            [workgroup_size_x, workgroup_size_y, workgroup_size_z]
        };

        let device = tensors[0].device().clone();
        self.kernel.run_with_query(
            &device,
            tensors,
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}
