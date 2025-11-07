use std::fmt::Display;

use crate::{
    DataTypeEnum,
    mir::globals::{ArrayType, KernelGlobalType, VectorType},
};

pub(crate) fn maybe_vec_storage_type(size: u32, dtype: DataTypeEnum) -> String {
    match size {
        1 => format!("{dtype}"),
        2..=4 => format!("vec{size}<{dtype}>"),
        _ => format!("array<{dtype}, {size}u>"),
    }
}

pub(crate) fn maybe_vec_storage_type_enum(size: u32, dtype: DataTypeEnum) -> KernelGlobalType {
    match size {
        1 => KernelGlobalType::Value(dtype),
        2..=4 => KernelGlobalType::Vector(VectorType::new(size.to_string(), dtype)),
        _ => KernelGlobalType::Array(ArrayType::new(size.to_string(), dtype)),
    }
}

pub(crate) fn maybe_vec_storage_subgroup_add(size: u32, value: impl Display) -> String {
    match size {
        1..=4 => format!("subgroupAdd({value})"),
        _ => format!(
            "array({})",
            (0..size)
                .map(|i| { format!("subgroupAdd({value}[{i}])") })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub(crate) fn maybe_vec_storage_add(
    size: u32,
    first: impl Display,
    second: impl Display,
) -> String {
    match size {
        1..=4 => format!("{first} + {second}"),
        _ => format!(
            "array({})",
            (0..size)
                .map(|i| { format!("{first}[{i}] + {second}[{i}]") })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub(crate) fn maybe_vec_storage_index(
    size: u32,
    value: impl Display,
    index: impl Display,
) -> String {
    match size {
        0 => unreachable!(),
        1 => format!("{value}"),
        2.. => format!("{value}[{index}]"),
    }
}

pub(crate) fn maybe_vec_dot(size: u32, first: impl Display, second: impl Display) -> String {
    match size {
        0 => unreachable!(),
        1 => format!("{first} * {second}"),
        2.. => format!("dot({first}, {second})"),
    }
}
