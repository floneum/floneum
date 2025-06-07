use std::fmt::Display;
use std::fmt::Write;

use crate::DataTypeEnum;

#[derive(Clone, Debug)]
pub(crate) struct TensorInput {
    pub(crate) tensor_binding: u32,
    pub(crate) info_binding: u32,
    pub(crate) rank: u32,
    pub(crate) mutable: bool,
    pub(crate) datatype: DataTypeEnum,
}

impl TensorInput {
    pub(crate) fn get_tensor_binding(&self) -> u32 {
        self.tensor_binding
    }

    pub(crate) fn get_info_binding(&self) -> u32 {
        self.info_binding
    }

    fn info_binding(&self) -> String {
        format!("i_{}", self.get_info_binding())
    }

    pub(crate) fn offset_binding(&self) -> String {
        format!("{}.offset", self.info_binding())
    }

    pub(crate) fn stride_binding(&self, rank: u32) -> String {
        format!("{}.stride_{}", self.info_binding(), rank)
    }

    pub(crate) fn shape_binding(&self, rank: u32) -> String {
        format!("{}.shape_{}", self.info_binding(), rank)
    }

    pub(crate) fn check_bounds(
        &self,
        write: &mut String,
        indexes: impl IntoIterator<Item = String>,
        in_bounds: impl FnOnce(&mut String),
    ) {
        write!(write, "if true ").unwrap();
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = self.shape_binding(i as u32);
            write!(write, "&& {index} < {stride} ").unwrap();
        }
        write!(write, "{{").unwrap();
        in_bounds(write);
        write!(write, "}}").unwrap();
    }

    pub(crate) fn strided_index(
        &self,
        write: &mut String,
        indexes: impl IntoIterator<Item = String>,
    ) {
        let offset = self.offset_binding();
        write!(write, "{offset} + ").unwrap();
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = self.stride_binding(i as u32);
            write!(write, "({index})*{stride} + ").unwrap();
        }
        for _ in 0..3 {
            write.pop();
        }
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }
}

impl Display for TensorInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.tensor_binding)
    }
}
