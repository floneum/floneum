use fusor_gguf::GgmlType;
use std::fmt::Write;
use std::fmt::Display;


#[derive(Clone)]
pub(crate) struct QMatrixInput {
    pub(crate) start_index: u32,
    pub(crate) datatype: GgmlType,
    pub(crate) rank: u32,
}

impl QMatrixInput {
    pub(crate) fn get_matrix_binding(&self) -> u32 {
        self.start_index
    }

    pub(crate) fn get_info_binding(&self) -> u32 {
        self.start_index + 1
    }

    fn info_binding(&self) -> String {
        format!("i_{}", self.get_info_binding())
    }

    pub(crate) fn strided_index(
        &self,
        write: &mut String,
        indexes: impl IntoIterator<Item = String>,
    ) {
        let mut strides = Vec::new();
        let mut product = "1".to_string();
        for i in (0..self.rank).rev() {
            let mut shape = self.shape_binding(i);
            if i == self.rank - 1 {
                write!(&mut shape, " / {}", self.datatype.block_size()).unwrap();
            }
            let new = format!("{} * {}", product, shape);
            strides.push(product);
            product = new;
        }
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = &strides[strides.len() - i - 1];
            write!(write, "({index})*{stride} + ").unwrap();
        }
        for _ in 0..3 {
            write.pop();
        }
    }

    pub(crate) fn shape_binding(&self, rank: u32) -> String {
        format!("{}.shape_{}", self.info_binding(), rank)
    }
}

impl Display for QMatrixInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.start_index)
    }
}
