use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Repeat a tensor along each axis count number of times
    pub fn repeat(&self, count: [usize; R]) -> Self {
        let mut current = self.clone();
        for (i, count) in count.iter().copied().enumerate() {
            if count == 1 {
                continue;
            }
            current = Tensor::cat(vec![current; count], i);
        }
        current
    }
}
