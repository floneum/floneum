use crate::{DataType, Max, Sum, Tensor};

impl<const R: usize, const R2: usize, D: DataType> Tensor<R, D>
where
    Tensor<R, D>: Max<Output = Tensor<R2, D>>,
    Tensor<R, D>: Sum<Output = Tensor<R2, D>>,
{
    pub fn softmax(&self, dim: usize) -> Self {
        let size = *self.shape();
        let max = self.max(dim);
        let normalized = self - &max.broadcast(size);
        let exp = normalized.exp();
        let sum = exp.sum(dim);
        exp / sum.broadcast(size)
    }

    pub fn softmax_last_dim(&self) -> Self {
        self.softmax(self.rank() - 1)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [1f32, -2., -3., 4., 5., -6.];
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 6] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 6] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 6] = std::array::from_fn(|i| exp[i] / sum);

    println!("{:?}", softmax_array);

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax(0);
    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}
