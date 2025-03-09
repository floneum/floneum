use crate::{DataType, Sum, Tensor};

impl<D: DataType> Tensor<1, D> {
    pub fn softmax(&self) -> Self {
        let size = *self.shape();
        let exp = self.exp();
        let sum_all = exp.sum(0);
        let sum_all: Tensor<1, D> = sum_all.broadcast(size);
        exp / sum_all
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

    let data = [1f32, -2., -3., 4., 5., -6.];
    let exp: [f32; 6] = std::array::from_fn(|i| data[i].exp());
    let sum = exp.iter().sum::<f32>();
    println!("{:?}", sum);
    let softmax_array: [f32; 6] = std::array::from_fn(|i| exp[i] / sum);

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.softmax();

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    println!("{:?}", softmax_array);
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}
