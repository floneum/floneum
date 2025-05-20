use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Self {
        let shape = self.shape();
        assert!(start + length <= shape[axis]);
        assert!(axis < R);
        self.slice(std::array::from_fn(|i| {
            if i == axis {
                start..start + length
            } else {
                0..shape[i]
            }
        }))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_narrow() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let narrowed = tensor.narrow(0, 1, 2);
    let as_slice = narrowed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 3.);
    assert_eq!(as_slice[[0, 1]], 4.);
    assert_eq!(as_slice[[1, 0]], 5.);
    assert_eq!(as_slice[[1, 1]], 6.);
}
