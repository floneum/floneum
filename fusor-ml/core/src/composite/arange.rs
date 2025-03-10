use crate::{DataType, Tensor};

impl<D: DataType> Tensor<1, D> {
    pub fn arange(device: &crate::Device, start: D, end: D) -> Self {
        Self::arange_step(device, start, end, D::one())
    }

    pub fn arange_step(device: &crate::Device, start: D, end: D, step: D) -> Self {
        let mut data = Vec::new();
        let mut current = start;
        if step >= D::zero() {
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            while current > end {
                data.push(current);
                current += step;
            }
        }
        Self::new(device, data.as_slice())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_arange() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = Tensor::arange(&device, 0., 10.);
    let as_slice = data.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice, [0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
}

#[cfg(test)]
#[tokio::test]
async fn test_arange_step() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = Tensor::arange_step(&device, 0., 10., 2.);
    let as_slice = data.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice, [0f32, 2., 4., 6., 8.]);
}
