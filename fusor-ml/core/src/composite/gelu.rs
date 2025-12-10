use crate::{DataType, FloatDataType, Tensor};

impl<const R: usize, D: FloatDataType + DataType> Tensor<R, D> {
    pub fn gelu(&self) -> Self {
        // gelu(x) = tanh(sqrt(2/pi) * x * (1.0 + 0.044715 * x * x))
        // Tanh is numerically unstable for large inputs, so we clamp the input to a reasonable range.
        let myself = self
            .max_elementwise(D::from_f32(-5.5))
            .min_elementwise(D::from_f32(5.5));
        let coeff = D::from_f32((2.0 / std::f32::consts::PI).sqrt());
        let x_squared = &myself * &myself;
        let inner = &myself * &((x_squared * D::from_f32(0.044715)) + D::from_f32(1.0));
        let tanh_inner = inner * coeff;
        let tanh = tanh_inner.tanh();
        let one_plus_tanh = tanh + D::from_f32(1.0);
        let half = D::from_f32(0.5);
        self * &one_plus_tanh * half
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_gelu_is_optimized() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.gelu();

    assert_eq!(tensor.count_kernels_to_resolve(), 1);
}

#[cfg(test)]
#[tokio::test]
async fn test_gelu() {
    use std::f32;

    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.gelu();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let gelu = |x: f32| {
        0.5 * x * (1.0 + ((2.0 / f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    };
    assert!((output[[0, 0]] - gelu(data[0][0])).abs() < 0.001);
    assert!((output[[0, 1]] - gelu(data[0][1])).abs() < 0.001);
    assert!((output[[1, 0]] - gelu(data[1][0])).abs() < 0.001);
    assert!((output[[1, 1]] - gelu(data[1][1])).abs() < 0.001);
    assert!((output[[2, 0]] - gelu(data[2][0])).abs() < 0.001);
    assert!((output[[2, 1]] - gelu(data[2][1])).abs() < 0.001);
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_gelu() {
    use std::f32;

    use crate::Device;

    let device = Device::new().await.unwrap();

    for i in 0..1000 {
        // let random: f32 = rand::random_range(-1e3..1e3);
        let random = i as f32 * 0.1;
        println!("testing {random:?}");

        let gelu = |x: f32| {
            0.5 * x * (1.0 + ((2.0 / f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        };
        let expected = gelu(random);
        println!("expected {expected:?}");
        let data = [random];

        let tensor = Tensor::new(&device, &data);

        let tensor = tensor.gelu();

        let output = tensor.as_slice().await.unwrap();
        println!("{output:?}");
        assert!((output[[0]] - expected).abs() < 0.001);
    }
}
