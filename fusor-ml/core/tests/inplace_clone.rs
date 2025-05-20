// Make sure in-place opts don't change immutable tensors

use fusor_ml_core::Device;
use fusor_ml_core::Tensor;

#[tokio::test]
async fn test_add_const() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    for _ in 0..2 {
        // This should not change the original tensor
        let tensor_plus_one = tensor.clone() + 1.0;

        let output = tensor_plus_one.as_slice().await.unwrap();
        println!("{output:?}");
        let result = [
            [[2.0, 3.0], [2.0, 3.0]],
            [[4.0, 5.0], [4.0, 5.0]],
            [[6.0, 7.0], [6.0, 7.0]],
        ];
        let result = Tensor::new(&device, &result);
        assert_eq!(output, result.as_slice().await.unwrap());
    }
}
