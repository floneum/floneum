// Make sure in-place opts don't change immutable tensors

use fusor_core::Device;
use fusor_core::Sum;
use fusor_core::Tensor;

#[tokio::test]
async fn test_fused_cached_results() {
    let device = Device::new().await.unwrap();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let doubled = tensor.clone() * 2.0;
    // Make sure this result gets cached instead of the doubled result
    let tensor_plus_one = (doubled.clone() + 1.0).sum(0);
    let tensor_plus_one_times_two = tensor_plus_one.clone() * 2.0;
    let tensor_plus_one_times_three = tensor_plus_one.clone() * 3.0;
    let fused_tensor_plus_one_times_two_result =
        tensor_plus_one_times_two.as_slice().await.unwrap();
    let fused_tensor_plus_one_times_three_result =
        tensor_plus_one_times_three.as_slice().await.unwrap();
    println!("{fused_tensor_plus_one_times_two_result:?}");
    println!("{fused_tensor_plus_one_times_three_result:?}");
    println!();
    println!();
    println!();

    let doubled = tensor * 2.0;
    println!("doubled: {:?}", doubled.as_slice().await.unwrap());
    let tensor_plus_one = (doubled.clone() + 1.0).sum(0);
    println!(
        "tensor_plus_one: {:?}",
        tensor_plus_one.as_slice().await.unwrap()
    );
    let tensor_plus_one_times_two = tensor_plus_one.clone() * 2.0;
    let tensor_plus_one_times_three = tensor_plus_one.clone() * 3.0;
    let tensor_plus_one_times_two_result = tensor_plus_one_times_two.as_slice().await.unwrap();
    let tensor_plus_one_times_three_result = tensor_plus_one_times_three.as_slice().await.unwrap();
    println!("{tensor_plus_one_times_two_result:?}");
    println!("{tensor_plus_one_times_three_result:?}");

    assert_eq!(
        fused_tensor_plus_one_times_two_result,
        tensor_plus_one_times_two_result
    );
    assert_eq!(
        fused_tensor_plus_one_times_three_result,
        tensor_plus_one_times_three_result
    );
}
