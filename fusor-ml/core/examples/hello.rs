use fusor_core::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    let device = Device::new().await.unwrap();

    let tensor = Tensor::new(&device, &vec![vec![[1.; 20]; 10]; 10]);
    let new = tensor.sum(0).sum(0).softmax(0).sum(0);
    let slice = tensor.as_slice().await;
    println!("{:?}", slice);
}
