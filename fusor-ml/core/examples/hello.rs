use fusor_core::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    let device = Device::new().await.unwrap();

    let tensor = Tensor::new(&device, &vec![vec![1.; 10]; 10]);
    let new = tensor.softmax_last_dim();
    println!("{}", new.graphvis());
    let slice = new.as_slice().await;
    println!("{:?}", slice);
}
