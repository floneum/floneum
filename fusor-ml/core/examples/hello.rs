use std::time::Duration;

use fusor_core::*;

#[tokio::main]
async fn main() {
    let device = Device::new().await.unwrap();

    let tensor = Tensor::new(&device, &vec![vec![[1.; 20]; 10]; 10]);
    let new = tensor.sum(0).sum(0).softmax(0).sum(0);
    let graph = new.graphvis();
    println!("{}", graph);
    let timing = new.all_timing_information().await;
    println!(
        "segment time: {:?}",
        timing.iter().map(|x| x.elapsed()).collect::<Vec<_>>()
    );
    println!("{:?}", timing.iter().map(|x| x.elapsed()).sum::<Duration>());
}
