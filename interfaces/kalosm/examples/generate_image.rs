use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = Wuerstchen::builder().build().await.unwrap();
    let settings = WuerstchenInferenceSettings::new(
        "a cute cat with a hat in a room covered with fur with incredible detail",
    )
    .with_n_steps(2);
    let images = model.run(settings).unwrap();
    for (i, img) in images.iter().enumerate() {
        img.save(&format!("{}.png", i)).unwrap();
    }
}
