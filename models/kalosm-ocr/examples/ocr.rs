use kalosm_ocr::*;

#[tokio::main]
async fn main() {
    {
        let mut model = Ocr::builder().build().await.unwrap();
        let image = image::open("examples/written.png").unwrap();
        let text = model
            .recognize_text(OcrInferenceSettings::new(image).unwrap())
            .unwrap();

        println!("{}", text);
    }
    {
        let mut model = Ocr::builder()
            .with_source(OcrSource::base_printed())
            .build()
            .await
            .unwrap();
        let image = image::open("examples/printed.png").unwrap();
        let text = model
            .recognize_text(OcrInferenceSettings::new(image).unwrap())
            .unwrap();

        println!("{}", text);
    }
}
