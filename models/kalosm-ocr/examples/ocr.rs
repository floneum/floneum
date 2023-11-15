use kalosm_ocr::*;

fn main() {
    let mut model = Ocr::builder().build().unwrap();
    let image = image::open("examples/ocr.png").unwrap();
    let text = model
        .recognize_text(OcrInferenceSettings::new(image).unwrap())
        .unwrap();

    println!("{}", text);
}
