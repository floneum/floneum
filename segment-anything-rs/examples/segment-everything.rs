use segment_anything_rs::*;

fn main() {
    let model = SegmentAnything::builder().build().unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let images = model.segment_everything(image).unwrap();
    for (i, img) in images.iter().enumerate() {
        img.save(&format!("{}.png", i)).unwrap();
    }
}
