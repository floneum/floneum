use kalosm::vision::*;

fn main() {
    let model = SegmentAnything::builder().build().unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let x = image.width() / 2;
    let y = image.height() / 4;
    let images = model
        .segment_from_points(SegmentAnythingInferenceSettings::new(image).add_goal_point(x, y))
        .unwrap();

    images.save("out.png").unwrap();
}
