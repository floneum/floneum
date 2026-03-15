use image::{GenericImageView, RgbaImage};
use segment_anything_rs::*;

/// Generate a distinct color by rotating around the HSV color wheel using the golden angle.
fn golden_color(index: usize) -> [u8; 3] {
    const GOLDEN_ANGLE: f32 = 137.508; // 360 / phi^2
    let hue = (index as f32 * GOLDEN_ANGLE) % 360.0;
    hsv_to_rgb(hue, 0.85, 0.95)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0..60 => (c, x, 0.0),
        60..120 => (x, c, 0.0),
        120..180 => (0.0, c, x),
        180..240 => (0.0, x, c),
        240..300 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder()
        .build()
        .await
        .expect("Failed to load model");

    let image_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
    let image = image::open(&image_path).unwrap();
    let (w, h) = image.dimensions();
    let masks = model.segment_everything(image.clone()).await.unwrap();

    println!("Found {} segments", masks.len());

    let mut output = RgbaImage::new(w, h);

    // Start with the original image
    for (x, y, pixel) in image.pixels() {
        output.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], 255]));
    }

    // Overlay each mask with a distinct color tint
    let tint_alpha = 0.45_f32;
    for (i, mask) in masks.iter().enumerate() {
        let [cr, cg, cb] = golden_color(i);
        let mask = mask.resize_exact(w, h, image::imageops::FilterType::Nearest);
        for (x, y, mask_pixel) in mask.pixels() {
            // mask is an Rgb image where 255 = foreground, 0 = background
            if mask_pixel[0] > 128 {
                let dst = output.get_pixel(x, y);
                let r = ((1.0 - tint_alpha) * dst[0] as f32 + tint_alpha * cr as f32) as u8;
                let g = ((1.0 - tint_alpha) * dst[1] as f32 + tint_alpha * cg as f32) as u8;
                let b = ((1.0 - tint_alpha) * dst[2] as f32 + tint_alpha * cb as f32) as u8;
                output.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
    }

    output.save("segmented.png").unwrap();
    println!("Saved segmented.png");
}
