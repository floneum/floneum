use candle_core::{Device, Tensor};
use image::DynamicImage;

pub(crate) fn process_image(
    image: &DynamicImage,
    patch_size: usize,
    merge_size: usize,
    min_pixels: Option<u32>,
    max_pixels: Option<u32>,
    device: &Device,
) -> candle_core::Result<(Tensor, [u32; 3])> {
    let merge_patch = (patch_size * merge_size) as u32;
    let resized = normalize_image_shape(
        [merge_patch, merge_patch],
        // min_pixels.unwrap_or(56 * 56),
        // max_pixels.unwrap_or(14 * 14 * 4 * 1280),
        min_pixels.unwrap_or(256 * 28 * 28),
        max_pixels.unwrap_or(512 * 28 * 28),
        image,
    );

    assert!(resized.height() % merge_patch == 0);
    assert!(resized.width() % merge_patch == 0);
    let rgb: Tensor = image_to_rgb(&resized, device)?;
    let grid_t = 1;
    let grid_h = resized.height() as usize / patch_size as usize;
    let grid_w = resized.width() as usize / patch_size as usize;
    let rgb = rgb.reshape(&[
        1,                                         // time size
        1,                                         // temporal patch size
        3,                                         // channels
        (resized.height() / merge_patch) as usize, // height patches
        merge_size,                                // height merge size
        patch_size,                                // height patch size
        (resized.width() / merge_patch) as usize,  // width patches
        merge_size,                                // width merge size
        patch_size,                                // width patch size
    ])?;
    // Repeat along time axis
    let rgb = Tensor::cat(&[&rgb, &rgb], 1)?;
    // Move the time, height, and width dimensions to the start
    let rgb = rgb.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;
    // Reshape to [patch count, patch data]
    let rgb = rgb.reshape(&[
        // patch count
        grid_h * grid_w,
        // patch data
        (3 * patch_size * patch_size * 2) as usize,
    ])?;
    Ok((rgb, [grid_t as u32, grid_h as u32, grid_w as u32]))
}

fn normalize_image_shape(
    patch_size: [u32; 2],
    min_pixels: u32,
    max_pixels: u32,
    image: &DynamicImage,
) -> DynamicImage {
    let mut width = image.width();
    let mut height = image.height();

    // Scale up the image while keeping the aspect ratio to at least the minimum pixels
    if width * height < min_pixels {
        let scale_up_by = ((width * height) as f32 / min_pixels as f32).sqrt();
        width = (width as f32 / scale_up_by / patch_size[0] as f32).floor() as u32 * patch_size[0];
        height =
            (height as f32 / scale_up_by / patch_size[1] as f32).floor() as u32 * patch_size[1];
    } else if width * height > max_pixels {
        // Scale down the image while keeping the aspect ratio to at most the maximum pixels
        let scale_down_by = ((width * height) as f32 / max_pixels as f32).sqrt();
        width =
            (width as f32 / scale_down_by / patch_size[0] as f32).floor() as u32 * patch_size[0];
        height =
            (height as f32 / scale_down_by / patch_size[1] as f32).floor() as u32 * patch_size[1];
    } else {
        // Round to the nearest multiple of the patch size
        width = (width as f32 / patch_size[0] as f32).round() as u32 * patch_size[0];
        height = (height as f32 / patch_size[1] as f32).round() as u32 * patch_size[1];
    }

    // Finally, resize the image to the new dimensions
    image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
}

fn image_to_rgb(image: &DynamicImage, device: &Device) -> candle_core::Result<Tensor> {
    let rgb = image.to_rgb32f();
    let grid = rgb.pixels().flat_map(|p| {
        let [r, g, b] = p.0;
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
    });

    Tensor::from_iter(grid, device)?.reshape(&[1, 3, rgb.height() as usize, rgb.width() as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_image() {
        // download image from https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
        let image_bytes = reqwest::get(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        )
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
        let image = image::load_from_memory(&image_bytes).unwrap();
        let spacial_merge_size = 2;
        let patch_size = 14;
        let resized = normalize_image_shape(
            [patch_size as u32 * spacial_merge_size as u32; 2],
            256 * 28 * 28,
            512 * 28 * 28,
            &image,
        );
        println!(
            "Resized image size: {:?}",
            [resized.height(), resized.width()]
        );
        assert_eq!(resized.height(), 504);
        assert_eq!(resized.width(), 756);
        let (rgb, [grid_t, grid_h, grid_w]) = process_image(
            &image,
            patch_size,
            spacial_merge_size,
            Some(256 * 28 * 28),
            Some(512 * 28 * 28),
            &Device::Cpu,
        )
        .unwrap();
        println!("RGB shape: {:?}", rgb);
        println!("Grid shape: {:?}", [grid_t, grid_h, grid_w]);
        assert_eq!(rgb.dims(), [1944, 1176]);
        assert_eq!([grid_t, grid_h, grid_w], [1, 36, 54]);
    }
}
