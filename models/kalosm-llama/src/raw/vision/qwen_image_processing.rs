use candle_core::{Device, Tensor};
use image::DynamicImage;

pub(crate) fn process_image(
    image: &DynamicImage,
    patch_size: usize,
    merge_size: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, [u32; 3])> {
    let merge_patch = (patch_size * merge_size) as u32;
    let resized = normalize_image_shape([merge_patch, merge_patch], [56, 56], [1001, 1001], image);

    assert!(resized.height() % merge_patch == 0);
    assert!(resized.width() % merge_patch == 0);
    let rgb = image_to_rgb(&resized, device)?;
    let grid_t = 1;
    let grid_h = resized.height() as usize / merge_patch as usize;
    let grid_w = resized.width() as usize / merge_patch as usize;
    let rgb = rgb
        .reshape(&[
            grid_t,     // time size
            3,          // channels
            grid_h,     // height patches
            merge_size, // height merge size
            patch_size, // height patch size
            grid_w,     // width patches
            merge_size, // width merge size
            patch_size, // width patch size
        ])
        .unwrap();
    // Move the time, height, and width dimensions to the start
    // shape is now [time patches, height patches, width patches, height merges, width merges, channels, height patch size, width patch size]
    let rgb = rgb.permute([0, 2, 5, 3, 6, 1, 4, 7]).unwrap();
    // Reshape to [patch count, patch data]
    let rgb = rgb
        .reshape(&[
            // patch count
            grid_h * grid_w,
            // patch data
            (3 * merge_patch * merge_patch) as usize,
        ])
        .unwrap();
    Ok((rgb, [grid_t as u32, grid_h as u32, grid_w as u32]))
}

fn normalize_image_shape(
    patch_size: [u32; 2],
    min_size: [u32; 2],
    max_size: [u32; 2],
    image: &DynamicImage,
) -> DynamicImage {
    let mut width = image.width();
    let mut height = image.height();

    // Scale up the image while keeping the aspect ratio to at least the minimum size
    if width < min_size[0] as u32 {
        let scale_up_by = min_size[0] as f32 / width as f32;
        width = (width as f32 * scale_up_by).round() as u32;
        height = (height as f32 * scale_up_by).round() as u32;
    }
    if height < min_size[1] as u32 {
        let scale_up_by = min_size[1] as f32 / height as f32;
        width = (width as f32 * scale_up_by).round() as u32;
        height = (height as f32 * scale_up_by).round() as u32;
    }

    // Scale down the image while keeping the aspect ratio to at most the maximum size
    if width > max_size[0] as u32 {
        let scale_down_by = max_size[0] as f32 / width as f32;
        width = (width as f32 * scale_down_by).round() as u32;
        height = (height as f32 * scale_down_by).round() as u32;
    }
    if height > max_size[1] as u32 {
        let scale_down_by = max_size[1] as f32 / height as f32;
        width = (width as f32 * scale_down_by).round() as u32;
        height = (height as f32 * scale_down_by).round() as u32;
    }

    // Round to the nearest multiple of the patch size
    width = (width as f32 / patch_size[0] as f32).round() as u32 * patch_size[0];
    height = (height as f32 / patch_size[1] as f32).round() as u32 * patch_size[1];

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
        let (rgb, [grid_t, grid_h, grid_w]) =
            process_image(&image, patch_size, spacial_merge_size, &Device::Cpu).unwrap();
        println!("RGB shape: {:?}", rgb);
        println!("Grid shape: {:?}", [grid_t, grid_h, grid_w]);
        assert_eq!(
            rgb.dims(),
            [
                (grid_h * grid_w) as usize,
                3 * patch_size * patch_size * spacial_merge_size * spacial_merge_size,
            ]
        );
        assert_eq!([grid_t, grid_h, grid_w], [1, 24, 36]);
    }
}
