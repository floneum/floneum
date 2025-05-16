use candle_core::{Device, Tensor};
use image::DynamicImage;

fn process_image(
    image: &DynamicImage,
    patch_size: usize,
    merge_size: usize,
) -> candle_core::Result<(Tensor, [usize; 3])> {
    let resized = normalize_image_shape([28, 28], [56, 56], [1001, 1001], image);

    let rgb = image_to_rgb(&resized, &Device::Cpu)?;
    let grid_t = 0;
    let grid_h = resized.height() as usize / patch_size;
    let grid_w = resized.width() as usize / patch_size;
    let rgb = rgb.reshape(&[
        grid_t,     // time size
        3,          // channels
        grid_h,     // height patches
        merge_size, // height merge size
        patch_size, // height patch size
        grid_w,     // width patches
        merge_size, // width merge size
        patch_size, // width patch size
    ])?;
    // Move the time, height, and width dimensions to the start
    // shape is now [time patches, height patches, width patches, height merges, width merges, channels, height patch size, width patch size]
    let rgb = rgb.permute([0, 2, 5, 3, 6, 1, 4, 7])?;
    // Reshape to [patch count, patch data]
    let rgb = rgb.reshape(&[
        // patch count
        grid_h * grid_w,
        // patch data
        3 * patch_size * patch_size,
    ])?;
    Ok((rgb, [grid_t, grid_h, grid_w]))
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
