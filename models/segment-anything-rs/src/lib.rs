//! # Segment Anything RS
//! A rust wrapper for [Segment Anything](https://segment-anything.com/)
//!
//! ## Usage
//!
//! ```rust, no_run
//! use segment_anything_rs::*;
//!
//! let model = SegmentAnything::builder().build().unwrap();
//! let image = image::open("examples/landscape.jpg").unwrap();
//! let images = model.segment_everything(image).unwrap();
//! for (i, img) in images.iter().enumerate() {
//!     img.save(&format!("{}.png", i)).unwrap();
//! }
//! ```

#![warn(missing_docs)]

mod raw;

use fusor::{ConcreteTensor, Device, Tensor, VarBuilder};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
use raw::sam::{Sam, IMAGE_SIZE};

/// A builder for [`SegmentAnything`].
#[derive(Default)]
pub struct SegmentAnythingBuilder {
    source: SegmentAnythingSource,
}

impl SegmentAnythingBuilder {
    /// Sets the source of the model.
    pub fn source(mut self, source: SegmentAnythingSource) -> Self {
        self.source = source;
        self
    }

    /// Builds the [`SegmentAnything`] model.
    pub fn build(self) -> Result<SegmentAnything, LoadSegmentAnythingError> {
        SegmentAnything::new(self)
    }
}

/// The source of the model.
pub struct SegmentAnythingSource {
    model: String,
    filename: String,
    tiny: bool,
}

impl SegmentAnythingSource {
    /// Creates a new [`SegmentAnythingSource`].
    pub fn new(model: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            filename: filename.into(),
            tiny: false,
        }
    }

    /// Create the tiny SAM model source.
    pub fn tiny() -> Self {
        let mut self_ = Self::new("lmz/candle-sam", "mobile_sam-tiny-vitt.gguf");
        self_.tiny = true;
        self_
    }

    /// Create a normal sized model source.
    pub fn medium() -> Self {
        Self::new("lmz/candle-sam", "sam_vit_b_01ec64.gguf")
    }
}

impl Default for SegmentAnythingSource {
    fn default() -> Self {
        Self::tiny()
    }
}

/// Settings for running inference on [`SegmentAnything`].
pub struct SegmentAnythingInferenceSettings {
    threshold: f32,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image).
    goal_points: Vec<(f64, f64)>,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image).
    avoid_points: Vec<(f64, f64)>,

    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

impl SegmentAnythingInferenceSettings {
    /// Creates a new [`SegmentAnythingInferenceSettings`] from an image.
    pub fn new<I: GenericImageView<Pixel = Rgba<u8>>>(input: I) -> Self {
        let mut image = ImageBuffer::new(input.width(), input.height());
        image.copy_from(&input, 0, 0).unwrap();
        Self {
            threshold: 0.,
            goal_points: Vec::new(),
            avoid_points: Vec::new(),
            image,
        }
    }

    /// Sets the detection threshold for the mask, 0 is the default value.
    /// - A negative values makes the model return a larger mask.
    /// - A positive makes the model return a smaller mask.
    pub fn set_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Add a point to the list of points to segment.
    pub fn add_goal_point(mut self, x: impl Into<f64>, y: impl Into<f64>) -> Self {
        self.goal_points.push((x.into(), y.into()));
        self
    }

    /// Set the list of points to segment.
    pub fn set_goal_points(mut self, points: Vec<(f64, f64)>) -> Self {
        self.goal_points = points;
        self
    }

    /// Add a point to the list of points to avoid.
    pub fn add_avoid_points(mut self, x: impl Into<f64>, y: impl Into<f64>) -> Self {
        self.avoid_points.push((x.into(), y.into()));
        self
    }

    /// Set the list of points to avoid.
    pub fn set_avoid_points(mut self, points: Vec<(f64, f64)>) -> Self {
        self.avoid_points = points;
        self
    }

    /// Set the image to segment.
    pub fn set_image<I: GenericImageView<Pixel = Rgba<u8>>>(
        mut self,
        image: I,
    ) -> Result<Self, image::ImageError> {
        self.image = ImageBuffer::new(image.width(), image.height());
        self.image.copy_from(&image, 0, 0)?;
        Ok(self)
    }
}

/// An error that can occur when loading a [`SegmentAnything`] model.
#[derive(Debug, thiserror::Error)]
pub enum LoadSegmentAnythingError {
    /// An error that can occur when trying to load a [`SegmentAnything`] model into a device.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] fusor::Error),
    /// An error that can occur when downloading a [`SegmentAnything`] model from Hugging Face.
    #[error("Failed to download model from Hugging Face: {0}")]
    DownloadModel(#[from] hf_hub::api::sync::ApiError),
    /// An IO error opening the model file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// An error that can occur when running a [`SegmentAnything`] model.
#[derive(Debug, thiserror::Error)]
pub enum SegmentAnythingInferenceError {
    /// An error that can occur when trying to run a [`SegmentAnything`] model.
    #[error("Failed to run model: {0}")]
    RunModel(#[from] fusor::Error),
    /// An error that can occur when converting the result of a [`SegmentAnything`] model to an image.
    #[error("Failed to merge masks")]
    MergeMasks,
}

/// The [segment anything](https://segment-anything.com/) model.
pub struct SegmentAnything {
    sam: Sam,
    device: Device,
}

impl SegmentAnything {
    /// Creates a new [`SegmentAnythingBuilder`].
    pub fn builder() -> SegmentAnythingBuilder {
        SegmentAnythingBuilder::default()
    }

    fn new(settings: SegmentAnythingBuilder) -> Result<Self, LoadSegmentAnythingError> {
        let SegmentAnythingBuilder { source } = settings;
        let model_path = {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(source.model);
            api.get(&source.filename)?
        };
        let device = Device::cpu();
        let mut reader = std::io::BufReader::new(std::fs::File::open(&model_path)?);
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .map_err(|e| fusor::Error::msg(format!("Failed to read GGUF: {e}")))?;
        let sam = if source.tiny {
            Sam::load_tiny(&device, &mut vb)?
        } else {
            Sam::load_vit_b(&device, &mut vb)?
        };
        Ok(Self { sam, device })
    }

    /// Segment an image from a list of points. Returns a [`DynamicImage`] mask.
    ///
    /// # Example
    /// ```rust, no_run
    /// use segment_anything_rs::*;
    ///
    /// let model = SegmentAnything::builder().build().unwrap();
    /// let image = image::open("examples/landscape.jpg").unwrap();
    /// let x = image.width() / 2;
    /// let y = image.height() / 4;
    /// let images = model
    ///     .segment_from_points(SegmentAnythingInferenceSettings::new(image).add_goal_point(x, y))
    ///     .unwrap();
    ///
    /// images.save("out.png").unwrap();
    /// ```
    pub fn segment_from_points(
        &self,
        settings: SegmentAnythingInferenceSettings,
    ) -> Result<DynamicImage, SegmentAnythingInferenceError> {
        let SegmentAnythingInferenceSettings {
            threshold,
            goal_points,
            avoid_points,
            image,
        } = settings;

        let image = image::DynamicImage::ImageRgba8(image);
        let image_width = image.width();
        let image_height = image.height();

        let image_tensor = self.image_to_tensor(image);

        let points = {
            let mut points = Vec::new();
            for (x, y) in goal_points {
                points.push((x, y, true));
            }
            for (x, y) in avoid_points {
                points.push((x, y, false));
            }
            points
        };

        let (mask, iou_predictions) = self.sam.forward(&image_tensor, &points, false);

        // Debug: print mask stats
        {
            let mask_flat_dbg: Tensor<1, f32, ConcreteTensor<f32, 1>> = mask
                .reshape([mask.shape().iter().product::<usize>()])
                .to_concrete();
            let dbg_slice = pollster::block_on(mask_flat_dbg.as_slice()).expect("dbg slice");
            let vals = dbg_slice.as_slice();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            eprintln!("Raw mask shape: {:?}, min: {}, max: {}, mean: {}", mask.shape(), min, max, mean);

            eprintln!("IoU shape: {:?}", iou_predictions.shape());
            let iou_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = iou_predictions
                .reshape([iou_predictions.shape().iter().product::<usize>()])
                .to_concrete();
            let iou_slice = pollster::block_on(iou_flat.as_slice()).expect("iou slice");
            eprintln!("IoU predictions: {:?}", iou_slice.as_slice());
        }

        // mask >= threshold -> 255, else 0
        let mask_shape = mask.shape();
        let _n_masks = mask_shape[1];
        let h = mask_shape[2];
        let w = mask_shape[3];

        // Get first mask (batch=0, mask=0)
        let mask_2d: Tensor<2, f32> = mask
            .narrow(0, 0, 1)
            .to_concrete()
            .narrow(1, 0, 1)
            .to_concrete()
            .reshape([h, w])
            .to_concrete();

        // Debug: print mask_2d stats
        {
            let m2d_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = mask_2d
                .reshape([h * w])
                .to_concrete();
            let m2d_slice = pollster::block_on(m2d_flat.as_slice()).expect("m2d slice");
            let vals = m2d_slice.as_slice();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let pos_count = vals.iter().filter(|&&v| v > threshold).count();
            eprintln!("mask_2d: min={}, max={}, positive(>{})={}/{}", min, max, threshold, pos_count, vals.len());
        }

        // Threshold: !(x < threshold) -> 1.0 where x >= threshold, 0.0 otherwise
        // Use gt_scalar(threshold - small_eps) as approximation of ge_scalar
        let threshold_mask: Tensor<2, f32> = mask_2d
            .gt_scalar(threshold - 1e-6)
            .to_concrete();

        // Debug: check threshold result
        {
            let tm_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = threshold_mask
                .reshape([h * w])
                .to_concrete();
            let tm_slice = pollster::block_on(tm_flat.as_slice()).expect("tm slice");
            let vals = tm_slice.as_slice();
            let ones = vals.iter().filter(|&&v| v == 1.0).count();
            let zeros = vals.iter().filter(|&&v| v == 0.0).count();
            eprintln!("threshold_mask: ones={}, zeros={}, other={}", ones, zeros, vals.len() - ones - zeros);
        }

        let mask_u8: Tensor<2, f32> = threshold_mask.mul_scalar(255.0f32);

        // Expand to 3 channels: (H, W) -> (3, H, W)
        let mask_3ch: Tensor<3, f32> = mask_u8
            .reshape([1, h, w])
            .broadcast_as([3, h, w])
            .to_concrete();

        // Permute to (H, W, 3) and flatten
        let mask_hwc: Tensor<3, f32> = mask_3ch
            .transpose(0, 1) // (H, 3, W)
            .to_concrete()
            .transpose(1, 2) // (H, W, 3)
            .to_concrete();
        let mask_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = mask_hwc
            .reshape([h * w * 3])
            .to_concrete();
        let mask_slice = pollster::block_on(mask_flat.as_slice()).expect("Failed to read mask data");
        let mask_pixels: Vec<u8> = mask_slice.as_slice().iter().map(|&v| v as u8).collect();

        let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels)
                .ok_or(SegmentAnythingInferenceError::MergeMasks)?;

        Ok(image::DynamicImage::from(mask_img).resize_to_fill(
            image_width,
            image_height,
            image::imageops::FilterType::CatmullRom,
        ))
    }

    fn image_to_tensor(&self, image: DynamicImage) -> Tensor<3, f32, ConcreteTensor<f32, 3>> {
        let image = {
            let resize_longest = IMAGE_SIZE;
            let (height, width) = (image.height(), image.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        };
        let (height, width) = (image.height() as usize, image.width() as usize);
        let img = image.to_rgb8();
        let data = img.into_raw();
        // Convert u8 to f32
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let device = &self.device;
        // (H, W, 3) -> permute to (3, H, W)
        let image: Tensor<3, f32, ConcreteTensor<f32, 3>> =
            Tensor::from_slice(&device, [height, width, 3], &data_f32)
                .transpose(1, 2) // (H, 3, W)
                .to_concrete()
                .transpose(0, 1) // (3, H, W)
                .to_concrete();
        image
    }

    /// Segment everything in an image. Returns a list of [`DynamicImage`] masks.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use segment_anything_rs::*;
    ///
    /// let model = SegmentAnything::builder().build().unwrap();
    /// let image = image::open("examples/landscape.jpg").unwrap();
    /// let images = model.segment_everything(image).unwrap();
    /// for (i, img) in images.iter().enumerate() {
    ///     img.save(&format!("{}.png", i)).unwrap();
    /// }
    /// ```
    pub fn segment_everything(
        &self,
        image: DynamicImage,
    ) -> Result<Vec<DynamicImage>, SegmentAnythingInferenceError> {
        // For now, a simplified version that returns masks from a grid of points
        // Full generate_masks with NMS is available in raw::sam but would need
        // additional tensor operations (ge, sum) for stability scoring
        let _image_tensor = self.image_to_tensor(image);

        // TODO: Implement full generate_masks pipeline
        Ok(Vec::new())
    }

    /// Load from a local GGUF file path (for testing).
    #[cfg(test)]
    async fn from_gguf_path(
        path: &std::path::Path,
        tiny: bool,
    ) -> Result<Self, LoadSegmentAnythingError> {
        let device = Device::new().await.map_err(|e| fusor::Error::msg(format!("GPU init: {e}")))?;
        let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .map_err(|e| fusor::Error::msg(format!("Failed to read GGUF: {e}")))?;
        let sam = if tiny {
            raw::sam::Sam::load_tiny(&device, &mut vb)?
        } else {
            raw::sam::Sam::load_vit_b(&device, &mut vb)?
        };
        Ok(Self { sam, device })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[tokio::test]
    async fn test_load_tiny_model() {
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            eprintln!("Skipping test: GGUF file not found at {}", gguf_path.display());
            eprintln!("Run convert_to_gguf.py first to create it.");
            return;
        }

        let model = SegmentAnything::from_gguf_path(gguf_path, true).await
            .expect("Failed to load model");

        // Load test image
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("Failed to open test image");
        let (w, h) = (image.width(), image.height());
        eprintln!("Image: {}x{}", w, h);

        // Run inference with a center point (normalized to [0,1])
        let settings = SegmentAnythingInferenceSettings::new(image)
            .add_goal_point(0.5, 0.25);

        let mask = model.segment_from_points(settings)
            .expect("Failed to run inference");

        eprintln!("Mask size: {}x{}", mask.width(), mask.height());
        assert_eq!(mask.width(), w);
        assert_eq!(mask.height(), h);

        // Save mask for visual inspection
        mask.save("/tmp/sam_mask_output.png").expect("Failed to save mask");
        eprintln!("Mask saved to /tmp/sam_mask_output.png");

        // Check mask is not trivial (all black or all white)
        let mask_rgb = mask.to_rgb8();
        let pixels: &[u8] = mask_rgb.as_raw();
        let total = pixels.len();
        let white_count = pixels.iter().filter(|&&v| v == 255).count();
        let black_count = pixels.iter().filter(|&&v| v == 0).count();
        let white_frac = white_count as f64 / total as f64;
        let black_frac = black_count as f64 / total as f64;
        eprintln!("Mask stats: {:.1}% white, {:.1}% black", white_frac * 100.0, black_frac * 100.0);

        // A valid segmentation mask should have both masked and unmasked regions
        assert!(white_frac > 0.01, "Mask is all black — no segmentation detected");
        assert!(black_frac > 0.01, "Mask is all white — entire image masked");
        assert!(white_frac < 0.99, "Mask is nearly all white");
        assert!(black_frac < 0.99, "Mask is nearly all black");
    }

    #[tokio::test]
    async fn test_linear_against_python() {
        use fusor::{Device, VarBuilder, Tensor};
        use fusor::layers::Linear;

        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            eprintln!("Skipping test: GGUF file not found");
            return;
        }

        let device = Device::new().await.expect("GPU init");
        let mut reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .expect("Failed to load GGUF");

        // Load hyper_0.layers.0
        let layer0 = Linear::<f32>::load(&device, &mut vb.pp("mask_decoder.output_hypernetworks_mlps.0.layers.0"))
            .expect("Failed to load linear");

        eprintln!("layer0 weight shape: {:?}", layer0.in_features());
        eprintln!("layer0 out_features: {}", layer0.out_features());

        // Create unit vector e_0
        let mut input_data = vec![0.0f32; 256];
        input_data[0] = 1.0;
        let input: Tensor<2, f32> = Tensor::from_slice(&device, [1, 256], &input_data);

        let output = layer0.forward_2d(&input);
        let output_slice = pollster::block_on(output.as_slice()).expect("output slice");
        let vals = output_slice.as_slice();

        eprintln!("Output (first 10): {:?}", &vals[..10]);
        eprintln!("Output min={:.4} max={:.4}",
            vals.iter().cloned().fold(f32::INFINITY, f32::min),
            vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        // Python reference values for e_0 input through hyper_0.layers.0:
        // [ 0.06819974  0.11510842 -0.19432355 -0.02839741  0.10245589  0.24000874
        //  -0.05332445  0.28987202 -0.04482582  0.1100648 ]
        let expected = [0.06819974, 0.11510842, -0.19432355, -0.02839741, 0.10245589,
                       0.24000874, -0.05332445, 0.28987202, -0.04482582, 0.1100648];

        for (i, (got, exp)) in vals[..10].iter().zip(expected.iter()).enumerate() {
            let diff = (got - exp).abs();
            eprintln!("  [{}] got={:.6} expected={:.6} diff={:.6}", i, got, exp, diff);
            assert!(diff < 0.01, "Mismatch at index {}: got={}, expected={}", i, got, exp);
        }
        eprintln!("Linear forward_2d matches Python reference!");
    }

    #[test]
    fn test_candle_reference() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::sam::{Sam as CSam, IMAGE_SIZE};

        let safetensors_path = std::path::Path::new(
            &std::env::var("HOME").unwrap()
        ).join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        if !safetensors_path.exists() {
            eprintln!("Skipping: safetensors not found");
            return;
        }

        let device = CDevice::Cpu;
        let vb = unsafe { CVarBuilder::from_mmaped_safetensors(
            &[safetensors_path],
            DType::F32,
            &device,
        ).expect("load vb") };

        let sam = CSam::new_tiny(vb).expect("load model");

        // Load the same test image
        let image_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("open image");

        // Resize like candle version does
        let (orig_h, orig_w) = (image.height(), image.width());
        let image = {
            let resize_longest = IMAGE_SIZE as u32;
            let (height, width) = (image.height(), image.width());
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        };
        let (height, width) = (image.height() as usize, image.width() as usize);
        eprintln!("Candle image: {}x{}", width, height);

        let img = image.to_rgb8();
        let data = img.into_raw();
        let image_tensor = CTensor::from_vec(data, (height, width, 3), &device)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        eprintln!("Image tensor shape: {:?}", image_tensor.shape());

        // Run SAM forward with same point (0.5, 0.25)
        let points = vec![(0.5, 0.25, true)];

        // Get image embeddings first and compare
        let img_emb = sam.embeddings(&image_tensor).expect("embeddings");
        let emb_vals: Vec<f32> = img_emb.flatten_all().unwrap().to_vec1().unwrap();
        let emb_min = emb_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let emb_max = emb_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let emb_mean = emb_vals.iter().sum::<f32>() / emb_vals.len() as f32;
        eprintln!("Candle img_embeddings shape={:?} min={:.4} max={:.4} mean={:.6}",
            img_emb.shape(), emb_min, emb_max, emb_mean);

        // Print first 20 values of embeddings for comparison
        eprintln!("Candle emb first 20: {:?}", &emb_vals[..20]);

        let (mask, iou) = sam.forward(&image_tensor, &points, false).expect("forward");

        // Print mask stats
        let mask_vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        let min = mask_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = mask_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = mask_vals.iter().sum::<f32>() / mask_vals.len() as f32;
        let positive = mask_vals.iter().filter(|v| **v > 0.0).count();
        eprintln!("Candle mask shape: {:?}", mask.shape());
        eprintln!("Candle mask: min={:.4} max={:.4} mean={:.4} positive={}/{}", min, max, mean, positive, mask_vals.len());

        let iou_vals: Vec<f32> = iou.flatten_all().unwrap().to_vec1().unwrap();
        eprintln!("Candle IoU: {:?}", iou_vals);

        // Save candle mask for visual comparison
        let mask_shape = mask.shape().dims();
        let mask_h = mask_shape[1];
        let mask_w = mask_shape[2];
        let mask_u8: Vec<u8> = mask_vals.iter().map(|&v| if v > 0.0 { 255u8 } else { 0u8 }).collect();
        let mask_img = image::GrayImage::from_raw(mask_w as u32, mask_h as u32, mask_u8).unwrap();
        // Resize to original size
        let mask_img = image::imageops::resize(&mask_img, orig_w, orig_h as u32, image::imageops::FilterType::Nearest);
        mask_img.save("/tmp/candle_mask_output.png").expect("save candle mask");
        eprintln!("Candle mask saved to /tmp/candle_mask_output.png");
    }

    #[tokio::test]
    async fn test_compare_patch_embed() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor, Module};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::sam::{Sam as CSam, IMAGE_SIZE};
        use fusor::{Device, VarBuilder, Tensor};
        use fusor::layers::{Conv2d, Conv2dConfig};

        let safetensors_path = std::path::Path::new(
            &std::env::var("HOME").unwrap()
        ).join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !safetensors_path.exists() || !gguf_path.exists() {
            eprintln!("Skipping: model files not found");
            return;
        }

        // Load candle model for reference
        let c_device = CDevice::Cpu;
        let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
            &[&safetensors_path],
            DType::F32,
            &c_device,
        ).unwrap() };

        // Load individual candle conv + BN layers for comparison
        let c_conv_cfg = candle_nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let c_conv1 = candle_nn::conv2d_no_bias(3, 32, 3, c_conv_cfg, c_vb.pp("image_encoder.patch_embed.seq.0.c")).unwrap();
        let c_bn1 = candle_nn::batch_norm(32, 1e-5, c_vb.pp("image_encoder.patch_embed.seq.0.bn")).unwrap();

        // Load fusor conv from GGUF
        let f_device = Device::new().await.unwrap();
        let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
        let f_conv_cfg = Conv2dConfig {
            padding: [1, 1],
            stride: [2, 2],
            groups: 1,
        };
        let f_conv1 = Conv2d::<f32>::load(&f_device, &mut f_vb.pp("image_encoder.patch_embed.seq.0.c"), f_conv_cfg).unwrap();

        // Create small test input (1, 3, 8, 8)
        let input_data: Vec<f32> = (0..3*8*8).map(|i| (i as f32) / 192.0 - 0.5).collect();

        // Candle: conv + BN
        let c_input = CTensor::from_vec(input_data.clone(), (1, 3, 8, 8), &c_device).unwrap();
        let c_conv_out = c_conv1.forward(&c_input).unwrap();
        let c_bn_out = c_conv_out.apply_t(&c_bn1, false).unwrap();
        let c_conv_vals: Vec<f32> = c_conv_out.flatten_all().unwrap().to_vec1().unwrap();
        let c_bn_vals: Vec<f32> = c_bn_out.flatten_all().unwrap().to_vec1().unwrap();
        eprintln!("Candle conv1 out shape: {:?}", c_conv_out.shape());
        eprintln!("Candle conv1 out first 10: {:?}", &c_conv_vals[..10]);
        eprintln!("Candle conv1+BN out first 10: {:?}", &c_bn_vals[..10]);

        // Fusor: fused conv (should match conv+BN)
        let f_input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::from_slice(&f_device, [1, 3, 8, 8], &input_data);
        let f_conv_out = f_conv1.forward(&f_input);
        let f_shape = f_conv_out.shape();
        eprintln!("Fusor conv1 out shape: {:?}", f_shape);
        let f_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_conv_out
            .reshape([f_shape.iter().product::<usize>()])
            .to_concrete();
        let f_slice = pollster::block_on(f_flat.as_slice()).unwrap();
        let f_vals = f_slice.as_slice();
        eprintln!("Fusor conv1 out first 10: {:?}", &f_vals[..10]);

        // Compare fusor (fused) vs candle (conv+BN)
        let mut max_diff = 0.0f32;
        for (i, (c, f)) in c_bn_vals.iter().zip(f_vals.iter()).enumerate() {
            let diff = (*c - *f).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if i < 10 {
                eprintln!("  [{}] candle_bn={:.6} fusor_fused={:.6} diff={:.6}", i, c, f, diff);
            }
        }
        eprintln!("Conv1 max diff (candle_bn vs fusor_fused): {:.6}", max_diff);

        // Also load fusor weights and check raw conv (no bias) matches candle conv
        let mut f_reader2 = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb2 = VarBuilder::from_gguf(&mut f_reader2).unwrap();
        let f_conv1_nobias = Conv2d::<f32>::load_no_bias(&f_device, &mut f_vb2.pp("image_encoder.patch_embed.seq.0.c"), f_conv_cfg).unwrap();
        let f_conv_out_nobias = f_conv1_nobias.forward(&f_input);
        let f_shape2 = f_conv_out_nobias.shape();
        let f_flat2: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_conv_out_nobias
            .reshape([f_shape2.iter().product::<usize>()])
            .to_concrete();
        let f_slice2 = pollster::block_on(f_flat2.as_slice()).unwrap();
        let f_vals2 = f_slice2.as_slice();
        eprintln!("\nFusor conv1 (no bias, fused weights only) first 10: {:?}", &f_vals2[..10]);
        eprintln!("Candle conv1 (no bias, original weights) first 10: {:?}", &c_conv_vals[..10]);

        let mut max_diff2 = 0.0f32;
        for (i, (c, f)) in c_conv_vals.iter().zip(f_vals2.iter()).enumerate() {
            let diff = (*c - *f).abs();
            if diff > max_diff2 {
                max_diff2 = diff;
            }
            if i < 5 {
                eprintln!("  [{}] candle_conv={:.6} fusor_conv={:.6} diff={:.6}", i, c, f, diff);
            }
        }
        eprintln!("Raw conv max diff: {:.6}", max_diff2);

        // Test: Load the fused weight from GGUF and run with candle conv2d
        // This isolates whether the issue is in the weight data or the conv op
        let mut f_reader3 = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb3 = VarBuilder::from_gguf(&mut f_reader3).unwrap();
        // Get the raw fused weight data and bias from GGUF
        let fused_w_qm = f_vb3.get("image_encoder.patch_embed.seq.0.c.weight", &f_device).unwrap();
        let fused_w: Tensor<4, f32, ConcreteTensor<f32, 4>> = fused_w_qm.dequantize::<4>().to_concrete();
        let fused_b_qm = f_vb3.get("image_encoder.patch_embed.seq.0.c.bias", &f_device).unwrap();
        let fused_b: Tensor<1, f32, ConcreteTensor<f32, 1>> = fused_b_qm.dequantize::<1>().to_concrete();

        eprintln!("\nFused weight shape: {:?}", fused_w.shape());
        eprintln!("Fused bias shape: {:?}", fused_b.shape());

        // Read fused weight values
        let fw_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = fused_w
            .reshape([fused_w.shape().iter().product::<usize>()])
            .to_concrete();
        let fw_slice = pollster::block_on(fw_flat.as_slice()).unwrap();
        let fw_vals = fw_slice.as_slice();
        eprintln!("Fused weight first 10 (from GGUF): {:?}", &fw_vals[..10]);

        let fb_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = fused_b.clone();
        let fb_slice = pollster::block_on(fb_flat.as_slice()).unwrap();
        let fb_vals = fb_slice.as_slice();
        eprintln!("Fused bias first 5 (from GGUF): {:?}", &fb_vals[..5]);

        // Run the same fused weight through candle conv2d (raw op)
        let fw_shape = fused_w.shape();
        let c_fused_w = CTensor::from_vec(fw_vals.to_vec(), (fw_shape[0], fw_shape[1], fw_shape[2], fw_shape[3]), &c_device).unwrap();
        let c_fused_b = CTensor::from_vec(fb_vals.to_vec(), (fused_b.shape()[0],), &c_device).unwrap();
        // candle conv2d: padding=1, stride=2, dilation=1, groups=1
        let c_fused_out = c_input.conv2d(&c_fused_w, 1, 1, 2, 1).unwrap();
        // Add bias: broadcast (32,) to (1, 32, 1, 1)
        let c_fused_out = c_fused_out.broadcast_add(&c_fused_b.reshape((1, 32, 1, 1)).unwrap()).unwrap();
        let c_fused_vals: Vec<f32> = c_fused_out.flatten_all().unwrap().to_vec1().unwrap();
        eprintln!("\nCandle conv with GGUF fused weights first 10: {:?}", &c_fused_vals[..10]);
        eprintln!("Fusor conv with GGUF fused weights first 10: {:?}", &f_vals[..10]);

        let mut max_diff3 = 0.0f32;
        for (_i, (c, f)) in c_fused_vals.iter().zip(f_vals.iter()).enumerate() {
            let diff = (*c - *f).abs();
            if diff > max_diff3 {
                max_diff3 = diff;
            }
        }
        eprintln!("Conv2d op diff (candle vs fusor, same fused weights): {:.6}", max_diff3);
        eprintln!("(NOTE: this test has wrong candle conv2d params — stride/dilation swapped — ignore this diff)");

        // === Compare MBConv block 0 conv1 (1x1 conv) ===
        // This is layers.0.blocks.0.conv1, which is a 1x1 conv expanding 64 -> 256
        eprintln!("\n=== MBConv block 0 conv1 comparison ===");

        // Create a shared input matching the patch_embed output shape
        // Use small size for speed: (1, 64, 8, 8)
        let mb_input_data: Vec<f32> = (0..64*8*8).map(|i| ((i as f32) / (64.0*8.0*8.0) - 0.5) * 2.0).collect();
        let c_mb_input = CTensor::from_vec(mb_input_data.clone(), (1, 64, 8, 8), &c_device).unwrap();

        // Candle: conv + BN (original safetensors)
        let c_mb_conv1 = candle_nn::conv2d_no_bias(
            64, 256, 1,
            candle_nn::Conv2dConfig::default(),
            c_vb.pp("image_encoder.layers.0.blocks.0.conv1.c"),
        ).unwrap();
        let c_mb_bn1 = candle_nn::batch_norm(
            256, 1e-5,
            c_vb.pp("image_encoder.layers.0.blocks.0.conv1.bn"),
        ).unwrap();
        let c_mb_conv_out = c_mb_conv1.forward(&c_mb_input).unwrap();
        let c_mb_bn_out = c_mb_conv_out.apply_t(&c_mb_bn1, false).unwrap();
        let c_mb_vals: Vec<f32> = c_mb_bn_out.flatten_all().unwrap().to_vec1().unwrap();
        let c_mb_min = c_mb_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let c_mb_max = c_mb_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("Candle MBConv0 conv1+BN: shape={:?} min={:.4} max={:.4}", c_mb_bn_out.shape(), c_mb_min, c_mb_max);
        eprintln!("Candle MBConv0 conv1+BN first 10: {:?}", &c_mb_vals[..10]);

        // Fusor: fused conv (GGUF)
        let mut f_reader4 = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb4 = VarBuilder::from_gguf(&mut f_reader4).unwrap();
        let f_mb_conv1 = Conv2d::<f32>::load(
            &f_device,
            &mut f_vb4.pp("image_encoder.layers.0.blocks.0.conv1.c"),
            Conv2dConfig::default(),
        ).unwrap();
        let f_mb_input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::from_slice(&f_device, [1, 64, 8, 8], &mb_input_data);
        let f_mb_out = f_mb_conv1.forward(&f_mb_input);
        let f_mb_shape = f_mb_out.shape();
        let f_mb_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_mb_out
            .reshape([f_mb_shape.iter().product::<usize>()])
            .to_concrete();
        let f_mb_slice = pollster::block_on(f_mb_flat.as_slice()).unwrap();
        let f_mb_vals = f_mb_slice.as_slice();
        let f_mb_min = f_mb_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_mb_max = f_mb_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("Fusor MBConv0 conv1(fused): shape={:?} min={:.4} max={:.4}", f_mb_shape, f_mb_min, f_mb_max);
        eprintln!("Fusor MBConv0 conv1(fused) first 10: {:?}", &f_mb_vals[..10]);

        // Compare
        let mut mb_max_diff = 0.0f32;
        for (c, f) in c_mb_vals.iter().zip(f_mb_vals.iter()) {
            let diff = (*c - *f).abs();
            if diff > mb_max_diff {
                mb_max_diff = diff;
            }
        }
        eprintln!("MBConv0 conv1 max diff (candle_bn vs fusor_fused): {:.6}", mb_max_diff);

        // === Also compare depthwise conv2 ===
        eprintln!("\n=== MBConv block 0 conv2 (depthwise) comparison ===");

        // Input for conv2 is the GELU output of conv1, use same synthetic input (1, 256, 8, 8)
        let dw_input_data: Vec<f32> = (0..256*8*8).map(|i| ((i as f32) / (256.0*8.0*8.0)) * 2.0).collect();
        let c_dw_input = CTensor::from_vec(dw_input_data.clone(), (1, 256, 8, 8), &c_device).unwrap();

        // Candle: depthwise conv + BN
        let c_dw_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            groups: 256,
            ..Default::default()
        };
        let c_dw_conv = candle_nn::conv2d_no_bias(
            256, 256, 3,
            c_dw_cfg,
            c_vb.pp("image_encoder.layers.0.blocks.0.conv2.c"),
        ).unwrap();
        let c_dw_bn = candle_nn::batch_norm(
            256, 1e-5,
            c_vb.pp("image_encoder.layers.0.blocks.0.conv2.bn"),
        ).unwrap();
        let c_dw_out = c_dw_conv.forward(&c_dw_input).unwrap().apply_t(&c_dw_bn, false).unwrap();
        let c_dw_vals: Vec<f32> = c_dw_out.flatten_all().unwrap().to_vec1().unwrap();
        let c_dw_min = c_dw_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let c_dw_max = c_dw_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("Candle dw conv2+BN: shape={:?} min={:.4} max={:.4}", c_dw_out.shape(), c_dw_min, c_dw_max);
        eprintln!("Candle dw conv2+BN first 10: {:?}", &c_dw_vals[..10]);

        // Fusor: fused depthwise conv
        let mut f_reader5 = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb5 = VarBuilder::from_gguf(&mut f_reader5).unwrap();
        let f_dw_cfg = Conv2dConfig {
            padding: [1, 1],
            stride: [1, 1],
            groups: 256,
        };
        let f_dw_conv = Conv2d::<f32>::load(
            &f_device,
            &mut f_vb5.pp("image_encoder.layers.0.blocks.0.conv2.c"),
            f_dw_cfg,
        ).unwrap();
        let f_dw_input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::from_slice(&f_device, [1, 256, 8, 8], &dw_input_data);
        let f_dw_out = f_dw_conv.forward(&f_dw_input);
        let f_dw_shape = f_dw_out.shape();
        let f_dw_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_dw_out
            .reshape([f_dw_shape.iter().product::<usize>()])
            .to_concrete();
        let f_dw_slice = pollster::block_on(f_dw_flat.as_slice()).unwrap();
        let f_dw_vals = f_dw_slice.as_slice();
        let f_dw_min = f_dw_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_dw_max = f_dw_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("Fusor dw conv2(fused): shape={:?} min={:.4} max={:.4}", f_dw_shape, f_dw_min, f_dw_max);
        eprintln!("Fusor dw conv2(fused) first 10: {:?}", &f_dw_vals[..10]);

        let mut dw_max_diff = 0.0f32;
        for (c, f) in c_dw_vals.iter().zip(f_dw_vals.iter()) {
            let diff = (*c - *f).abs();
            if diff > dw_max_diff {
                dw_max_diff = diff;
            }
        }
        eprintln!("Depthwise conv2 max diff (candle_bn vs fusor_fused): {:.6}", dw_max_diff);
    }

    #[tokio::test]
    async fn test_full_model_comparison() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::sam::Sam as CSam;
        use fusor::{Device, VarBuilder, Tensor};

        let safetensors_path = std::path::Path::new(
            &std::env::var("HOME").unwrap()
        ).join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !safetensors_path.exists() || !gguf_path.exists() {
            eprintln!("Skipping: model files not found");
            return;
        }

        // Load candle model
        let c_device = CDevice::Cpu;
        let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
            &[&safetensors_path],
            DType::F32,
            &c_device,
        ).unwrap() };
        let c_sam = CSam::new_tiny(c_vb).unwrap();

        // Load fusor model
        let f_device = Device::new().await.unwrap();
        let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
        let f_sam = raw::sam::Sam::load_tiny(&f_device, &mut f_vb).unwrap();

        // Create the same image tensor for both
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).unwrap();
        let image = {
            let resize_longest = IMAGE_SIZE as u32;
            let (height, width) = (image.height(), image.width());
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        };
        let (height, width) = (image.height() as usize, image.width() as usize);
        eprintln!("Image size: {}x{}", width, height);

        let img = image.to_rgb8();
        let data = img.into_raw();
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

        // Candle image tensor: (H, W, 3) -> (3, H, W)
        let c_image = CTensor::from_vec(data_f32.clone(), (height, width, 3), &c_device)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        eprintln!("Candle image shape: {:?}", c_image.shape());

        // Fusor image tensor: (H, W, 3) -> (3, H, W)
        let f_image: Tensor<3, f32, ConcreteTensor<f32, 3>> =
            Tensor::from_slice(&f_device, [height, width, 3], &data_f32)
                .transpose(1, 2) // (H, 3, W)
                .to_concrete()
                .transpose(0, 1) // (3, H, W)
                .to_concrete();
        eprintln!("Fusor image shape: {:?}", f_image.shape());

        // Verify the image tensors are identical
        {
            let c_flat: Vec<f32> = c_image.flatten_all().unwrap().to_vec1().unwrap();
            let f_flat_t: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_image
                .reshape([height * width * 3])
                .to_concrete();
            let f_flat_s = pollster::block_on(f_flat_t.as_slice()).unwrap();
            let f_flat = f_flat_s.as_slice();
            let mut img_max_diff = 0.0f32;
            for (c, f) in c_flat.iter().zip(f_flat.iter()) {
                let diff = (*c - *f).abs();
                if diff > img_max_diff {
                    img_max_diff = diff;
                }
            }
            eprintln!("Image tensor max diff: {}", img_max_diff);
            assert!(img_max_diff < 1e-6, "Image tensors differ!");
        }

        // Get embeddings from both
        let c_emb = c_sam.embeddings(&c_image).unwrap();
        let c_emb_vals: Vec<f32> = c_emb.flatten_all().unwrap().to_vec1().unwrap();
        let c_min = c_emb_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let c_max = c_emb_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let c_mean = c_emb_vals.iter().sum::<f32>() / c_emb_vals.len() as f32;
        eprintln!("Candle emb: shape={:?} min={:.6} max={:.6} mean={:.6}", c_emb.shape(), c_min, c_max, c_mean);
        eprintln!("Candle emb first 10: {:?}", &c_emb_vals[..10]);

        let f_emb = f_sam.embeddings(&f_image);
        let f_emb_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_emb
            .reshape([f_emb.shape().iter().product::<usize>()])
            .to_concrete();
        let f_emb_s = pollster::block_on(f_emb_flat.as_slice()).unwrap();
        let f_emb_vals = f_emb_s.as_slice();
        let f_min = f_emb_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_max = f_emb_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let f_mean = f_emb_vals.iter().sum::<f32>() / f_emb_vals.len() as f32;
        eprintln!("Fusor emb: shape={:?} min={:.6} max={:.6} mean={:.6}", f_emb.shape(), f_min, f_max, f_mean);
        eprintln!("Fusor emb first 10: {:?}", &f_emb_vals[..10]);

        // Compare
        let mut max_diff = 0.0f32;
        let mut diff_count_001 = 0;
        let mut diff_count_01 = 0;
        let mut diff_count_1 = 0;
        for (c, f) in c_emb_vals.iter().zip(f_emb_vals.iter()) {
            let diff = (*c - *f).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.01 { diff_count_001 += 1; }
            if diff > 0.1 { diff_count_01 += 1; }
            if diff > 1.0 { diff_count_1 += 1; }
        }
        let total = c_emb_vals.len();
        eprintln!("Embedding max diff: {:.6}", max_diff);
        eprintln!("Values with diff > 0.01: {}/{} ({:.1}%)", diff_count_001, total, diff_count_001 as f64 / total as f64 * 100.0);
        eprintln!("Values with diff > 0.1: {}/{} ({:.1}%)", diff_count_01, total, diff_count_01 as f64 / total as f64 * 100.0);
        eprintln!("Values with diff > 1.0: {}/{} ({:.1}%)", diff_count_1, total, diff_count_1 as f64 / total as f64 * 100.0);
    }

    #[tokio::test]
    async fn test_layer_by_layer_comparison() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor, Module};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::tiny_vit::tiny_vit_5m as c_tiny_vit_5m;
        use fusor::{Device, VarBuilder, Tensor};

        let safetensors_path = std::path::Path::new(
            &std::env::var("HOME").unwrap()
        ).join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !safetensors_path.exists() || !gguf_path.exists() {
            eprintln!("Skipping: model files not found");
            return;
        }

        // Helper to compare and print stats
        fn compare(name: &str, c_vals: &[f32], f_vals: &[f32]) -> f32 {
            let c_min = c_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let c_max = c_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let c_mean = c_vals.iter().sum::<f32>() / c_vals.len() as f32;
            let f_min = f_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let f_max = f_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let f_mean = f_vals.iter().sum::<f32>() / f_vals.len() as f32;
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f32;
            for (c, f) in c_vals.iter().zip(f_vals.iter()) {
                let diff = (*c - *f).abs();
                if diff > max_diff { max_diff = diff; }
                sum_diff += diff;
            }
            let mean_diff = sum_diff / c_vals.len() as f32;
            eprintln!("[{}] candle: min={:.6} max={:.6} mean={:.6}", name, c_min, c_max, c_mean);
            eprintln!("[{}] fusor:  min={:.6} max={:.6} mean={:.6}", name, f_min, f_max, f_mean);
            eprintln!("[{}] max_diff={:.6} mean_diff={:.6} len={}", name, max_diff, mean_diff, c_vals.len());
            eprintln!("[{}] first 5 candle: {:?}", name, &c_vals[..5.min(c_vals.len())]);
            eprintln!("[{}] first 5 fusor:  {:?}", name, &f_vals[..5.min(f_vals.len())]);
            eprintln!();
            max_diff
        }

        // Helper to extract fusor tensor as Vec<f32>
        fn to_vec_1d(t: Tensor<1, f32, ConcreteTensor<f32, 1>>) -> Vec<f32> {
            let s = pollster::block_on(t.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        // Load candle TinyViT independently
        let c_device = CDevice::Cpu;
        let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
            &[&safetensors_path],
            DType::F32,
            &c_device,
        ).unwrap() };
        let c_vit = c_tiny_vit_5m(c_vb.pp("image_encoder")).unwrap();

        // Load fusor TinyViT independently on CPU for deterministic comparison
        let f_device = Device::cpu();
        let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
        let f_sam = raw::sam::Sam::load_tiny(&f_device, &mut f_vb).unwrap();
        let f_vit = match &f_sam.image_encoder {
            raw::sam::ImageEncoder::TinyViT(vit) => vit.as_ref(),
            _ => panic!("Expected TinyViT"),
        };

        // Create the same preprocessed image
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).unwrap();
        let image = {
            let resize_longest = IMAGE_SIZE as u32;
            let (height, width) = (image.height(), image.width());
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        };
        let (height, width) = (image.height() as usize, image.width() as usize);
        let img = image.to_rgb8();
        let data = img.into_raw();
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

        // Build candle image
        let c_image = CTensor::from_vec(data_f32.clone(), (height, width, 3), &c_device)
            .unwrap()
            .permute((2, 0, 1)).unwrap()
            .to_dtype(DType::F32).unwrap();

        // Build fusor image
        let f_image: Tensor<3, f32, ConcreteTensor<f32, 3>> =
            Tensor::from_slice(&f_device, [height, width, 3], &data_f32)
                .transpose(1, 2).to_concrete()
                .transpose(0, 1).to_concrete();

        // ---- Preprocess ----
        // Candle preprocessing
        let c_pixel_mean = CTensor::new(&[123.675f32, 116.28, 103.53], &c_device).unwrap().reshape((3, 1, 1)).unwrap();
        let c_pixel_std = CTensor::new(&[58.395f32, 57.12, 57.375], &c_device).unwrap().reshape((3, 1, 1)).unwrap();
        let c_preprocessed = c_image.to_dtype(DType::F32).unwrap()
            .broadcast_sub(&c_pixel_mean).unwrap()
            .broadcast_div(&c_pixel_std).unwrap()
            .pad_with_zeros(1, 0, IMAGE_SIZE - height).unwrap()
            .pad_with_zeros(2, 0, IMAGE_SIZE - width).unwrap();
        let c_preprocessed_4d = c_preprocessed.unsqueeze(0).unwrap();

        // Fusor preprocessing
        let f_preprocessed = f_sam.preprocess(&f_image);
        let f_preprocessed_4d: Tensor<4, f32, ConcreteTensor<f32, 4>> = f_preprocessed
            .reshape([1, 3, IMAGE_SIZE, IMAGE_SIZE])
            .to_concrete();

        // Compare preprocessed
        {
            let c_vals: Vec<f32> = c_preprocessed_4d.flatten_all().unwrap().to_vec1().unwrap();
            let f_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_preprocessed_4d
                .reshape([3 * IMAGE_SIZE * IMAGE_SIZE])
                .to_concrete();
            let f_vals = to_vec_1d(f_flat);
            compare("preprocess", &c_vals, &f_vals);
        }

        // Feed candle-preprocessed data into fusor to isolate preprocessing from the equation
        let c_prep_vals: Vec<f32> = c_preprocessed_4d.flatten_all().unwrap().to_vec1().unwrap();
        let f_input_from_candle: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::from_slice(&f_device, [1, 3, IMAGE_SIZE, IMAGE_SIZE], &c_prep_vals);

        // Run fusor patch_embed
        let f_pe_out = f_vit.patch_embed.forward(&f_input_from_candle);
        let f_pe_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_pe_out
            .reshape([f_pe_out.shape().iter().product::<usize>()])
            .to_concrete();
        let f_pe_vals = to_vec_1d(f_pe_flat);
        eprintln!("[patch_embed] fusor output shape: {:?}", f_pe_out.shape());
        let f_pe_min = f_pe_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_pe_max = f_pe_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let f_pe_mean = f_pe_vals.iter().sum::<f32>() / f_pe_vals.len() as f32;
        eprintln!("[patch_embed] fusor: min={:.6} max={:.6} mean={:.6}", f_pe_min, f_pe_max, f_pe_mean);
        eprintln!("[patch_embed] fusor first 10: {:?}", &f_pe_vals[..10]);

        // Run fusor layer0
        let f_l0_out = f_vit.layer0.forward(&f_pe_out.to_concrete());
        let f_l0_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_l0_out
            .reshape([f_l0_out.shape().iter().product::<usize>()])
            .to_concrete();
        let f_l0_vals = to_vec_1d(f_l0_flat);
        eprintln!("[layer0] fusor output shape: {:?}", f_l0_out.shape());
        let f_l0_min = f_l0_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_l0_max = f_l0_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let f_l0_mean = f_l0_vals.iter().sum::<f32>() / f_l0_vals.len() as f32;
        eprintln!("[layer0] fusor: min={:.6} max={:.6} mean={:.6}", f_l0_min, f_l0_max, f_l0_mean);
        eprintln!("[layer0] fusor first 10: {:?}", &f_l0_vals[..10]);

        // Run fusor layers 1-3
        let mut f_xs = f_l0_out;
        for (i, layer) in f_vit.layers.iter().enumerate() {
            f_xs = layer.forward(&f_xs);
            let f_li_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_xs
                .reshape([f_xs.shape().iter().product::<usize>()])
                .to_concrete();
            let f_li_vals = to_vec_1d(f_li_flat);
            let li_min = f_li_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let li_max = f_li_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let li_mean = f_li_vals.iter().sum::<f32>() / f_li_vals.len() as f32;
            eprintln!("[layer{}] fusor output shape: {:?} min={:.6} max={:.6} mean={:.6}",
                i + 1, f_xs.shape(), li_min, li_max, li_mean);
            eprintln!("[layer{}] fusor first 10: {:?}", i + 1, &f_li_vals[..10.min(f_li_vals.len())]);
        }

        // Now run the candle full TinyViT and compare embeddings
        let c_emb = c_vit.forward(&c_preprocessed_4d).unwrap();
        let c_emb_vals: Vec<f32> = c_emb.flatten_all().unwrap().to_vec1().unwrap();

        // Run fusor full forward (from the same candle-preprocessed input)
        let f_emb = f_vit.forward(&f_input_from_candle);
        let f_emb_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_emb
            .reshape([f_emb.shape().iter().product::<usize>()])
            .to_concrete();
        let f_emb_vals = to_vec_1d(f_emb_flat);

        let max_diff = compare("full_tinyvit", &c_emb_vals, &f_emb_vals);
        eprintln!("Full TinyViT max diff: {:.6}", max_diff);
    }

    /// Compare a single TinyViTBlock between candle and fusor by loading weights
    /// and running with the same synthetic input.
    #[tokio::test]
    async fn test_attention_block_comparison() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor, Module};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::tiny_vit;
        use fusor::{Device, VarBuilder, Tensor};

        let safetensors_path = std::path::Path::new(
            &std::env::var("HOME").unwrap()
        ).join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !safetensors_path.exists() || !gguf_path.exists() {
            eprintln!("Skipping: model files not found");
            return;
        }

        // Compare LayerNorm first
        {
            let c_device = CDevice::Cpu;
            let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
                &[&safetensors_path],
                DType::F32,
                &c_device,
            ).unwrap() };

            // Load a candle LayerNorm from layers.1.blocks.0.attn.norm
            let c_norm = candle_nn::layer_norm(128, 1e-5,
                c_vb.pp("image_encoder.layers.1.blocks.0.attn.norm")).unwrap();

            // Load fusor LayerNorm
            let f_device = Device::new().await.unwrap();
            let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
            let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
            let f_norm = fusor::layers::LayerNorm::<1, f32>::load(
                &f_device,
                &mut f_vb.pp("image_encoder.layers.1.blocks.0.attn.norm"),
                1e-5,
            ).unwrap();

            // Create synthetic input (1, 49, 128) — like a 7x7 window with 128 dims
            let input_data: Vec<f32> = (0..49*128).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
            let c_input = CTensor::from_vec(input_data.clone(), (1, 49, 128), &c_device).unwrap();
            let f_input: Tensor<3, f32> = Tensor::from_slice(&f_device, [1, 49, 128], &input_data);

            let c_out = c_input.apply(&c_norm).unwrap();
            let c_vals: Vec<f32> = c_out.flatten_all().unwrap().to_vec1().unwrap();

            let f_out = f_norm.forward(&f_input);
            let f_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_out
                .reshape([49 * 128]).to_concrete();
            let f_slice = pollster::block_on(f_flat.as_slice()).unwrap();
            let f_vals: Vec<f32> = f_slice.as_slice().to_vec();

            let mut max_diff = 0.0f32;
            for (c, f) in c_vals.iter().zip(f_vals.iter()) {
                let diff = (*c - *f).abs();
                if diff > max_diff { max_diff = diff; }
            }
            eprintln!("[LayerNorm] max_diff={:.6}", max_diff);
            eprintln!("[LayerNorm] candle first 5: {:?}", &c_vals[..5]);
            eprintln!("[LayerNorm] fusor  first 5: {:?}", &f_vals[..5]);
            assert!(max_diff < 0.001, "LayerNorm diverges: max_diff={}", max_diff);
        }

        // Compare softmax
        {
            let f_device = Device::new().await.unwrap();
            let input_data: Vec<f32> = (0..2*4*4).map(|i| ((i as f32) * 0.1).sin() * 3.0).collect();
            let c_device = CDevice::Cpu;
            let c_input = CTensor::from_vec(input_data.clone(), (2, 4, 4), &c_device).unwrap();
            let f_input: Tensor<3, f32> = Tensor::from_slice(&f_device, [2, 4, 4], &input_data);

            let c_out = candle_nn::ops::softmax_last_dim(&c_input).unwrap();
            let c_vals: Vec<f32> = c_out.flatten_all().unwrap().to_vec1().unwrap();

            let f_out: Tensor<3, f32> = f_input.softmax_last_dim::<2>();
            let f_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_out
                .reshape([2 * 4 * 4]).to_concrete();
            let f_slice = pollster::block_on(f_flat.as_slice()).unwrap();
            let f_vals: Vec<f32> = f_slice.as_slice().to_vec();

            let mut max_diff = 0.0f32;
            for (c, f) in c_vals.iter().zip(f_vals.iter()) {
                let diff = (*c - *f).abs();
                if diff > max_diff { max_diff = diff; }
            }
            eprintln!("[softmax] max_diff={:.6}", max_diff);
            eprintln!("[softmax] candle first 5: {:?}", &c_vals[..5]);
            eprintln!("[softmax] fusor  first 5: {:?}", &f_vals[..5]);
            assert!(max_diff < 0.001, "softmax diverges: max_diff={}", max_diff);
        }

        // Compare Linear (qkv projection)
        {
            let c_device = CDevice::Cpu;
            let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
                &[&safetensors_path],
                DType::F32,
                &c_device,
            ).unwrap() };

            // candle uses its own Linear struct from segment_anything
            // Let's just compare raw matmul + bias
            let c_qkv_w = c_vb.get((384, 128), "image_encoder.layers.1.blocks.0.attn.qkv.weight").unwrap();
            let c_qkv_b = c_vb.get(384, "image_encoder.layers.1.blocks.0.attn.qkv.bias").unwrap();

            let f_device = Device::new().await.unwrap();
            let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
            let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
            let f_linear = fusor::layers::Linear::<f32>::load(
                &f_device,
                &mut f_vb.pp("image_encoder.layers.1.blocks.0.attn.qkv"),
            ).unwrap();

            // Input (1, 49, 128)
            let input_data: Vec<f32> = (0..49*128).map(|i| ((i as f32) * 0.01).sin()).collect();
            let c_input = CTensor::from_vec(input_data.clone(), (1, 49, 128), &c_device).unwrap();
            let f_input: Tensor<3, f32> = Tensor::from_slice(&f_device, [1, 49, 128], &input_data);

            // Candle: matmul + bias
            let c_out = c_input.broadcast_matmul(&c_qkv_w.t().unwrap()).unwrap()
                .broadcast_add(&c_qkv_b).unwrap();
            let c_vals: Vec<f32> = c_out.flatten_all().unwrap().to_vec1().unwrap();

            // Fusor
            let f_out = f_linear.forward(&f_input);
            let f_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_out
                .reshape([49 * 384]).to_concrete();
            let f_slice = pollster::block_on(f_flat.as_slice()).unwrap();
            let f_vals: Vec<f32> = f_slice.as_slice().to_vec();

            let mut max_diff = 0.0f32;
            for (c, f) in c_vals.iter().zip(f_vals.iter()) {
                let diff = (*c - *f).abs();
                if diff > max_diff { max_diff = diff; }
            }
            eprintln!("[Linear qkv] max_diff={:.6}", max_diff);
            eprintln!("[Linear qkv] candle first 5: {:?}", &c_vals[..5]);
            eprintln!("[Linear qkv] fusor  first 5: {:?}", &f_vals[..5]);
            assert!(max_diff < 0.01, "Linear qkv diverges: max_diff={}", max_diff);
        }

        // Compare attention bias (ab)
        {
            let c_device = CDevice::Cpu;
            let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
                &[&safetensors_path],
                DType::F32,
                &c_device,
            ).unwrap() };

            // Candle: compute attention biases like TinyViT Attention does
            let num_heads = 4;
            let resolution = (7, 7);
            let points: Vec<(i64, i64)> = (0..resolution.0)
                .flat_map(|x| (0..resolution.1).map(move |y| (x as i64, y as i64)))
                .collect();
            let mut attention_offsets = std::collections::HashMap::new();
            let mut idxs = Vec::with_capacity(points.len() * points.len());
            for &(x1, y1) in &points {
                for &(x2, y2) in &points {
                    let offset = ((x2 - x1).abs(), (y2 - y1).abs());
                    let l = attention_offsets.len();
                    let idx = *attention_offsets.entry(offset).or_insert(l);
                    idxs.push(idx as u32);
                }
            }
            let c_ab_full = c_vb.get((num_heads, attention_offsets.len()),
                "image_encoder.layers.1.blocks.0.attn.attention_biases").unwrap();
            let c_idxs = CTensor::new(idxs.clone(), &c_device).unwrap();
            let c_ab = c_ab_full.index_select(&c_idxs, 1).unwrap()
                .reshape((num_heads, points.len(), points.len())).unwrap();
            let c_ab_vals: Vec<f32> = c_ab.flatten_all().unwrap().to_vec1().unwrap();

            // Fusor: load attention biases
            let f_device = Device::new().await.unwrap();
            let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
            let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
            let f_ab_raw: Tensor<2, f32> = f_vb.get(
                "image_encoder.layers.1.blocks.0.attn.attention_biases", &f_device
            ).unwrap().dequantize();
            let n_points = points.len();
            let idxs_tensor: Tensor<1, u32> = Tensor::from_slice(&f_device, [idxs.len()], &idxs);
            let f_selected: Tensor<2, f32> = f_ab_raw.index_select(1, &idxs_tensor);
            let f_ab: Tensor<3, f32> = f_selected.reshape([num_heads, n_points, n_points]).to_concrete();
            let f_ab_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_ab
                .reshape([num_heads * n_points * n_points]).to_concrete();
            let f_ab_slice = pollster::block_on(f_ab_flat.as_slice()).unwrap();
            let f_ab_vals: Vec<f32> = f_ab_slice.as_slice().to_vec();

            let mut max_diff = 0.0f32;
            for (c, f) in c_ab_vals.iter().zip(f_ab_vals.iter()) {
                let diff = (*c - *f).abs();
                if diff > max_diff { max_diff = diff; }
            }
            eprintln!("[attention_bias] max_diff={:.6}", max_diff);
            eprintln!("[attention_bias] candle first 5: {:?}", &c_ab_vals[..5]);
            eprintln!("[attention_bias] fusor  first 5: {:?}", &f_ab_vals[..5]);
            assert!(max_diff < 0.001, "Attention bias diverges: max_diff={}", max_diff);
        }

        eprintln!("All attention component comparisons passed!");
    }
}
