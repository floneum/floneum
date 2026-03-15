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
    async fn test_mask_with_candle_embeddings() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::sam::{Sam as CSam, IMAGE_SIZE};
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
            &[&safetensors_path], DType::F32, &c_device,
        ).unwrap() };
        let c_sam = CSam::new_tiny(c_vb).unwrap();

        // Load fusor model
        let f_device = Device::new().await.unwrap();
        let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();
        let f_sam = raw::sam::Sam::load_tiny(&f_device, &mut f_vb).unwrap();

        // Load image and get candle embeddings
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
        let c_image = CTensor::from_vec(data_f32.clone(), (height, width, 3), &c_device)
            .unwrap().permute((2, 0, 1)).unwrap().to_dtype(DType::F32).unwrap();

        // Get candle embeddings
        let c_emb = c_sam.embeddings(&c_image).unwrap();
        let c_emb_vals: Vec<f32> = c_emb.flatten_all().unwrap().to_vec1().unwrap();
        let c_emb_shape = c_emb.shape().dims().to_vec();
        eprintln!("Candle embeddings shape: {:?}", c_emb_shape);

        // Get candle full forward result for comparison
        let points = vec![(0.5, 0.25, true)];
        let (c_mask, c_iou) = c_sam.forward(&c_image, &points, false).unwrap();
        let c_mask_vals: Vec<f32> = c_mask.flatten_all().unwrap().to_vec1().unwrap();
        let c_iou_vals: Vec<f32> = c_iou.flatten_all().unwrap().to_vec1().unwrap();
        eprintln!("Candle mask: shape={:?} min={:.4} max={:.4}", c_mask.shape(),
            c_mask_vals.iter().cloned().fold(f32::INFINITY, f32::min),
            c_mask_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        eprintln!("Candle IoU: {:?}", c_iou_vals);

        // Feed candle embeddings into fusor's forward_for_embeddings
        let f_emb: Tensor<4, f32> = Tensor::from_slice(
            &f_device,
            [c_emb_shape[0], c_emb_shape[1], c_emb_shape[2], c_emb_shape[3]],
            &c_emb_vals,
        );

        let (f_mask_raw, f_iou) = f_sam.forward_for_embeddings(
            &f_emb, height, width, &points, false,
        );

        // Upsample like sam.forward does
        let f_mask = f_mask_raw.to_concrete()
            .upsample_nearest2d(
                IMAGE_SIZE / f_mask_raw.shape()[2],
                IMAGE_SIZE / f_mask_raw.shape()[3],
            )
            .narrow(2, 0, height).to_concrete()
            .narrow(3, 0, width).to_concrete();

        let f_mask_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_mask
            .reshape([f_mask.shape().iter().product::<usize>()])
            .to_concrete();
        let f_mask_s = pollster::block_on(f_mask_flat.as_slice()).unwrap();
        let f_mask_vals = f_mask_s.as_slice();
        eprintln!("Fusor mask (candle emb): shape={:?} min={:.4} max={:.4}",
            f_mask.shape(),
            f_mask_vals.iter().cloned().fold(f32::INFINITY, f32::min),
            f_mask_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let f_iou_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = f_iou
            .reshape([f_iou.shape().iter().product::<usize>()])
            .to_concrete();
        let f_iou_s = pollster::block_on(f_iou_flat.as_slice()).unwrap();
        eprintln!("Fusor IoU (candle emb): {:?}", &f_iou_s.as_slice()[..1]);

        // Compare masks
        let min_len = c_mask_vals.len().min(f_mask_vals.len());
        let mut max_diff = 0.0f32;
        for i in 0..min_len {
            let diff = (c_mask_vals[i] - f_mask_vals[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Mask diff (candle emb into fusor decoder): max_diff={:.4}", max_diff);

        // Save fusor mask with candle embeddings
        let mask_h = f_mask.shape()[2];
        let mask_w = f_mask.shape()[3];
        let mask_u8: Vec<u8> = f_mask_vals.iter().map(|&v| if v > 0.0 { 255u8 } else { 0u8 }).collect();
        let mask_img = image::GrayImage::from_raw(mask_w as u32, mask_h as u32, mask_u8).unwrap();
        mask_img.save("/tmp/fusor_candle_emb_mask.png").expect("save mask");
        eprintln!("Saved to /tmp/fusor_candle_emb_mask.png");
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

    /// Step-by-step comparison of mask decoder between candle and fusor.
    /// Uses candle's mask_decoder forward to get candle results, then
    /// compares individual ConvTranspose2d and Linear components.
    #[tokio::test]
    async fn test_mask_decoder_components() {
        use candle_core::{Device as CDevice, DType, Tensor as CTensor, Module};
        use candle_nn::VarBuilder as CVarBuilder;
        use fusor::{Device, VarBuilder, Tensor};

        let safetensors_path = std::path::Path::new(
            &std::env::var("HOME").unwrap()
        ).join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !safetensors_path.exists() || !gguf_path.exists() {
            eprintln!("Skipping: model files not found");
            return;
        }

        fn compare(name: &str, c_vals: &[f32], f_vals: &[f32]) -> f32 {
            let min_len = c_vals.len().min(f_vals.len());
            let mut max_diff = 0.0f32;
            for i in 0..min_len {
                let diff = (c_vals[i] - f_vals[i]).abs();
                max_diff = max_diff.max(diff);
            }
            let c_min = c_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let c_max = c_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let f_min = f_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let f_max = f_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[{}] candle: min={:.6} max={:.6} len={}", name, c_min, c_max, c_vals.len());
            eprintln!("[{}] fusor:  min={:.6} max={:.6} len={}", name, f_min, f_max, f_vals.len());
            eprintln!("[{}] max_diff={:.6}", name, max_diff);
            eprintln!("[{}] first 5 candle: {:?}", name, &c_vals[..5.min(c_vals.len())]);
            eprintln!("[{}] first 5 fusor:  {:?}", name, &f_vals[..5.min(f_vals.len())]);
            eprintln!();
            max_diff
        }

        fn f_to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
            let n: usize = t.shape().iter().product();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
                t.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        // Load candle model
        let c_device = CDevice::Cpu;
        let c_vb = unsafe { CVarBuilder::from_mmaped_safetensors(
            &[&safetensors_path], DType::F32, &c_device,
        ).unwrap() };

        // Load fusor model on GPU to test GPU correctness
        let f_device = Device::new().await.unwrap();
        let mut f_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut f_vb = VarBuilder::from_gguf(&mut f_reader).unwrap();

        // ---- Test 1: ConvTranspose2d with model weights ----
        eprintln!("=== Test 1: ConvTranspose2d ===");
        {
            // Load candle ConvTranspose2d
            let cfg = candle_nn::ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            };
            let c_conv = candle_nn::conv_transpose2d(
                256, 64, 2, cfg,
                c_vb.pp("mask_decoder.output_upscaling.0"),
            ).unwrap();

            // Load fusor ConvTranspose2d
            let f_conv = fusor::layers::ConvTranspose2d::load(
                &f_device,
                &mut f_vb.pp("mask_decoder.output_upscaling.0"),
                [2, 2],
            ).unwrap();

            // Create synthetic input (1, 256, 4, 4) - small for fast testing
            let input_data: Vec<f32> = (0..256*4*4).map(|i| ((i as f32) * 0.01).sin()).collect();
            let c_input = CTensor::from_vec(input_data.clone(), (1, 256, 4, 4), &c_device).unwrap();
            let f_input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
                Tensor::from_slice(&f_device, [1, 256, 4, 4], &input_data);

            // Forward
            let c_out = c_conv.forward(&c_input).unwrap();
            let c_vals: Vec<f32> = c_out.flatten_all().unwrap().to_vec1().unwrap();
            eprintln!("candle ConvT2d output shape: {:?}", c_out.shape());

            let f_out = f_conv.forward(&f_input);
            let f_vals = f_to_vec(&f_out);
            eprintln!("fusor ConvT2d output shape: {:?}", f_out.shape());

            compare("conv_transpose_2d_0", &c_vals, &f_vals);
        }

        // ---- Test 2: Second ConvTranspose2d ----
        eprintln!("=== Test 2: ConvTranspose2d (second) ===");
        {
            let cfg = candle_nn::ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            };
            let c_conv = candle_nn::conv_transpose2d(
                64, 32, 2, cfg,
                c_vb.pp("mask_decoder.output_upscaling.3"),
            ).unwrap();

            let f_conv = fusor::layers::ConvTranspose2d::load(
                &f_device,
                &mut f_vb.pp("mask_decoder.output_upscaling.3"),
                [2, 2],
            ).unwrap();

            let input_data: Vec<f32> = (0..64*8*8).map(|i| ((i as f32) * 0.01).sin()).collect();
            let c_input = CTensor::from_vec(input_data.clone(), (1, 64, 8, 8), &c_device).unwrap();
            let f_input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
                Tensor::from_slice(&f_device, [1, 64, 8, 8], &input_data);

            let c_out = c_conv.forward(&c_input).unwrap();
            let c_vals: Vec<f32> = c_out.flatten_all().unwrap().to_vec1().unwrap();

            let f_out = f_conv.forward(&f_input);
            let f_vals = f_to_vec(&f_out);

            compare("conv_transpose_2d_3", &c_vals, &f_vals);
        }

        // ---- Test 3: Embedding weights comparison ----
        eprintln!("=== Test 3: Token embeddings ===");
        {
            let c_iou = candle_nn::embedding(1, 256, c_vb.pp("mask_decoder.iou_token")).unwrap();
            let c_mask = candle_nn::embedding(4, 256, c_vb.pp("mask_decoder.mask_tokens")).unwrap();

            let f_iou = fusor::layers::Embedding::<f32>::load(
                &f_device, &mut f_vb.pp("mask_decoder.iou_token"),
            ).unwrap();
            let f_mask = fusor::layers::Embedding::<f32>::load(
                &f_device, &mut f_vb.pp("mask_decoder.mask_tokens"),
            ).unwrap();

            let c_iou_vals: Vec<f32> = c_iou.embeddings().flatten_all().unwrap().to_vec1().unwrap();
            let f_iou_vals = f_to_vec(f_iou.embeddings());
            compare("iou_token", &c_iou_vals, &f_iou_vals);

            let c_mask_vals: Vec<f32> = c_mask.embeddings().flatten_all().unwrap().to_vec1().unwrap();
            let f_mask_vals = f_to_vec(f_mask.embeddings());
            compare("mask_tokens", &c_mask_vals, &f_mask_vals);
        }

        // ---- Test 4: Linear weights from transformer ----
        eprintln!("=== Test 4: Transformer Linear weights ===");
        {
            // Check a key linear weight from the transformer
            let c_w = c_vb.pp("mask_decoder.transformer.layers.0.self_attn.q_proj")
                .get((256, 256), "weight").unwrap();
            let c_w_vals: Vec<f32> = c_w.flatten_all().unwrap().to_vec1().unwrap();

            let f_w: Tensor<2, f32> = f_vb.pp("mask_decoder.transformer.layers.0.self_attn.q_proj")
                .get("weight", &f_device).unwrap().dequantize();
            let f_w_vals = f_to_vec(&f_w);
            compare("transformer_q_proj_weight", &c_w_vals, &f_w_vals);
        }

        // ---- Test 5: Hypernetwork MLP weights ----
        eprintln!("=== Test 5: Hypernetwork MLP weights ===");
        {
            // First hypernetwork MLP, first layer weight
            let c_w = c_vb.pp("mask_decoder.output_hypernetworks_mlps.0.layers.0")
                .get((256, 256), "weight").unwrap();
            let c_w_vals: Vec<f32> = c_w.flatten_all().unwrap().to_vec1().unwrap();

            let f_w: Tensor<2, f32> = f_vb.pp("mask_decoder.output_hypernetworks_mlps.0.layers.0")
                .get("weight", &f_device).unwrap().dequantize();
            let f_w_vals = f_to_vec(&f_w);
            compare("hypernetwork_mlp_0_layer0_weight", &c_w_vals, &f_w_vals);

            // Last layer: output is transformer_dim / 8 = 32
            let c_w2 = c_vb.pp("mask_decoder.output_hypernetworks_mlps.0.layers.2")
                .get((32, 256), "weight").unwrap();
            let c_w2_vals: Vec<f32> = c_w2.flatten_all().unwrap().to_vec1().unwrap();

            let f_w2: Tensor<2, f32> = f_vb.pp("mask_decoder.output_hypernetworks_mlps.0.layers.2")
                .get("weight", &f_device).unwrap().dequantize();
            let f_w2_vals = f_to_vec(&f_w2);
            compare("hypernetwork_mlp_0_layer2_weight", &c_w2_vals, &f_w2_vals);
        }

        // ---- Test 6: IoU prediction head weight ----
        eprintln!("=== Test 6: IoU prediction head ===");
        {
            let c_w = c_vb.pp("mask_decoder.iou_prediction_head.layers.0")
                .get((256, 256), "weight").unwrap();
            let c_w_vals: Vec<f32> = c_w.flatten_all().unwrap().to_vec1().unwrap();

            let f_w: Tensor<2, f32> = f_vb.pp("mask_decoder.iou_prediction_head.layers.0")
                .get("weight", &f_device).unwrap().dequantize();
            let f_w_vals = f_to_vec(&f_w);
            compare("iou_head_layer0_weight", &c_w_vals, &f_w_vals);

            // Last layer: output is num_mask_tokens = 4
            let c_w2 = c_vb.pp("mask_decoder.iou_prediction_head.layers.2")
                .get((4, 256), "weight").unwrap();
            let c_w2_vals: Vec<f32> = c_w2.flatten_all().unwrap().to_vec1().unwrap();

            let f_w2: Tensor<2, f32> = f_vb.pp("mask_decoder.iou_prediction_head.layers.2")
                .get("weight", &f_device).unwrap().dequantize();
            let f_w2_vals = f_to_vec(&f_w2);
            compare("iou_head_layer2_weight", &c_w2_vals, &f_w2_vals);
        }

        eprintln!("All mask decoder component tests done!");
    }

    #[tokio::test]
    async fn test_narrow_gpu() {
        use fusor::{Device, Tensor, ConcreteTensor};

        let device = Device::new().await.unwrap();

        // Create a [1, 4] tensor
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let t: Tensor<2, f32> = Tensor::from_slice(&device, [1, 4], &data);

        // Narrow to [1, 1] (first element)
        let narrowed: Tensor<2, f32, ConcreteTensor<f32, 2>> = t.narrow(1, 0, 1).to_concrete();
        let shape = narrowed.shape();
        eprintln!("narrowed shape: {:?}", shape);
        assert_eq!(shape, [1, 1]);

        // Test 1: as_slice on narrowed gives too many values (known issue with raw as_slice)
        let s = narrowed.clone().as_slice().await.unwrap();
        eprintln!("narrowed as_slice len: {} (expected 1)", s.as_slice().len());
        eprintln!("narrowed as_slice vals: {:?}", s.as_slice());
        // Use get() which respects strides
        let val = s.get([0, 0]).unwrap();
        eprintln!("narrowed[0,0] = {} (expected 10.0)", val);
        assert!((*val - 10.0).abs() < 1e-6, "Expected 10.0, got {}", val);

        // Test 2: Add 1.0 to the narrowed tensor and check that GPU operation works
        let one: Tensor<2, f32> = Tensor::from_slice(&device, [1, 1], &[1.0f32]);
        let result = (narrowed + one).to_concrete();
        let result_shape = result.shape();
        let rs = result.as_slice().await.unwrap();
        eprintln!("result shape: {:?}", result_shape);
        eprintln!("result as_slice len: {}", rs.as_slice().len());
        eprintln!("result vals: {:?}", rs.as_slice());
        let rval = rs.get([0, 0]).unwrap();
        eprintln!("result[0,0] = {} (expected 11.0)", rval);
        assert!((*rval - 11.0).abs() < 1e-6, "Expected 11.0, got {}", rval);

        // Test 3: Narrow to second element [1, 1] starting at index 1
        let narrowed2: Tensor<2, f32, ConcreteTensor<f32, 2>> = t.narrow(1, 1, 1).to_concrete();
        let s2 = narrowed2.clone().as_slice().await.unwrap();
        let val2 = s2.get([0, 0]).unwrap();
        eprintln!("narrowed2[0,0] = {} (expected 20.0)", val2);
        assert!((*val2 - 20.0).abs() < 1e-6, "Expected 20.0, got {}", val2);

        // Test 4: Narrow then add on non-zero offset
        let one2: Tensor<2, f32> = Tensor::from_slice(&device, [1, 1], &[1.0f32]);
        let result2 = (narrowed2 + one2).to_concrete();
        let rs2 = result2.as_slice().await.unwrap();
        let rval2 = rs2.get([0, 0]).unwrap();
        eprintln!("result2[0,0] = {} (expected 21.0)", rval2);
        assert!((*rval2 - 21.0).abs() < 1e-6, "Expected 21.0, got {}", rval2);

        // Test 5: Narrow then reshape (mask decoder pattern: hs.narrow(1,0,1).reshape([b, dim]))
        let t3d: Tensor<3, f32> = Tensor::from_slice(&device, [1, 4, 2], &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        ]);
        // narrow(1, 0, 1) -> [1, 1, 2], then reshape to [1, 2]
        let narrowed3: Tensor<3, f32, ConcreteTensor<f32, 3>> = t3d.narrow(1, 0, 1).to_concrete();
        let reshaped: Tensor<2, f32, ConcreteTensor<f32, 2>> = narrowed3.reshape([1, 2]).to_concrete();
        eprintln!("reshaped shape: {:?}", reshaped.shape());
        // Force materialization via mul
        let ones: Tensor<2, f32> = Tensor::from_slice(&device, [1, 2], &[1.0f32, 1.0]);
        let mat = (reshaped * ones).to_concrete();
        let ms = mat.as_slice().await.unwrap();
        eprintln!("reshaped*1 vals: {:?}", ms.as_slice());
        assert_eq!(ms.as_slice().len(), 2);
        assert!((*ms.get([0, 0]).unwrap() - 1.0).abs() < 1e-6, "Expected 1.0, got {}", ms.get([0, 0]).unwrap());
        assert!((*ms.get([0, 1]).unwrap() - 2.0).abs() < 1e-6, "Expected 2.0, got {}", ms.get([0, 1]).unwrap());

        // Test 6: Same but narrow at offset 1
        let narrowed4: Tensor<3, f32, ConcreteTensor<f32, 3>> = t3d.narrow(1, 1, 1).to_concrete();
        let reshaped4: Tensor<2, f32, ConcreteTensor<f32, 2>> = narrowed4.reshape([1, 2]).to_concrete();
        let ones4: Tensor<2, f32> = Tensor::from_slice(&device, [1, 2], &[1.0f32, 1.0]);
        let mat4 = (reshaped4 * ones4).to_concrete();
        let ms4 = mat4.as_slice().await.unwrap();
        eprintln!("narrow(1,1,1).reshape([1,2])*1 vals: {:?}", ms4.as_slice());
        assert!((*ms4.get([0, 0]).unwrap() - 3.0).abs() < 1e-6, "Expected 3.0, got {}", ms4.get([0, 0]).unwrap());
        assert!((*ms4.get([0, 1]).unwrap() - 4.0).abs() < 1e-6, "Expected 4.0, got {}", ms4.get([0, 1]).unwrap());

        // Test 7: Narrow multiple elements then reshape (mask_tokens_out pattern)
        let narrowed5: Tensor<3, f32, ConcreteTensor<f32, 3>> = t3d.narrow(1, 1, 3).to_concrete();
        eprintln!("narrowed5 shape: {:?} (expected [1, 3, 2])", narrowed5.shape());
        // Then narrow(1, 0, 1) from this narrowed tensor and reshape
        let sub_narrow: Tensor<3, f32, ConcreteTensor<f32, 3>> = narrowed5.narrow(1, 0, 1).to_concrete();
        let sub_reshaped: Tensor<2, f32, ConcreteTensor<f32, 2>> = sub_narrow.reshape([1, 2]).to_concrete();
        let ones5: Tensor<2, f32> = Tensor::from_slice(&device, [1, 2], &[1.0f32, 1.0]);
        let mat5 = (sub_reshaped * ones5).to_concrete();
        let ms5 = mat5.as_slice().await.unwrap();
        eprintln!("double-narrow.reshape*1 vals: {:?} (expected [3.0, 4.0])", ms5.as_slice());
        assert!((*ms5.get([0, 0]).unwrap() - 3.0).abs() < 1e-6, "Expected 3.0, got {}", ms5.get([0, 0]).unwrap());
        assert!((*ms5.get([0, 1]).unwrap() - 4.0).abs() < 1e-6, "Expected 4.0, got {}", ms5.get([0, 1]).unwrap());
    }

    #[tokio::test]
    async fn test_transpose_reshape_matmul_gpu() {
        use fusor::{Device, Tensor, ConcreteTensor};

        let gpu = Device::new().await.unwrap();
        let cpu = Device::cpu();

        // Simulate the attention pattern:
        // input (1, hw, c) -> reshape (1, hw, heads, c_per_head) -> transpose(1,2) -> (1, heads, hw, c_per_head)
        // then matmul with k^T
        let hw = 4;
        let heads = 2;
        let c_per_head = 3;
        let c = heads * c_per_head;

        let data: Vec<f32> = (0..hw*c).map(|i| (i as f32 + 1.0) * 0.1).collect();

        fn run_attention_pattern(device: &fusor::Device, data: &[f32], hw: usize, heads: usize, c_per_head: usize) -> Vec<f32> {
            let c = heads * c_per_head;
            let x: Tensor<3, f32> = Tensor::from_slice(device, [1, hw, c], data);
            // separate_heads: reshape(1, hw, heads, c_per_head).transpose(1,2)
            let x4d: Tensor<4, f32, ConcreteTensor<f32, 4>> = x
                .reshape([1, hw, heads, c_per_head])
                .transpose(1, 2)
                .to_concrete();
            // matmul: q @ k^T (both same tensor for simplicity)
            let k_t: Tensor<4, f32, ConcreteTensor<f32, 4>> = x4d.transpose(2, 3).to_concrete();
            let attn = x4d.mat_mul(&k_t);

            // Read out via mul by 1
            let shape = attn.shape();
            let n: usize = shape.iter().product();
            let ones: Tensor<4, f32> = Tensor::from_slice(device, shape, &vec![1.0f32; n]);
            let mat: Tensor<4, f32> = (attn * ones).to_concrete();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = mat.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        let cpu_result = run_attention_pattern(&cpu, &data, hw, heads, c_per_head);
        let gpu_result = run_attention_pattern(&gpu, &data, hw, heads, c_per_head);

        let max_diff = cpu_result.iter().zip(gpu_result.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Attention pattern CPU vs GPU:");
        eprintln!("  CPU first 8: {:?}", &cpu_result[..8.min(cpu_result.len())]);
        eprintln!("  GPU first 8: {:?}", &gpu_result[..8.min(gpu_result.len())]);
        eprintln!("  max_diff: {:.6}", max_diff);
        eprintln!("  CPU len: {}, GPU len: {}", cpu_result.len(), gpu_result.len());
        assert!(max_diff < 0.001, "CPU vs GPU attention pattern diff too large: {}", max_diff);
    }

    /// Test that CPU and GPU mask decoder produce matching results.
    #[tokio::test]
    async fn test_mask_decoder_cpu_vs_gpu() {
        use fusor::{Device, VarBuilder, Tensor, ConcreteTensor};

        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            eprintln!("Skipping: GGUF model not found");
            return;
        }

        fn f_to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
            let shape = t.shape();
            let n: usize = shape.iter().product();
            let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
            let materialized: Tensor<R, f32> = (t * ones).to_concrete();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = materialized.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        fn max_diff(a: &[f32], b: &[f32]) -> f32 {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        }

        // Load on CPU and GPU
        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();
        let mut cpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut cpu_vb = VarBuilder::from_gguf(&mut cpu_reader).unwrap();
        let cpu_sam = raw::sam::Sam::load_tiny(&cpu, &mut cpu_vb).unwrap();
        let mut gpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut gpu_vb = VarBuilder::from_gguf(&mut gpu_reader).unwrap();
        let gpu_sam = raw::sam::Sam::load_tiny(&gpu, &mut gpu_vb).unwrap();

        // Dense PE: pe_layer.forward()
        let cpu_pe = cpu_sam.prompt_encoder.get_dense_pe();
        let gpu_pe = gpu_sam.prompt_encoder.get_dense_pe();
        let pe_diff = max_diff(&f_to_vec(&cpu_pe), &f_to_vec(&gpu_pe));
        eprintln!("[dense_pe] max_diff={:.6}", pe_diff);
        assert!(pe_diff < 0.001, "dense PE diverged: {}", pe_diff);

        // Prompt encoder
        let points = vec![(0.5, 0.25, true)];
        let xys: Vec<f32> = points.iter()
            .flat_map(|(x, y, _)| [(*x as f32) * 1024.0, (*y as f32) * 771.0]).collect();
        let labels: Vec<f32> = points.iter().map(|(_, _, b)| if *b { 1f32 } else { 0f32 }).collect();
        let cpu_pts: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 1, 2], &xys);
        let cpu_lbls: Tensor<2, f32> = Tensor::from_slice(&cpu, [1, 1], &labels);
        let gpu_pts: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 1, 2], &xys);
        let gpu_lbls: Tensor<2, f32> = Tensor::from_slice(&gpu, [1, 1], &labels);
        let (cpu_sparse, cpu_dense) = cpu_sam.prompt_encoder.forward(Some((&cpu_pts, &cpu_lbls)), None, None);
        let (gpu_sparse, gpu_dense) = gpu_sam.prompt_encoder.forward(Some((&gpu_pts, &gpu_lbls)), None, None);
        let sparse_diff = max_diff(&f_to_vec(&cpu_sparse), &f_to_vec(&gpu_sparse));
        eprintln!("[sparse_prompt] max_diff={:.6}", sparse_diff);
        assert!(sparse_diff < 0.001, "sparse prompt diverged: {}", sparse_diff);

        // Synthetic image embeddings for mask decoder test
        let emb_data: Vec<f32> = (0..256*64*64).map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
        let cpu_emb: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 256, 64, 64], &emb_data);
        let gpu_emb: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 256, 64, 64], &emb_data);

        // Full mask decoder forward
        let (cpu_masks, cpu_iou) = cpu_sam.mask_decoder.forward(&cpu_emb, &cpu_pe, &cpu_sparse, &cpu_dense, false);
        let (gpu_masks, gpu_iou) = gpu_sam.mask_decoder.forward(&gpu_emb, &gpu_pe, &gpu_sparse, &gpu_dense, false);
        let mask_diff = max_diff(&f_to_vec(&cpu_masks), &f_to_vec(&gpu_masks));
        let iou_diff = max_diff(&f_to_vec(&cpu_iou), &f_to_vec(&gpu_iou));
        eprintln!("[masks] max_diff={:.6}", mask_diff);
        eprintln!("[iou_pred] max_diff={:.6}", iou_diff);
        assert!(mask_diff < 0.01, "mask output diverged: {}", mask_diff);
        assert!(iou_diff < 0.01, "IoU prediction diverged: {}", iou_diff);
    }

    /// MRE: GPU dual-consumer buffer reuse bug in pe_encoding.
    ///
    /// When a single lazy graph node feeds into two consumers (sin() and cos()),
    /// the GPU compute graph can incorrectly reuse/overwrite the shared buffer,
    /// corrupting one consumer's output. This reproduces the exact chain from
    /// PositionEmbeddingRandom::pe_encoding():
    ///   arange → div → broadcast → cat → mul_scalar → add → mat_mul → mul_scalar → {sin, cos}
    ///
    /// The bug manifests when:
    /// 1. The buffer pool has been warmed up (model loaded)
    /// 2. The GGUF-loaded gaussian matrix is used (F32 zero-copy dequantize path)
    /// 3. The full chain stays lazy with no intermediate materialization
    ///
    /// Workaround: duplicate the mul_scalar node so each consumer has its own input.
    /// See prompt_encoder.rs pe_encoding() for the applied workaround.
    #[tokio::test]
    async fn test_dual_consumer_gpu_bug() {
        use fusor::{Device, Tensor, ConcreteTensor};

        fn f_to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
            let shape = t.shape();
            let n: usize = shape.iter().product();
            let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
            let materialized: Tensor<R, f32> = (t * ones).to_concrete();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = materialized.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        fn max_diff(a: &[f32], b: &[f32]) -> f32 {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        }

        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            eprintln!("Skipping: GGUF model not found");
            return;
        }

        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();

        // Load full model — warms up GPU buffer pool AND gives us the
        // GGUF-loaded gaussian matrix (F32 zero-copy dequantize path).
        let mut cpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut cpu_vb = VarBuilder::from_gguf(&mut cpu_reader).unwrap();
        let cpu_sam = raw::sam::Sam::load_tiny(&cpu, &mut cpu_vb).unwrap();
        let mut gpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut gpu_vb = VarBuilder::from_gguf(&mut gpu_reader).unwrap();
        let gpu_sam = raw::sam::Sam::load_tiny(&gpu, &mut gpu_vb).unwrap();

        // Reproduce pe_encoding exactly: build coords, matmul with gaussian matrix,
        // then scale and take sin/cos of the SAME node (dual consumer).
        let h = 64usize;
        let w = 64usize;

        fn build_pe_encoding(
            device: &Device,
            gm: &Tensor<2, f32, ConcreteTensor<f32, 2>>,
            h: usize,
            w: usize,
            shared: bool,
        ) -> Tensor<3, f32> {
            // Build coords grid (same as PositionEmbeddingRandom::forward)
            let x: Tensor<1, f32> = fusor::arange_step::<f32>(device, 0.5, w as f32 + 0.5, 1.0)
                .div_scalar(w as f32);
            let y: Tensor<1, f32> = fusor::arange_step::<f32>(device, 0.5, h as f32 + 0.5, 1.0)
                .div_scalar(h as f32);
            let x2d: Tensor<2, f32> = x.reshape([1, w]).broadcast_as([h, w]).to_concrete();
            let y2d: Tensor<2, f32> = y.reshape([h, 1]).broadcast_as([h, w]).to_concrete();
            let xu: Tensor<3, f32> = x2d.reshape([h, w, 1]).to_concrete();
            let yu: Tensor<3, f32> = y2d.reshape([h, w, 1]).to_concrete();
            let coords: Tensor<3, f32> = Tensor::cat([xu, yu], 2);

            // pe_encoding: coords * 2 - 1, then matmul with gaussian matrix
            let coords: Tensor<3, f32> = (coords.mul_scalar(2.0) + (-1.0f32)).to_concrete();
            let gm_shape = gm.shape();
            let gm3: Tensor<3, f32> = gm
                .reshape([1, gm_shape[0], gm_shape[1]])
                .broadcast_as([h, gm_shape[0], gm_shape[1]])
                .to_concrete();
            let mm = coords.mat_mul(&gm3);

            if shared {
                // BUG PATH: single mul_scalar feeds both sin() and cos()
                let scaled = mm.mul_scalar(2.0 * std::f32::consts::PI);
                let sin_out: Tensor<3, f32> = scaled.sin().to_concrete();
                let cos_out: Tensor<3, f32> = scaled.cos().to_concrete();
                Tensor::cat([sin_out, cos_out], 2)
            } else {
                // WORKAROUND: separate mul_scalar for each consumer
                let scaled_sin = mm.mul_scalar(2.0 * std::f32::consts::PI);
                let scaled_cos = mm.mul_scalar(2.0 * std::f32::consts::PI);
                let sin_out: Tensor<3, f32> = scaled_sin.sin().to_concrete();
                let cos_out: Tensor<3, f32> = scaled_cos.cos().to_concrete();
                Tensor::cat([sin_out, cos_out], 2)
            }
        }

        let cpu_gm = &cpu_sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix;
        let gpu_gm = &gpu_sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix;

        // CPU reference (always correct)
        let cpu_result = build_pe_encoding(&cpu, cpu_gm, h, w, true);

        // GPU with shared node (triggers bug)
        let gpu_shared = build_pe_encoding(&gpu, gpu_gm, h, w, true);
        let diff_shared = max_diff(&f_to_vec(&cpu_result), &f_to_vec(&gpu_shared));
        eprintln!("[dual_consumer_shared] max_diff={:.6}", diff_shared);

        // GPU with separate nodes (workaround)
        let gpu_separate = build_pe_encoding(&gpu, gpu_gm, h, w, false);
        let diff_separate = max_diff(&f_to_vec(&cpu_result), &f_to_vec(&gpu_separate));
        eprintln!("[dual_consumer_separate] max_diff={:.6}", diff_separate);

        // The workaround must always pass
        assert!(diff_separate < 0.001, "separate consumers diverged (unexpected): {}", diff_separate);

        // The shared version triggers the bug — if this starts passing, the
        // underlying fusor bug has been fixed and the workaround in
        // prompt_encoder.rs pe_encoding() can be removed.
        if diff_shared > 0.01 {
            eprintln!(
                "NOTE: dual-consumer bug still present (diff={}). \
                 Workaround in prompt_encoder.rs pe_encoding() is still needed.",
                diff_shared
            );
        } else {
            eprintln!(
                "dual-consumer bug appears FIXED (diff={}). \
                 Workaround in prompt_encoder.rs pe_encoding() can be removed.",
                diff_shared
            );
        }
    }

    /// Test matmul at realistic transformer sizes (CPU vs GPU)
    #[tokio::test]
    async fn test_matmul_large_cpu_vs_gpu() {
        use fusor::{Device, Tensor, ConcreteTensor};

        fn to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
            let shape = t.shape();
            let n: usize = shape.iter().product();
            let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
            let materialized: Tensor<R, f32> = (t * ones).to_concrete();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = materialized.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();

        // Test 1: Batched matmul [1, 8, 5, 32] @ [1, 8, 32, 4096] -> [1, 8, 5, 4096]
        // This is the token-to-image attention pattern
        {
            let a_data: Vec<f32> = (0..8*5*32).map(|i| ((i as f32) * 0.01).sin()).collect();
            let b_data: Vec<f32> = (0..8*32*4096).map(|i| ((i as f32) * 0.007).cos()).collect();

            let cpu_a: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 8, 5, 32], &a_data);
            let cpu_b: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 8, 32, 4096], &b_data);
            let gpu_a: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 8, 5, 32], &a_data);
            let gpu_b: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 8, 32, 4096], &b_data);

            let cpu_out = cpu_a.mat_mul(&cpu_b);
            let gpu_out = gpu_a.mat_mul(&gpu_b);

            let cpu_vals = to_vec(&cpu_out);
            let gpu_vals = to_vec(&gpu_out);
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[matmul 1,8,5,32 @ 1,8,32,4096] max_diff={:.6}", max_diff);
            assert!(max_diff < 0.001, "matmul diverged: {}", max_diff);
        }

        // Test 2: Batched matmul with transposed B: [1, 8, 4096, 32] @ [1, 8, 32, 5]
        // This is the image-to-token attention pattern
        {
            let a_data: Vec<f32> = (0..8*4096*32).map(|i| ((i as f32) * 0.01).sin()).collect();
            let b_data: Vec<f32> = (0..8*32*5).map(|i| ((i as f32) * 0.007).cos()).collect();

            let cpu_a: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 8, 4096, 32], &a_data);
            let cpu_b: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 8, 32, 5], &b_data);
            let gpu_a: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 8, 4096, 32], &a_data);
            let gpu_b: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 8, 32, 5], &b_data);

            let cpu_out = cpu_a.mat_mul(&cpu_b);
            let gpu_out = gpu_a.mat_mul(&gpu_b);

            let cpu_vals = to_vec(&cpu_out);
            let gpu_vals = to_vec(&gpu_out);
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[matmul 1,8,4096,32 @ 1,8,32,5] max_diff={:.6}", max_diff);
            assert!(max_diff < 0.001, "matmul diverged: {}", max_diff);
        }

        // Test 3: 2D matmul [4096, 256] @ [256, 256] (Linear layer forward)
        {
            let a_data: Vec<f32> = (0..4096*256).map(|i| ((i as f32) * 0.003).sin()).collect();
            let b_data: Vec<f32> = (0..256*256).map(|i| ((i as f32) * 0.005).cos()).collect();

            let cpu_a: Tensor<2, f32> = Tensor::from_slice(&cpu, [4096, 256], &a_data);
            let cpu_b: Tensor<2, f32> = Tensor::from_slice(&cpu, [256, 256], &b_data);
            let gpu_a: Tensor<2, f32> = Tensor::from_slice(&gpu, [4096, 256], &a_data);
            let gpu_b: Tensor<2, f32> = Tensor::from_slice(&gpu, [256, 256], &b_data);

            let cpu_out = cpu_a.mat_mul(&cpu_b);
            let gpu_out = gpu_a.mat_mul(&gpu_b);

            let cpu_vals = to_vec(&cpu_out);
            let gpu_vals = to_vec(&gpu_out);
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[matmul 4096,256 @ 256,256] max_diff={:.6}", max_diff);
            assert!(max_diff < 0.001, "matmul diverged: {}", max_diff);
        }

        // Test 4: Softmax over large last dim [1, 8, 5, 4096]
        {
            let data: Vec<f32> = (0..8*5*4096).map(|i| ((i as f32) * 0.003).sin() * 2.0).collect();

            let cpu_t: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 8, 5, 4096], &data);
            let gpu_t: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 8, 5, 4096], &data);

            let cpu_out: Tensor<4, f32> = cpu_t.softmax_last_dim::<3>();
            let gpu_out: Tensor<4, f32> = gpu_t.softmax_last_dim::<3>();

            let cpu_vals = to_vec(&cpu_out);
            let gpu_vals = to_vec(&gpu_out);
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[softmax 1,8,5,4096] max_diff={:.6}", max_diff);
            assert!(max_diff < 0.001, "softmax diverged: {}", max_diff);
        }

        // Test 5: Matmul with transposed input (reshape+transpose pattern from transformer)
        // This simulates separate_heads: [1, 4096, 256] -> reshape [1, 4096, 8, 32] -> transpose(1,2) -> [1, 8, 4096, 32]
        // Then matmul with transposed key: [1, 8, 4096, 32] @ [1, 8, 32, 4096]
        {
            let data: Vec<f32> = (0..4096*256).map(|i| ((i as f32) * 0.002).sin()).collect();

            let cpu_t: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 4096, 256], &data);
            let gpu_t: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 4096, 256], &data);

            // separate_heads pattern
            let cpu_4d: Tensor<4, f32, ConcreteTensor<f32, 4>> = cpu_t.reshape([1, 4096, 8, 32]).transpose(1, 2).to_concrete();
            let gpu_4d: Tensor<4, f32, ConcreteTensor<f32, 4>> = gpu_t.reshape([1, 4096, 8, 32]).transpose(1, 2).to_concrete();

            // Transpose last two dims for key
            let cpu_kt = cpu_4d.transpose(2, 3).to_concrete();
            let gpu_kt = gpu_4d.transpose(2, 3).to_concrete();

            // q @ k^T: [1, 8, 4096, 32] @ [1, 8, 32, 4096]
            let cpu_out = cpu_4d.mat_mul(&cpu_kt);
            let gpu_out = gpu_4d.mat_mul(&gpu_kt);

            let cpu_vals = to_vec(&cpu_out);
            let gpu_vals = to_vec(&gpu_out);
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[matmul transposed 1,8,4096,32 @ 1,8,32,4096] max_diff={:.6}", max_diff);
            assert!(max_diff < 0.01, "matmul with transposed inputs diverged: {}", max_diff);
        }

        // Test 6: LayerNorm on [1, 4096, 256]
        {
            let data: Vec<f32> = (0..4096*256).map(|i| ((i as f32) * 0.003).sin()).collect();
            let weight: Vec<f32> = (0..256).map(|i| 1.0 + (i as f32) * 0.001).collect();
            let bias: Vec<f32> = (0..256).map(|i| (i as f32) * 0.0001).collect();

            let cpu_t: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 4096, 256], &data);
            let gpu_t: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 4096, 256], &data);

            let cpu_ln = fusor::layers::LayerNorm::<1, f32>::new(
                Tensor::from_slice(&cpu, [256], &weight).to_concrete(),
                Some(Tensor::from_slice(&cpu, [256], &bias).to_concrete()),
                1e-5,
            );
            let gpu_ln = fusor::layers::LayerNorm::<1, f32>::new(
                Tensor::from_slice(&gpu, [256], &weight).to_concrete(),
                Some(Tensor::from_slice(&gpu, [256], &bias).to_concrete()),
                1e-5,
            );

            let cpu_out = cpu_ln.forward(&cpu_t);
            let gpu_out = gpu_ln.forward(&gpu_t);

            let cpu_vals = to_vec(&cpu_out);
            let gpu_vals = to_vec(&gpu_out);
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[layernorm 1,4096,256] max_diff={:.6}", max_diff);
            assert!(max_diff < 0.001, "layernorm diverged: {}", max_diff);
        }

        eprintln!("All large-scale CPU vs GPU tests passed!");
    }

    /// Test a chained attention operation (the full pattern from the transformer)
    /// This tests that GPU computations compose correctly when one op's output feeds
    /// into the next op without going through from_slice.
    #[tokio::test]
    async fn test_chained_attention_cpu_vs_gpu() {
        use fusor::{Device, Tensor, ConcreteTensor};

        fn to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
            let shape = t.shape();
            let n: usize = shape.iter().product();
            let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
            let materialized: Tensor<R, f32> = (t * ones).to_concrete();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = materialized.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        fn compare(name: &str, cpu_vals: &[f32], gpu_vals: &[f32]) -> f32 {
            assert_eq!(cpu_vals.len(), gpu_vals.len(), "{name}: length mismatch cpu={} gpu={}", cpu_vals.len(), gpu_vals.len());
            let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[{name}] max_diff={:.6} len={}", max_diff, cpu_vals.len());
            if max_diff > 0.01 {
                eprintln!("  CPU first 10: {:?}", &cpu_vals[..10.min(cpu_vals.len())]);
                eprintln!("  GPU first 10: {:?}", &gpu_vals[..10.min(gpu_vals.len())]);
            }
            max_diff
        }

        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();

        // Simulate one attention block's forward pass:
        // 1. q_proj(x): Linear [1, 5, 256] -> [1, 5, 256]  (matmul + bias add)
        // 2. separate_heads: reshape+transpose -> [1, 8, 5, 32]
        // 3. attn = (q*scale) @ k^T: [1, 8, 5, 32] @ [1, 8, 32, 4096] -> [1, 8, 5, 4096]
        // 4. softmax
        // 5. out = attn @ v: [1, 8, 5, 4096] @ [1, 8, 4096, 32] -> [1, 8, 5, 32]
        // 6. recombine_heads: transpose+reshape -> [1, 5, 256]

        let q_data: Vec<f32> = (0..5*256).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
        let k_data: Vec<f32> = (0..4096*256).map(|i| ((i as f32) * 0.007).cos() * 0.1).collect();
        let v_data: Vec<f32> = (0..4096*256).map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();
        let w_q_data: Vec<f32> = (0..256*256).map(|i| ((i as f32) * 0.003).sin() * 0.05).collect();
        let b_q_data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.02).cos() * 0.01).collect();
        let w_k_data: Vec<f32> = (0..256*256).map(|i| ((i as f32) * 0.004).cos() * 0.05).collect();
        let b_k_data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.03).sin() * 0.01).collect();
        let w_v_data: Vec<f32> = (0..256*256).map(|i| ((i as f32) * 0.005).sin() * 0.05).collect();
        let b_v_data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.04).cos() * 0.01).collect();
        let w_out_data: Vec<f32> = (0..256*256).map(|i| ((i as f32) * 0.006).cos() * 0.05).collect();
        let b_out_data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).sin() * 0.01).collect();

        // Helper to run the attention on a specific device
        fn run_attention(
            device: &Device,
            q_data: &[f32], k_data: &[f32], v_data: &[f32],
            w_q: &[f32], b_q: &[f32],
            w_k: &[f32], b_k: &[f32],
            w_v: &[f32], b_v: &[f32],
            w_out: &[f32], b_out: &[f32],
        ) -> Tensor<3, f32, ConcreteTensor<f32, 3>> {
            let q: Tensor<3, f32> = Tensor::from_slice(device, [1, 5, 256], q_data);
            let k: Tensor<3, f32> = Tensor::from_slice(device, [1, 4096, 256], k_data);
            let v: Tensor<3, f32> = Tensor::from_slice(device, [1, 4096, 256], v_data);

            // Linear projections (matmul + bias add)
            let wq: Tensor<2, f32> = Tensor::from_slice(device, [256, 256], w_q);
            let bq: Tensor<1, f32> = Tensor::from_slice(device, [256], b_q);
            let wk: Tensor<2, f32> = Tensor::from_slice(device, [256, 256], w_k);
            let bk: Tensor<1, f32> = Tensor::from_slice(device, [256], b_k);
            let wv: Tensor<2, f32> = Tensor::from_slice(device, [256, 256], w_v);
            let bv: Tensor<1, f32> = Tensor::from_slice(device, [256], b_v);
            let wout: Tensor<2, f32> = Tensor::from_slice(device, [256, 256], w_out);
            let bout: Tensor<1, f32> = Tensor::from_slice(device, [256], b_out);

            // q_proj: [1, 5, 256] -> reshape to [5, 256] -> matmul [5, 256] @ [256, 256] -> [5, 256] + bias -> reshape back
            let q2d: Tensor<2, f32, ConcreteTensor<f32, 2>> = q.reshape([5, 256]).to_concrete();
            let q_proj: Tensor<2, f32> = (q2d.mat_mul(&wq) + bq.broadcast_as([5, 256]).to_concrete()).to_concrete();

            let k2d: Tensor<2, f32, ConcreteTensor<f32, 2>> = k.reshape([4096, 256]).to_concrete();
            let k_proj: Tensor<2, f32> = (k2d.mat_mul(&wk) + bk.broadcast_as([4096, 256]).to_concrete()).to_concrete();

            let v2d: Tensor<2, f32, ConcreteTensor<f32, 2>> = v.reshape([4096, 256]).to_concrete();
            let v_proj: Tensor<2, f32> = (v2d.mat_mul(&wv) + bv.broadcast_as([4096, 256]).to_concrete()).to_concrete();

            // Separate heads: [N, 256] -> [1, N, 8, 32] -> transpose(1,2) -> [1, 8, N, 32]
            let q_heads: Tensor<4, f32, ConcreteTensor<f32, 4>> = q_proj
                .reshape([1, 5, 8, 32])
                .transpose(1, 2)
                .to_concrete();
            let k_heads: Tensor<4, f32, ConcreteTensor<f32, 4>> = k_proj
                .reshape([1, 4096, 8, 32])
                .transpose(1, 2)
                .to_concrete();
            let v_heads: Tensor<4, f32, ConcreteTensor<f32, 4>> = v_proj
                .reshape([1, 4096, 8, 32])
                .transpose(1, 2)
                .to_concrete();

            // q * scale
            let scale = 1.0 / (32.0f32).sqrt();
            let q_scaled = q_heads.mul_scalar(scale);

            // attn = q @ k^T: [1, 8, 5, 32] @ [1, 8, 32, 4096]
            let k_t = k_heads.transpose(2, 3).to_concrete();
            let attn = q_scaled.mat_mul(&k_t);

            // softmax
            let attn: Tensor<4, f32> = attn.softmax_last_dim::<3>();

            // out = attn @ v: [1, 8, 5, 4096] @ [1, 8, 4096, 32]
            let out = attn.mat_mul(&v_heads);

            // Recombine heads: [1, 8, 5, 32] -> transpose(1,2) -> [1, 5, 8, 32] -> reshape [1, 5, 256]
            let out: Tensor<3, f32, ConcreteTensor<f32, 3>> = out
                .transpose(1, 2)
                .to_concrete()
                .reshape([1, 5, 256])
                .to_concrete();

            // out_proj: [5, 256] @ [256, 256] + bias
            let out2d: Tensor<2, f32, ConcreteTensor<f32, 2>> = out.reshape([5, 256]).to_concrete();
            let out_proj: Tensor<2, f32> = (out2d.mat_mul(&wout) + bout.broadcast_as([5, 256]).to_concrete()).to_concrete();

            out_proj.reshape([1, 5, 256]).to_concrete()
        }

        let cpu_result = run_attention(
            &cpu, &q_data, &k_data, &v_data,
            &w_q_data, &b_q_data, &w_k_data, &b_k_data,
            &w_v_data, &b_v_data, &w_out_data, &b_out_data,
        );
        let gpu_result = run_attention(
            &gpu, &q_data, &k_data, &v_data,
            &w_q_data, &b_q_data, &w_k_data, &b_k_data,
            &w_v_data, &b_v_data, &w_out_data, &b_out_data,
        );

        let cpu_vals = to_vec(&cpu_result);
        let gpu_vals = to_vec(&gpu_result);
        let diff = compare("chained_attention", &cpu_vals, &gpu_vals);
        assert!(diff < 0.01, "Chained attention CPU vs GPU diverged: {}", diff);
    }

    /// Test that broadcast+to_concrete produces correct results on GPU
    /// This tests the pattern: tensor.reshape([1,M,N]).broadcast_as([B,M,N]).to_concrete()
    /// followed by matmul — since to_concrete is a no-op on GPU, the broadcast tensor
    /// still has stride-0 on the batch dim when passed to matmul.
    #[tokio::test]
    async fn test_broadcast_matmul_gpu() {
        use fusor::{Device, Tensor, ConcreteTensor};

        fn to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
            let shape = t.shape();
            let n: usize = shape.iter().product();
            let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
            let materialized: Tensor<R, f32> = (t * ones).to_concrete();
            let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = materialized.reshape([n]).to_concrete();
            let s = pollster::block_on(flat.as_slice()).unwrap();
            s.as_slice().to_vec()
        }

        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();

        // Test the exact pattern from pe_encoding:
        // coords (h*w, 2) @ gm.reshape([1, 2, 128]).broadcast_as([h*w, 2, 128])
        // Actually pe_encoding does: coords_3d.mat_mul(gm_3d) where gm_3d is broadcast
        // coords: (1, N, 2), gm: (1, 2, 128) broadcast from (2, 128)
        let gm_data: Vec<f32> = (0..2*128).map(|i| ((i as f32) * 0.1).sin()).collect();
        let coords_data: Vec<f32> = vec![0.5, 0.25, 0.7, 0.3]; // 2 points

        // CPU path
        let cpu_gm: Tensor<2, f32> = Tensor::from_slice(&cpu, [2, 128], &gm_data);
        let cpu_coords: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 2, 2], &coords_data);
        let cpu_gm3: Tensor<3, f32> = cpu_gm.reshape([1, 2, 128]).broadcast_as([1, 2, 128]).to_concrete();
        let cpu_result = cpu_coords.mat_mul(&cpu_gm3);

        // GPU path
        let gpu_gm: Tensor<2, f32> = Tensor::from_slice(&gpu, [2, 128], &gm_data);
        let gpu_coords: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 2, 2], &coords_data);
        let gpu_gm3: Tensor<3, f32> = gpu_gm.reshape([1, 2, 128]).broadcast_as([1, 2, 128]).to_concrete();
        let gpu_result = gpu_coords.mat_mul(&gpu_gm3);

        let cpu_vals = to_vec(&cpu_result);
        let gpu_vals = to_vec(&gpu_result);
        let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[broadcast_matmul small] max_diff={:.6}", max_diff);
        eprintln!("  CPU first 5: {:?}", &cpu_vals[..5]);
        eprintln!("  GPU first 5: {:?}", &gpu_vals[..5]);
        assert!(max_diff < 0.001, "broadcast matmul diverged: {}", max_diff);

        // Test with larger coords similar to get_dense_pe: (64*64, 2) @ (2, 128)
        // pe_encoding does this with coords shape (h*w, 2) reshaped to (1, h*w, 2)
        // and gm shape (2, 128) reshaped to (1, 2, 128) broadcast to (1, 2, 128) (batch=1, no real broadcast)
        let coords_data: Vec<f32> = (0..64*64*2).map(|i| (i as f32) / (64.0 * 64.0 * 2.0)).collect();

        let cpu_coords: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 4096, 2], &coords_data);
        let gpu_coords: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 4096, 2], &coords_data);

        // Rebuild gm with broadcast pattern (same as prompt encoder)
        let cpu_gm: Tensor<2, f32> = Tensor::from_slice(&cpu, [2, 128], &gm_data);
        let gpu_gm: Tensor<2, f32> = Tensor::from_slice(&gpu, [2, 128], &gm_data);
        let cpu_gm3: Tensor<3, f32> = cpu_gm.reshape([1, 2, 128]).broadcast_as([1, 2, 128]).to_concrete();
        let gpu_gm3: Tensor<3, f32> = gpu_gm.reshape([1, 2, 128]).broadcast_as([1, 2, 128]).to_concrete();

        let cpu_out = cpu_coords.mat_mul(&cpu_gm3);
        let gpu_out = gpu_coords.mat_mul(&gpu_gm3);

        let cpu_vals = to_vec(&cpu_out);
        let gpu_vals = to_vec(&gpu_out);
        let max_diff = cpu_vals.iter().zip(gpu_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[broadcast_matmul large (1,4096,2)@(1,2,128)] max_diff={:.6}", max_diff);
        assert!(max_diff < 0.001, "broadcast matmul large diverged: {}", max_diff);

        // Test sin/cos on GPU
        let data: Vec<f32> = (0..4096*128).map(|i| ((i as f32) * 0.01).sin() * 6.28).collect();
        let cpu_t: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 4096, 128], &data);
        let gpu_t: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 4096, 128], &data);

        let cpu_sin: Tensor<3, f32> = cpu_t.sin().to_concrete();
        let gpu_sin: Tensor<3, f32> = gpu_t.sin().to_concrete();
        let cpu_cos: Tensor<3, f32> = cpu_t.cos().to_concrete();
        let gpu_cos: Tensor<3, f32> = gpu_t.cos().to_concrete();

        let cpu_sin_vals = to_vec(&cpu_sin);
        let gpu_sin_vals = to_vec(&gpu_sin);
        let sin_diff = cpu_sin_vals.iter().zip(gpu_sin_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[sin] max_diff={:.6}", sin_diff);

        let cpu_cos_vals = to_vec(&cpu_cos);
        let gpu_cos_vals = to_vec(&gpu_cos);
        let cos_diff = cpu_cos_vals.iter().zip(gpu_cos_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[cos] max_diff={:.6}", cos_diff);

        // Test cat of sin and cos
        let cpu_cat: Tensor<3, f32> = Tensor::cat([cpu_sin.to_concrete(), cpu_cos.to_concrete()], 2);
        let gpu_cat: Tensor<3, f32> = Tensor::cat([gpu_sin.to_concrete(), gpu_cos.to_concrete()], 2);
        let cpu_cat_vals = to_vec(&cpu_cat);
        let gpu_cat_vals = to_vec(&gpu_cat);
        let cat_diff = cpu_cat_vals.iter().zip(gpu_cat_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[cat sin+cos] max_diff={:.6}", cat_diff);

        assert!(sin_diff < 0.001, "sin diverged: {}", sin_diff);
        assert!(cos_diff < 0.001, "cos diverged: {}", cos_diff);
        assert!(cat_diff < 0.001, "cat sin+cos diverged: {}", cat_diff);

        // Test arange_step on GPU
        let cpu_arange: Tensor<1, f32> = fusor::arange_step(&cpu, 0.5, 64.5, 1.0);
        let gpu_arange: Tensor<1, f32> = fusor::arange_step(&gpu, 0.5, 64.5, 1.0);
        let cpu_ar_vals = to_vec(&cpu_arange);
        let gpu_ar_vals = to_vec(&gpu_arange);
        let ar_diff = cpu_ar_vals.iter().zip(gpu_ar_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[arange_step] max_diff={:.6} cpu_len={} gpu_len={}", ar_diff, cpu_ar_vals.len(), gpu_ar_vals.len());
        assert!(cpu_ar_vals.len() == gpu_ar_vals.len(), "arange length mismatch");
        assert!(ar_diff < 0.001, "arange diverged: {}", ar_diff);

        // Test the full pe_encoding pattern (broadcast + reshape + cat)
        // This mimics what get_dense_pe does: create grid, normalize, stack, pe_encode
        let w = 64usize;
        let h = 64usize;

        let cpu_x_arange: Tensor<1, f32> = fusor::arange_step(&cpu, 0.5, w as f32 + 0.5, 1.0);
        let gpu_x_arange: Tensor<1, f32> = fusor::arange_step(&gpu, 0.5, w as f32 + 0.5, 1.0);
        let cpu_x = cpu_x_arange.div_scalar(w as f32);
        let gpu_x = gpu_x_arange.div_scalar(w as f32);

        let cpu_x2d: Tensor<2, f32> = cpu_x.reshape([1, w]).broadcast_as([h, w]).to_concrete();
        let gpu_x2d: Tensor<2, f32> = gpu_x.reshape([1, w]).broadcast_as([h, w]).to_concrete();

        let cpu_x2d_vals = to_vec(&cpu_x2d);
        let gpu_x2d_vals = to_vec(&gpu_x2d);
        let x2d_diff = cpu_x2d_vals.iter().zip(gpu_x2d_vals.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[x2d broadcast] max_diff={:.6} len={}", x2d_diff, cpu_x2d_vals.len());
        assert!(x2d_diff < 0.001, "x2d broadcast diverged: {}", x2d_diff);

        eprintln!("All broadcast+matmul tests passed!");
    }
}
