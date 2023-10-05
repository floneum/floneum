//! SAM: Segment Anything Model
//! https://github.com/facebookresearch/segment-anything

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::{self, Sam};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};

pub struct SegmentAnythingBuilder {
    source: Option<SegmentAnythingSource>,

    cpu: bool,

    /// Use the TinyViT based models from MobileSAM
    use_tiny: bool,
}

impl Default for SegmentAnythingBuilder {
    fn default() -> Self {
        Self {
            source: None,
            cpu: false,
            use_tiny: true,
        }
    }
}

impl SegmentAnythingBuilder {
    pub fn source(mut self, source: SegmentAnythingSource) -> Self {
        self.source = Some(source);
        self
    }

    pub fn cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn use_tiny(mut self, use_tiny: bool) -> Self {
        self.use_tiny = use_tiny;
        self
    }

    pub fn build(self) -> anyhow::Result<SegmentAnything> {
        SegmentAnything::new(self)
    }
}

pub struct SegmentAnythingSource {
    model: String,
    filename: String,
}

pub struct InferenceSettings {
    threshold: f32,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image).
    goal_points: Vec<(f64, f64)>,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image).
    avoid_points: Vec<(f64, f64)>,

    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

impl InferenceSettings {
    pub fn new<I: GenericImageView<Pixel = Rgba<u8>>>(input: I) -> anyhow::Result<Self> {
        let mut image = ImageBuffer::new(input.width() as u32, input.height() as u32);
        image.copy_from(&input, 0, 0)?;
        Ok(Self {
            threshold: 0.,
            goal_points: Vec::new(),
            avoid_points: Vec::new(),
            image,
        })
    }
}

impl InferenceSettings {
    /// Sets the detection threshold for the mask, 0 is the default value.
    /// - A negative values makes the model return a larger mask.
    /// - A positive makes the model return a smaller mask.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Add a point to the list of points to segment.
    pub fn add_goal_points(&mut self, x: f64, y: f64) {
        self.goal_points.push((x, y));
    }

    /// Set the list of points to segment.
    pub fn set_goal_points(&mut self, points: Vec<(f64, f64)>) {
        self.goal_points = points;
    }

    /// Add a point to the list of points to avoid.
    pub fn add_avoid_points(&mut self, x: f64, y: f64) {
        self.avoid_points.push((x, y));
    }

    /// Set the list of points to avoid.
    pub fn set_avoid_points(&mut self, points: Vec<(f64, f64)>) {
        self.avoid_points = points;
    }

    /// Set the image to segment.
    pub fn set_image<I: GenericImageView<Pixel = Rgba<u8>>>(
        &mut self,
        image: I,
    ) -> anyhow::Result<()> {
        self.image = ImageBuffer::new(image.width() as u32, image.height() as u32);
        Ok(self.image.copy_from(&image, 0, 0)?)
    }
}

pub struct SegmentAnything {
    device: Device,
    sam: Sam,
}

impl SegmentAnything {
    pub fn builder() -> SegmentAnythingBuilder {
        SegmentAnythingBuilder::default()
    }

    fn new(settings: SegmentAnythingBuilder) -> anyhow::Result<Self> {
        let SegmentAnythingBuilder {
            source,
            cpu,
            use_tiny,
        } = settings;
        let source = source.unwrap_or_else(|| SegmentAnythingSource {
            model: "lmz/candle-sam".into(),
            filename: if use_tiny {
                "mobile_sam-tiny-vitt.safetensors"
            } else {
                "sam_vit_b_01ec64.safetensors"
            }
            .into(),
        });
        let model = {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(source.model);
            api.get(&source.filename)?
        };
        let device = device(cpu)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
        let sam = if use_tiny {
            sam::Sam::new_tiny(vb)? // tiny vit_t
        } else {
            sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)? // sam_vit_b
        };
        Ok(Self { device, sam })
    }

    pub fn segment_from_points(&self, settings: InferenceSettings) -> anyhow::Result<()> {
        let InferenceSettings {
            threshold,
            goal_points,
            avoid_points,
            image,
        } = settings;

        let image = image::DynamicImage::ImageRgba8(image);

        let mut output_image = image.clone();
        let image_tensor = self.image_to_tensor(image)?;

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

        let (mask, _iou_predictions) = self.sam.forward(&image_tensor, &*points, false)?;

        let mask = (mask.ge(threshold)? * 255.)?;
        let (_one, h, w) = mask.dims3()?;
        let mask = mask.expand((3, h, w))?;

        let mask_pixels = mask.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            match image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels) {
                Some(image) => image,
                None => anyhow::bail!("error saving merged image"),
            };
        let mask_img = image::DynamicImage::from(mask_img).resize_to_fill(
            output_image.width(),
            output_image.height(),
            image::imageops::FilterType::CatmullRom,
        );
        for x in 0..output_image.width() {
            for y in 0..output_image.height() {
                let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_img, x, y);
                if mask_p.0[0] > 100 {
                    let mut img_p = imageproc::drawing::Canvas::get_pixel(&output_image, x, y);
                    img_p.0[2] = 255 - (255 - img_p.0[2]) / 2;
                    img_p.0[1] /= 2;
                    img_p.0[0] /= 2;
                    imageproc::drawing::Canvas::draw_pixel(&mut output_image, x, y, img_p)
                }
            }
        }

        Ok(())
    }

    fn image_to_tensor(&self, image: DynamicImage) -> anyhow::Result<Tensor> {
        let image = {
            let resize_longest = sam::IMAGE_SIZE;
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
        let image = Tensor::from_vec(data, (height, width, 3), &self.device)?.permute((2, 0, 1))?;

        let image = image.to_device(&self.device)?;

        Ok(image)
    }

    pub fn segment_everything(&self, image: DynamicImage) -> anyhow::Result<Vec<DynamicImage>> {
        let image = self.image_to_tensor(image)?;

        // Default options similar to the Python version.
        let bboxes = self.sam.generate_masks(
            &image,
            /* points_per_side */ 32,
            /* crop_n_layer */ 0,
            /* crop_overlap_ratio */ 512. / 1500.,
            /* crop_n_points_downscale_factor */ 1,
        )?;
        let mut masks = Vec::new();
        for bbox in bboxes {
            let mask = (&bbox.data.to_dtype(DType::U8)? * 255.)?;
            let (h, w) = mask.dims2()?;
            let mask = mask.broadcast_as((3, h, w))?;
            let (channel, height, width) = mask.dims3()?;
            if channel != 3 {
                anyhow::bail!("save_image expects an input of shape (3, height, width)")
            }
            let mask = mask.permute((1, 2, 0))?.flatten_all()?;
            let pixels = mask.to_vec1::<u8>()?;
            let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                    Some(image) => image,
                    None => anyhow::bail!("error creating image from tensor"),
                };
            let image = image::DynamicImage::from(image);
            let image =
                image.resize_to_fill(w as u32, h as u32, image::imageops::FilterType::CatmullRom);
            masks.push(image);
        }

        Ok(masks)
    }
}

fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            tracing::warn!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(device)
    }
}
