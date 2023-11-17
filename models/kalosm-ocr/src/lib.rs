//! # Kalosm OCR
//!
//! A rust wrapper for [TR OCR](https://huggingface.co/docs/transformers/model_doc/trocr)
//!
//! ## Usage
//!
//! ```rust, no_run
//! use kalosm_ocr::*;
//!
//! let mut model = Ocr::builder().build().unwrap();
//! let image = image::open("examples/ocr.png").unwrap();
//! let text = model
//!     .recognize_text(
//!         OcrInferenceSettings::new(image)
//!             .unwrap(),
//!     )
//!     .unwrap();
//!
//! println!("{}", text);
//! ```

#![warn(missing_docs)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod image_processor;

use anyhow::anyhow;
use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::trocr;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use image::{GenericImage, GenericImageView, ImageBuffer, Rgba};
use tokenizers::Tokenizer;

/// A builder for [`Ocr`].
#[derive(Default)]
pub struct OcrBuilder {
    source: OcrSource,

    cpu: bool,
}

impl OcrBuilder {
    /// Sets the source of the model.
    pub fn source(mut self, source: OcrSource) -> Self {
        self.source = source;
        self
    }

    /// Set to true to run the model on CPU.
    pub fn cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    /// Builds the [`Ocr`] model.
    pub fn build(self) -> anyhow::Result<Ocr> {
        Ocr::new(self)
    }
}

/// The source of the model.
pub struct OcrSource {
    repo: String,
    branch: String,
    filename: String,
}

impl OcrSource {
    /// Creates a new [`OcrSource`].
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            repo: model.into(),
            branch: "main".to_string(),
            filename: "model.safetensors".to_string(),
        }
    }

    /// Sets the branch of the model.
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = branch.into();
        self
    }

    /// Sets the filename of the model.
    pub fn filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = filename.into();
        self
    }

    /// Create the base model source.
    pub fn base() -> Self {
        Self::new("microsoft/trocr-base-handwritten").branch("refs/pr/3")
    }

    /// Create a normal sized model source.
    pub fn large() -> Self {
        Self::new("microsoft/trocr-large-handwritten").branch("refs/pr/6")
    }
}

impl Default for OcrSource {
    fn default() -> Self {
        Self::base()
    }
}

/// Settings for running inference on [`Ocr`].
pub struct OcrInferenceSettings {
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

impl OcrInferenceSettings {
    /// Creates a new [`OcrInferenceSettings`] from an image.
    pub fn new<I: GenericImageView<Pixel = Rgba<u8>>>(input: I) -> anyhow::Result<Self> {
        let mut image = ImageBuffer::new(input.width(), input.height());
        image.copy_from(&input, 0, 0)?;
        Ok(Self { image })
    }

    /// Set the image to segment.
    pub fn set_image<I: GenericImageView<Pixel = Rgba<u8>>>(
        mut self,
        image: I,
    ) -> anyhow::Result<Self> {
        self.image = ImageBuffer::new(image.width(), image.height());
        self.image.copy_from(&image, 0, 0)?;
        Ok(self)
    }
}

/// The [segment anything](https://segment-anything.com/) model.
pub struct Ocr {
    device: Device,
    decoder: trocr::TrOCRModel,
    decoder_config: trocr::TrOCRConfig,
    processor: image_processor::ViTImageProcessor,
    tokenizer_dec: Tokenizer,
}

impl Ocr {
    /// Creates a new [`OcrBuilder`].
    pub fn builder() -> OcrBuilder {
        OcrBuilder::default()
    }

    fn new(settings: OcrBuilder) -> anyhow::Result<Self> {
        let OcrBuilder { source, cpu } = settings;
        let tokenizer_dec = {
            let tokenizer = Api::new()?
                .model(String::from("ToluClassics/candle-trocr-tokenizer"))
                .get("tokenizer.json")?;

            Tokenizer::from_file(&tokenizer).map_err(|e| anyhow!(e))?
        };
        let device = device(cpu)?;

        let vb = {
            let model = Api::new()?
                .repo(Repo::with_revision(
                    source.repo.clone(),
                    hf_hub::RepoType::Model,
                    source.branch.clone(),
                ))
                .get(&source.filename)?;
            unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? }
        };

        let encoder_config =
            candle_transformers::models::vit::Config::microsoft_trocr_base_handwritten();

        let decoder_config = trocr::TrOCRConfig::default();
        let model = trocr::TrOCRModel::new(&encoder_config, &decoder_config, vb)?;

        let config = image_processor::ProcessorConfig::default();
        let processor = image_processor::ViTImageProcessor::new(&config);

        Ok(Self {
            device,
            decoder: model,
            processor,
            decoder_config,
            tokenizer_dec,
        })
    }

    /// Segment an image from a list of points. Returns a [`DynamicImage`] mask.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm_ocr::*;
    ///
    /// let mut model = Ocr::builder().build().unwrap();
    /// let image = image::open("examples/ocr.png").unwrap();
    /// let text = model
    ///     .recognize_text(
    ///         OcrInferenceSettings::new(image)
    ///             .unwrap(),
    ///     )
    ///     .unwrap();
    ///
    /// println!("{}", text);
    /// ```
    pub fn recognize_text(&mut self, settings: OcrInferenceSettings) -> anyhow::Result<String> {
        let OcrInferenceSettings { image } = settings;

        let image = image::DynamicImage::ImageRgba8(image);

        let image = vec![image];
        let image = self.processor.preprocess(image)?;

        let encoder_xs = self.decoder.encoder().forward(&image)?;

        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(1337, None, None);

        let mut token_ids: Vec<u32> = vec![self.decoder_config.decoder_start_token_id];
        for index in 0..1000 {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self.decoder.decode(&input_ids, &encoder_xs, start_pos)?;

            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            token_ids.push(token);

            if token == self.decoder_config.eos_token_id {
                break;
            }
        }

        let decoded = self
            .tokenizer_dec
            .decode(&token_ids, true)
            .map_err(|e| anyhow!(e))?;

        Ok(decoded)
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
