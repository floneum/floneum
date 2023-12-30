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
use serde::Deserialize;
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

/// Configuration for a [`TrOCRModel`] decoder.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OcrDecoderConfig {
    /// The vocabulary size of the model.
    pub vocab_size: usize,
    /// The dimensionality of the model.
    pub d_model: usize,
    /// The hidden size of the model.
    #[serde(alias = "cross_attention_hidden_size")]
    pub hidden_size: usize,
    /// The number of layers in the model.
    pub decoder_layers: usize,
    /// The number of attention heads in the model.
    pub decoder_attention_heads: usize,
    /// The dimensionality of the feed forward network in the model.
    pub decoder_ffn_dim: usize,
    /// The activation function of the model.
    pub activation_function: candle_nn::Activation,
    /// The maximum position embeddings of the model.
    pub max_position_embeddings: usize,
    /// The dropout of the model.
    pub dropout: f64,
    /// The attention dropout of the model.
    pub attention_dropout: f64,
    /// The activation dropout of the model.
    pub activation_dropout: f64,
    /// The start token id of the model.
    pub decoder_start_token_id: u32,
    /// The init std of the model.
    pub init_std: f64,
    /// The decoder layerdrop of the model.
    pub decoder_layerdrop: f64,
    /// Whether to use cache in the model.
    pub use_cache: bool,
    /// Whether to scale the embedding in the model.
    pub scale_embedding: bool,
    /// Whether to use learned position embeddings in the model.
    pub use_learned_position_embeddings: bool,
    /// Whether to use layer norm in the model.
    pub layernorm_embedding: bool,
    /// The padding token id of the model.
    pub pad_token_id: usize,
    /// The beginning of sentence token id of the model.
    pub bos_token_id: usize,
    /// The end of sentence token id of the model.
    pub eos_token_id: u32,
    /// The number of attention heads in the model.
    pub num_attention_heads: usize,
    /// The vocabulary size of the model.
    pub decoder_vocab_size: Option<usize>,
}

impl OcrDecoderConfig {
    /// Create a new [`OcrDecoderConfig`] for a [large print text model](https://huggingface.co/microsoft/trocr-large-printed)
    pub fn microsoft_trocr_large_printed() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 1024,
            hidden_size: 1024,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: candle_nn::Activation::Relu,
            max_position_embeddings: 1024,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: 2,
            init_std: 0.02,
            decoder_layerdrop: 0.0,
            use_cache: true,
            scale_embedding: true,
            use_learned_position_embeddings: false,
            layernorm_embedding: false,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            num_attention_heads: 16,
            decoder_vocab_size: Some(50265),
        }
    }
}

impl Default for OcrDecoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 1024,
            hidden_size: 768,
            decoder_layers: 12,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            activation_function: candle_nn::Activation::Gelu,
            max_position_embeddings: 512,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_dropout: 0.0,
            decoder_start_token_id: 2,
            init_std: 0.02,
            decoder_layerdrop: 0.0,
            use_cache: true,
            scale_embedding: false,
            use_learned_position_embeddings: true,
            layernorm_embedding: true,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            num_attention_heads: 12,
            decoder_vocab_size: Some(50265),
        }
    }
}

impl From<OcrDecoderConfig> for trocr::TrOCRConfig {
    fn from(val: OcrDecoderConfig) -> Self {
        trocr::TrOCRConfig {
            vocab_size: val.vocab_size,
            d_model: val.d_model,
            hidden_size: val.hidden_size,
            decoder_layers: val.decoder_layers,
            decoder_attention_heads: val.decoder_attention_heads,
            decoder_ffn_dim: val.decoder_ffn_dim,
            activation_function: val.activation_function,
            max_position_embeddings: val.max_position_embeddings,
            dropout: val.dropout,
            attention_dropout: val.attention_dropout,
            activation_dropout: val.activation_dropout,
            decoder_start_token_id: val.decoder_start_token_id,
            init_std: val.init_std,
            decoder_layerdrop: val.decoder_layerdrop,
            use_cache: val.use_cache,
            scale_embedding: val.scale_embedding,
            use_learned_position_embeddings: val.use_learned_position_embeddings,
            layernorm_embedding: val.layernorm_embedding,
            pad_token_id: val.pad_token_id,
            bos_token_id: val.bos_token_id,
            eos_token_id: val.eos_token_id,
            num_attention_heads: val.num_attention_heads,
            decoder_vocab_size: val.decoder_vocab_size,
        }
    }
}

/// Configuration for a [`Ocr`] model encoder.
#[derive(Debug, Clone, Deserialize)]
pub struct OcrEncoderConfig {
    /// The hidden size of the model.
    pub hidden_size: usize,
    /// The number of hidden layers in the model.
    pub num_hidden_layers: usize,
    /// The number of attention heads in the model.
    pub num_attention_heads: usize,
    /// The intermediate size of the model.
    pub intermediate_size: usize,
    /// The hidden activation of the model.
    pub hidden_act: candle_nn::Activation,
    /// The layer norm epsilon of the model.
    pub layer_norm_eps: f64,
    /// The image size of the model.
    pub image_size: usize,
    /// The patch size of the model.
    pub patch_size: usize,
    /// The number of channels in the model.
    pub num_channels: usize,
    /// Whether to use qkv bias in the model.
    pub qkv_bias: bool,
}

impl From<OcrEncoderConfig> for candle_transformers::models::vit::Config {
    fn from(val: OcrEncoderConfig) -> Self {
        candle_transformers::models::vit::Config {
            hidden_size: val.hidden_size,
            num_hidden_layers: val.num_hidden_layers,
            num_attention_heads: val.num_attention_heads,
            intermediate_size: val.intermediate_size,
            hidden_act: val.hidden_act,
            layer_norm_eps: val.layer_norm_eps,
            image_size: val.image_size,
            patch_size: val.patch_size,
            num_channels: val.num_channels,
            qkv_bias: val.qkv_bias,
        }
    }
}

impl OcrEncoderConfig {
    /// Create a new [`OcrEncoderConfig`] for a [base model](https://huggingface.co/google/vit-base-patch16-224/blob/main/config.json)
    pub fn vit_base_patch16_224() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: true,
        }
    }

    /// Create a new [`OcrEncoderConfig`] for a [handwriting OCR model](https://huggingface.co/microsoft/trocr-base-handwritten)
    pub fn microsoft_trocr_base_handwritten() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 384,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: false,
        }
    }

    /// Create a new [`OcrEncoderConfig`] for a [large print text model](https://huggingface.co/microsoft/trocr-large-printed)
    pub fn microsoft_trocr_large_printed() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 384,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: false,
        }
    }
}

/// Configuration for a [`Ocr`] model.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OcrModelConfig {
    encoder: OcrEncoderConfig,
    decoder: OcrDecoderConfig,
}

impl OcrModelConfig {
    /// Create a new [`OcrModelConfig`] for a [base model](https://huggingface.co/google/vit-base-patch16-224/blob/main/config.json)
    pub fn vit_base_patch16_224() -> Self {
        Self {
            encoder: OcrEncoderConfig::vit_base_patch16_224(),
            decoder: OcrDecoderConfig::default(),
        }
    }

    /// Create a new [`OcrModelConfig`] for a [handwriting OCR model](https://huggingface.co/microsoft/trocr-base-handwritten)
    pub fn microsoft_trocr_base_handwritten() -> Self {
        Self {
            encoder: OcrEncoderConfig::microsoft_trocr_base_handwritten(),
            decoder: OcrDecoderConfig::default(),
        }
    }

    /// Create a new [`OcrModelConfig`] for a [large print text model](https://huggingface.co/microsoft/trocr-large-printed)
    pub fn microsoft_trocr_large_printed() -> Self {
        Self {
            encoder: OcrEncoderConfig::microsoft_trocr_large_printed(),
            decoder: OcrDecoderConfig::microsoft_trocr_large_printed(),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum ConfigSource {
    File(String),
    Config(OcrModelConfig),
}

/// The source of the model.
pub struct OcrSource {
    repo: String,
    branch: String,
    filename: String,
    config: ConfigSource,
}

impl OcrSource {
    /// Creates a new [`OcrSource`].
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            repo: model.into(),
            branch: "main".to_string(),
            filename: "model.safetensors".to_string(),
            config: ConfigSource::Config(OcrModelConfig::vit_base_patch16_224()),
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

    /// Set the config file of the model.
    // TODO: make this work for more config files
    #[allow(dead_code)]
    fn config_file(mut self, config: impl Into<String>) -> Self {
        self.config = ConfigSource::File(config.into());
        self
    }

    /// Set the config of the model.
    pub fn config(mut self, config: OcrModelConfig) -> Self {
        self.config = ConfigSource::Config(config);
        self
    }

    /// Create the base model source.
    pub fn base() -> Self {
        Self::new("microsoft/trocr-base-handwritten")
            .branch("refs/pr/3")
            .config(OcrModelConfig::microsoft_trocr_base_handwritten())
    }

    /// Create a normal sized model source.
    pub fn large() -> Self {
        Self::new("microsoft/trocr-large-handwritten")
            .branch("refs/pr/6")
            .config(OcrModelConfig::microsoft_trocr_base_handwritten())
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
        let repo = Api::new()?.repo(Repo::with_revision(
            source.repo.clone(),
            hf_hub::RepoType::Model,
            source.branch.clone(),
        ));

        let vb = {
            let model = repo.get(&source.filename)?;
            unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? }
        };

        let config = match source.config {
            ConfigSource::File(config) => {
                let config = repo.get(&config)?;

                let config = std::fs::read_to_string(config).map_err(|e| anyhow!(e))?;

                println!("{}", config);

                serde_json::from_str(&config).map_err(|e| anyhow!(e))?
            }
            ConfigSource::Config(config) => config,
        };

        let encoder_config = config.encoder.into();

        let decoder_config = config.decoder.into();
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
