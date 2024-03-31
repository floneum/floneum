//! # RWuerstchen
//!
//! RWuerstchen is a rust wrapper for library for [Wuerstchen](https://huggingface.co/papers/2306.00637) implemented in the [Candle](https://github.com/huggingface/candle) ML framework.
//!
//! RWuerstchen generates images efficiently from text prompts.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use rwuerstchen::*;
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     let model = Wuerstchen::builder().build().await?;
//!     let settings = WuerstchenInferenceSettings::new(
//!         "a cute cat with a hat in a room covered with fur with incredible detail",
//!     )
//!     .with_n_steps(2);
//!     let images = model.run(settings)?;
//!     for (i, img) in images.iter().enumerate() {
//!         img.save(&format!("{}.png", i))?;
//!     }
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_core::IndexOp;
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use candle_transformers::models::wuerstchen;
use candle_transformers::models::wuerstchen::paella_vq::PaellaVQ;
use candle_transformers::models::wuerstchen::prior::WPrior;
use candle_transformers::models::{stable_diffusion, wuerstchen::diffnext::WDiffNeXt};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use image::ImageBuffer;
use kalosm_common::FileSource;
pub use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::ModelBuilder;
use tokenizers::Tokenizer;

const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

/// A builder for the Wuerstchen model.
pub struct WuerstchenBuilder {
    use_flash_attn: bool,

    /// The decoder weight file, in .safetensors format.
    decoder_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    clip_weights: Option<String>,

    /// The CLIP weight file used by the prior model, in .safetensors format.
    prior_clip_weights: Option<String>,

    /// The prior weight file, in .safetensors format.
    prior_weights: Option<String>,

    /// The VQGAN weight file, in .safetensors format.
    vqgan_weights: Option<String>,

    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The file specifying the tokenizer to used for prior tokenization.
    prior_tokenizer: Option<String>,
}

impl Default for WuerstchenBuilder {
    fn default() -> Self {
        Self {
            use_flash_attn: { cfg!(feature = "flash-attn") },
            decoder_weights: None,
            clip_weights: None,
            prior_clip_weights: None,
            prior_weights: None,
            vqgan_weights: None,
            tokenizer: None,
            prior_tokenizer: None,
        }
    }
}

impl WuerstchenBuilder {
    /// Set whether to use the Flash Attention implementation.
    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }

    /// Set the decoder weight file, in .safetensors format.
    pub fn with_decoder_weights(mut self, decoder_weights: impl Into<String>) -> Self {
        self.decoder_weights = Some(decoder_weights.into());
        self
    }

    /// Set the CLIP weight file, in .safetensors format.
    pub fn with_clip_weights(mut self, clip_weights: impl Into<String>) -> Self {
        self.clip_weights = Some(clip_weights.into());
        self
    }

    /// Set the CLIP weight file used by the prior model, in .safetensors format.
    pub fn with_prior_clip_weights(mut self, prior_clip_weights: impl Into<String>) -> Self {
        self.prior_clip_weights = Some(prior_clip_weights.into());
        self
    }

    /// Set the prior weight file, in .safetensors format.
    pub fn with_prior_weights(mut self, prior_weights: impl Into<String>) -> Self {
        self.prior_weights = Some(prior_weights.into());
        self
    }

    /// Set the VQGAN weight file, in .safetensors format.
    pub fn with_vqgan_weights(mut self, vqgan_weights: impl Into<String>) -> Self {
        self.vqgan_weights = Some(vqgan_weights.into());
        self
    }

    /// Set the file specifying the tokenizer to used for tokenization.
    pub fn with_tokenizer(mut self, tokenizer: impl Into<String>) -> Self {
        self.tokenizer = Some(tokenizer.into());
        self
    }

    /// Set the file specifying the tokenizer to used for prior tokenization.
    pub fn with_prior_tokenizer(mut self, prior_tokenizer: impl Into<String>) -> Self {
        self.prior_tokenizer = Some(prior_tokenizer.into());
        self
    }

    /// Build the model.
    pub async fn build(self) -> Result<Wuerstchen> {
        self.build_with_loading_handler(|_| {}).await
    }

    /// Build the model with a handler for progress as the download and loading progresses.
    pub async fn build_with_loading_handler(
        self,
        progress_handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Wuerstchen> {
        Wuerstchen::new(self, progress_handler).await
    }
}

#[async_trait::async_trait]
impl ModelBuilder for WuerstchenBuilder {
    type Model = Wuerstchen;

    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> anyhow::Result<Self::Model> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        let downloaded_decoder_weights = self.decoder_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Decoder).downloaded();
        let downloaded_clip_weights = self.clip_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Clip).downloaded();
        let downloaded_prior_clip_weights = self.prior_clip_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::PriorClip).downloaded();
        let downloaded_prior_weights = self.prior_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Prior).downloaded();
        let downloaded_vqgan_weights = self.vqgan_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::VqGan).downloaded();
        let downloaded_tokenizer = self.tokenizer.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Tokenizer).downloaded();
        let downloaded_prior_tokenizer = self.prior_tokenizer.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::PriorTokenizer).downloaded();

        !(downloaded_decoder_weights
            && downloaded_clip_weights
            && downloaded_prior_clip_weights
            && downloaded_prior_weights
            && downloaded_vqgan_weights
            && downloaded_tokenizer
            && downloaded_prior_tokenizer)
    }
}

/// Settings for running inference with the Wuerstchen model.
pub struct WuerstchenInferenceSettings {
    /// The prompt to be used for image generation.
    prompt: String,

    uncond_prompt: String,

    /// The height in pixels of the generated image.
    height: usize,

    /// The width in pixels of the generated image.
    width: usize,

    /// The number of steps to run the diffusion for.
    n_steps: usize,

    /// The number of samples to generate.
    num_samples: i64,

    /// Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
    prior_guidance_scale: f64,
}

impl WuerstchenInferenceSettings {
    /// Create a new settings object with the given prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),

            uncond_prompt: String::new(),

            height: 1024,

            width: 1024,

            n_steps: 30,

            num_samples: 1,

            prior_guidance_scale: 4.0,
        }
    }

    /// Set the negative prompt to be used for image generation.
    pub fn with_negative_prompt(mut self, uncond_prompt: impl Into<String>) -> Self {
        self.uncond_prompt = uncond_prompt.into();
        self
    }

    /// Set the height in pixels of the generated image.
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Set the width in pixels of the generated image.
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the number of steps to run the diffusion for.
    pub fn with_n_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }

    /// Set the number of samples to generate.
    pub fn with_num_samples(mut self, num_samples: i64) -> Self {
        self.num_samples = num_samples;
        self
    }

    /// Set the prior guidance scale.
    pub fn with_prior_guidance_scale(mut self, prior_guidance_scale: f64) -> Self {
        self.prior_guidance_scale = prior_guidance_scale;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    PriorTokenizer,
    Clip,
    PriorClip,
    Decoder,
    VqGan,
    Prior,
}

impl ModelFile {
    fn get(&self, filename: Option<String>) -> FileSource {
        match filename {
            Some(filename) => FileSource::local(std::path::PathBuf::from(filename)),
            None => self.into(),
        }
    }
}

impl From<&ModelFile> for FileSource {
    fn from(val: &ModelFile) -> Self {
        let repo_main = "warp-ai/wuerstchen";
        let repo_prior = "warp-ai/wuerstchen-prior";
        let (repo, path) = match val {
            ModelFile::Tokenizer => (repo_main, "tokenizer/tokenizer.json"),
            ModelFile::PriorTokenizer => (repo_prior, "tokenizer/tokenizer.json"),
            ModelFile::Clip => (repo_main, "text_encoder/model.safetensors"),
            ModelFile::PriorClip => (repo_prior, "text_encoder/model.safetensors"),
            ModelFile::Decoder => (repo_main, "decoder/diffusion_pytorch_model.safetensors"),
            ModelFile::VqGan => (repo_main, "vqgan/diffusion_pytorch_model.safetensors"),
            ModelFile::Prior => (repo_prior, "prior/diffusion_pytorch_model.safetensors"),
        };
        FileSource::huggingface(repo.to_owned(), "main".to_owned(), path.to_owned())
    }
}

/// The Wuerstchen model.
pub struct Wuerstchen {
    clip: ClipTextTransformer,
    clip_config: stable_diffusion::clip::Config,
    prior_clip: ClipTextTransformer,
    prior_clip_config: stable_diffusion::clip::Config,
    decoder: WDiffNeXt,
    prior: WPrior,
    vqgan: PaellaVQ,
    prior_tokenizer: Tokenizer,
    tokenizer: Tokenizer,
    device: Device,
}

impl Wuerstchen {
    /// Create a new builder for the Wuerstchen model.
    pub fn builder() -> WuerstchenBuilder {
        WuerstchenBuilder::default()
    }

    async fn new(
        settings: WuerstchenBuilder,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self> {
        let WuerstchenBuilder {
            use_flash_attn,
            decoder_weights,
            clip_weights,
            prior_clip_weights,
            prior_weights,
            vqgan_weights,
            tokenizer,
            prior_tokenizer,
        } = settings;

        let prior_tokenizer_source = ModelFile::PriorTokenizer.get(prior_tokenizer);
        let prior_tokenizer_path = prior_tokenizer_source
            .download(|progress| {
                progress_handler(ModelLoadingProgress::downloading(
                    format!("Prior Tokenizer ({})", prior_tokenizer_source),
                    progress,
                ))
            })
            .await?;
        let prior_tokenizer = Tokenizer::from_file(prior_tokenizer_path).map_err(E::msg)?;

        let tokenizer_source = ModelFile::Tokenizer.get(tokenizer);
        let tokenizer_path = tokenizer_source
            .download(|progress| {
                progress_handler(ModelLoadingProgress::downloading(
                    format!("Tokenizer ({})", tokenizer_source),
                    progress,
                ))
            })
            .await?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let device = kalosm_common::accelerated_device_if_available()?;

        let clip_weights_source = ModelFile::Clip.get(clip_weights);
        let clip_weights = clip_weights_source
            .download(|progress| {
                progress_handler(ModelLoadingProgress::downloading(
                    format!("Weights ({})", clip_weights_source),
                    progress,
                ))
            })
            .await?;

        let clip_config = stable_diffusion::clip::Config::wuerstchen();
        let clip = stable_diffusion::build_clip_transformer(
            &clip_config,
            clip_weights,
            &device,
            DType::F32,
        )?;

        let prior_clip_weights_source = ModelFile::PriorClip.get(prior_clip_weights);
        let prior_clip_weights = prior_clip_weights_source
            .download(|progress| {
                progress_handler(ModelLoadingProgress::downloading(
                    format!("Prior Weights ({})", prior_clip_weights_source),
                    progress,
                ))
            })
            .await?;

        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let prior_clip = stable_diffusion::build_clip_transformer(
            &prior_clip_config,
            prior_clip_weights,
            &device,
            DType::F32,
        )?;

        let decoder = {
            let file_source = ModelFile::Decoder.get(decoder_weights);
            let file = file_source
                .download(|progress| {
                    progress_handler(ModelLoadingProgress::downloading(
                        format!("Decoder ({})", file_source),
                        progress,
                    ))
                })
                .await?;

            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device)?
            };
            wuerstchen::diffnext::WDiffNeXt::new(
                DECODER_CIN,
                DECODER_CIN,
                64,
                1024,
                1024,
                2,
                use_flash_attn,
                vb,
            )?
        };

        let prior = {
            let file_source = ModelFile::Prior.get(prior_weights);
            let file = file_source
                .download(|progress| {
                    progress_handler(ModelLoadingProgress::downloading(
                        format!("Decoder Prior ({})", file_source),
                        progress,
                    ))
                })
                .await?;

            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device)?
            };
            wuerstchen::prior::WPrior::new(
                /* c_in */ PRIOR_CIN,
                /* c */ 1536,
                /* c_cond */ 1280,
                /* c_r */ 64,
                /* depth */ 32,
                /* nhead */ 24,
                use_flash_attn,
                vb,
            )?
        };

        let vqgan = {
            let file_source = ModelFile::VqGan.get(vqgan_weights);
            let file = file_source
                .download(|progress| {
                    progress_handler(ModelLoadingProgress::downloading(
                        format!("VqGan ({})", file_source),
                        progress,
                    ))
                })
                .await?;

            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], DType::F32, &device)?
            };
            wuerstchen::paella_vq::PaellaVQ::new(vb)?
        };

        Ok(Self {
            clip,
            clip_config,
            prior_clip,
            prior_clip_config,
            decoder,
            prior,
            vqgan,
            prior_tokenizer,
            tokenizer,
            device,
        })
    }

    fn encode_prompt(
        &self,
        prompt: &str,
        uncond_prompt: Option<&str>,
        tokenizer: &Tokenizer,
        clip: &ClipTextTransformer,
        clip_config: &stable_diffusion::clip::Config,
    ) -> Result<Tensor> {
        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let pad_id = match &clip_config.pad_with {
            Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
        };
        let tokens_len = tokens.len();
        while tokens.len() < clip_config.max_position_embeddings {
            tokens.push(pad_id)
        }
        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        let text_embeddings = clip.forward_with_mask(&tokens, tokens_len - 1)?;
        match uncond_prompt {
            None => Ok(text_embeddings),
            Some(uncond_prompt) => {
                let mut uncond_tokens = tokenizer
                    .encode(uncond_prompt, true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let uncond_tokens_len = uncond_tokens.len();
                while uncond_tokens.len() < clip_config.max_position_embeddings {
                    uncond_tokens.push(pad_id)
                }
                let uncond_tokens =
                    Tensor::new(uncond_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

                let uncond_embeddings =
                    clip.forward_with_mask(&uncond_tokens, uncond_tokens_len - 1)?;
                let text_embeddings = Tensor::cat(&[text_embeddings, uncond_embeddings], 0)?;
                Ok(text_embeddings)
            }
        }
    }

    /// Run inference with the given settings.
    pub fn run(
        &self,
        settings: WuerstchenInferenceSettings,
    ) -> Result<Vec<ImageBuffer<image::Rgb<u8>, Vec<u8>>>> {
        let height = settings.height;
        let width = settings.width;

        if height < 1024 || width < 1024 {
            println!("Warning: Würstchen was trained on image resolutions between 1024x1024 & 1536x1536. {}x{} is below the minimum resolution. Image quality may be poor.", height, width);
        }
        if height > 1536 || width > 1536 {
            println!("Warning: Würstchen was trained on image resolutions between 1024x1024 & 1536x1536. {}x{} is above the maximum resolution. Image quality may be poor.", height, width);
        }
        if height % 128 != 0 || width % 128 != 0 {
            return Err(E::msg("Image resolution must be a multiple of 128"));
        }

        let prior_text_embeddings = {
            self.encode_prompt(
                &settings.prompt,
                Some(&settings.uncond_prompt),
                &self.prior_tokenizer,
                &self.prior_clip,
                &self.prior_clip_config,
            )?
        };

        let text_embeddings = {
            self.encode_prompt(
                &settings.prompt,
                None,
                &self.tokenizer,
                &self.clip,
                &self.clip_config,
            )?
        };

        let b_size = 1;
        let image_embeddings = {
            // https://huggingface.co/warp-ai/wuerstchen-prior/blob/main/prior/config.json
            let latent_height = (height as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
            let latent_width = (width as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
            let mut latents = Tensor::randn(
                0f32,
                1f32,
                (b_size, PRIOR_CIN, latent_height, latent_width),
                &self.device,
            )?;

            let prior_scheduler = wuerstchen::ddpm::DDPMWScheduler::new(60, Default::default())?;
            let timesteps = prior_scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            for &t in timesteps {
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
                let ratio = (Tensor::ones(2, DType::F32, &self.device)? * t)?;
                let noise_pred =
                    self.prior
                        .forward(&latent_model_input, &ratio, &prior_text_embeddings)?;
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_text, noise_pred_uncond) = (&noise_pred[0], &noise_pred[1]);
                let noise_pred = (noise_pred_uncond
                    + ((noise_pred_text - noise_pred_uncond)? * settings.prior_guidance_scale)?)?;
                latents = prior_scheduler.step(&noise_pred, t, &latents)?;
                tracing::trace!(
                    "generating embeddings t: {}, noise_pred: {:?}",
                    t,
                    noise_pred
                );
            }
            ((latents * 42.)? - 1.)?
        };

        let mut images = Vec::new();
        for _ in 0..settings.num_samples {
            tracing::trace!(
                "Generating image {}/{}",
                images.len() + 1,
                settings.num_samples
            );
            // https://huggingface.co/warp-ai/wuerstchen/blob/main/model_index.json
            let latent_height = (image_embeddings.dim(2)? as f64 * LATENT_DIM_SCALE) as usize;
            let latent_width = (image_embeddings.dim(3)? as f64 * LATENT_DIM_SCALE) as usize;

            let mut latents = Tensor::randn(
                0f32,
                1f32,
                (b_size, DECODER_CIN, latent_height, latent_width),
                &self.device,
            )?;

            let scheduler = wuerstchen::ddpm::DDPMWScheduler::new(12, Default::default())?;
            let timesteps = scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            for &t in timesteps {
                let ratio = (Tensor::ones(1, DType::F32, &self.device)? * t)?;
                let noise_pred = self.decoder.forward(
                    &latents,
                    &ratio,
                    &image_embeddings,
                    Some(&text_embeddings),
                )?;
                latents = scheduler.step(&noise_pred, t, &latents)?;
                tracing::trace!("t: {}, noise_pred: {:?}", t, noise_pred)
            }
            let img_tensor = self.vqgan.decode(&(&latents * 0.3764)?)?;
            // TODO: Add the clamping between 0 and 1.
            let img_tensor = (img_tensor * 255.)?.to_dtype(DType::U8)?.i(0)?;
            let (channel, height, width) = img_tensor.dims3()?;
            if channel != 3 {
                anyhow::bail!("image must have 3 channels");
            }
            let img = img_tensor.permute((1, 2, 0))?.flatten_all()?;
            let pixels = img.to_vec1::<u8>()?;
            let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                    Some(image) => image,
                    None => anyhow::bail!("error creating image {img_tensor:?}"),
                };

            images.push(image);
        }
        Ok(images)
    }
}
