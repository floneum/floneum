#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_core::IndexOp;
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use candle_transformers::models::wuerstchen;
use candle_transformers::models::wuerstchen::paella_vq::PaellaVQ;
use candle_transformers::models::wuerstchen::prior::WPrior;
use candle_transformers::models::{stable_diffusion, wuerstchen::diffnext::WDiffNeXt};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use image::ImageBuffer;
use tokenizers::Tokenizer;

use crate::{DiffusionResult, Image, WuerstchenInferenceSettings};

const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

pub(crate) struct WuerstcheModelSettings {
    pub(crate) use_flash_attn: bool,

    /// The decoder weight file, in .safetensors format.
    pub(crate) decoder_weights: PathBuf,

    /// The CLIP weight file, in .safetensors format.
    pub(crate) clip_weights: PathBuf,

    /// The CLIP weight file used by the prior model, in .safetensors format.
    pub(crate) prior_clip_weights: PathBuf,

    /// The prior weight file, in .safetensors format.
    pub(crate) prior_weights: PathBuf,

    /// The VQGAN weight file, in .safetensors format.
    pub(crate) vqgan_weights: PathBuf,

    /// The file specifying the tokenizer to used for tokenization.
    pub(crate) tokenizer: PathBuf,

    /// The file specifying the tokenizer to used for prior tokenization.
    pub(crate) prior_tokenizer: PathBuf,
}
/// The Wuerstchen model.
pub(crate) struct WuerstchenInner {
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

impl WuerstchenInner {
    pub(crate) fn new(settings: WuerstcheModelSettings) -> Result<Self> {
        let WuerstcheModelSettings {
            use_flash_attn,
            decoder_weights,
            clip_weights,
            prior_clip_weights,
            prior_weights,
            vqgan_weights,
            tokenizer,
            prior_tokenizer,
        } = settings;

        let prior_tokenizer = Tokenizer::from_file(prior_tokenizer).map_err(E::msg)?;

        let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

        let device = kalosm_common::accelerated_device_if_available()?;

        let clip_config = stable_diffusion::clip::Config::wuerstchen();
        let clip = stable_diffusion::build_clip_transformer(
            &clip_config,
            clip_weights,
            &device,
            DType::F32,
        )?;

        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let prior_clip = stable_diffusion::build_clip_transformer(
            &prior_clip_config,
            prior_clip_weights,
            &device,
            DType::F32,
        )?;

        let decoder = {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[decoder_weights],
                    DType::F32,
                    &device,
                )?
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
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[prior_weights],
                    DType::F32,
                    &device,
                )?
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
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[vqgan_weights],
                    DType::F32,
                    &device,
                )?
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

    fn image_embeddings(
        &self,
        settings: &WuerstchenInferenceSettings,
        b_size: usize,
    ) -> Result<Tensor> {
        let height = settings.height;
        let width = settings.width;

        let prior_text_embeddings = {
            self.encode_prompt(
                &settings.prompt,
                Some(&settings.uncond_prompt),
                &self.prior_tokenizer,
                &self.prior_clip,
                &self.prior_clip_config,
            )?
        };

        {
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
            ((latents * 42.)? - 1.).map_err(Into::into)
        }
    }

    fn generate_image(
        &self,
        text_embeddings: &Tensor,
        image_embeddings: &Tensor,
        b_size: usize,
    ) -> Result<ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
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
            let noise_pred =
                self.decoder
                    .forward(&latents, &ratio, image_embeddings, Some(text_embeddings))?;
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
        // let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        //     match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
        //         Some(image) => image,
        //         None => anyhow::bail!("error creating image {img_tensor:?}"),
        //     };
        ImageBuffer::from_raw(width as u32, height as u32, pixels)
            .ok_or(E::msg(format!("error creating image {img_tensor:?}")))
    }

    /// Run inference with the given settings.
    pub fn run(
        &self,
        settings: WuerstchenInferenceSettings,
        result: tokio::sync::mpsc::UnboundedSender<Image>,
    ) {
        let start_time = Instant::now();
        let height = settings.height;
        let width = settings.width;

        if height < 1024 || width < 1024 {
            println!("Warning: Würstchen was trained on image resolutions between 1024x1024 & 1536x1536. {}x{} is below the minimum resolution. Image quality may be poor.", height, width);
        }
        if height > 1536 || width > 1536 {
            println!("Warning: Würstchen was trained on image resolutions between 1024x1024 & 1536x1536. {}x{} is above the maximum resolution. Image quality may be poor.", height, width);
        }
        let chech_dims = if height % 128 != 0 || width % 128 != 0 {
            Err(E::msg("Image resolution must be a multiple of 128"))
        } else {
            Ok(())
        };

        let b_size = 1;

        let text_embeddings = {
            self.encode_prompt(
                &settings.prompt,
                None,
                &self.tokenizer,
                &self.clip,
                &self.clip_config,
            )
        };

        let image_embeddings = self.image_embeddings(&settings, b_size);
        if chech_dims.is_err() || text_embeddings.is_err() || image_embeddings.is_err() {
            let err = Err(chech_dims
                .err()
                .or_else(|| text_embeddings.err().or_else(|| image_embeddings.err()))
                .unwrap());
            let image = Image {
                sample_num: 0,
                elapsed_time: start_time.elapsed(),
                remaining_time: Duration::from_secs(0),
                progress: 1.,
                result: err,
            };
            if let Err(err) = result.send(image) {
                tracing::error!("Error sending segment: {err}");
            }
            return;
        }

        let text_embeddings = text_embeddings.unwrap();
        let image_embeddings = image_embeddings.unwrap();

        for index in 1..=settings.num_samples {
            let iter_start_time = Instant::now();
            let remaining_samples = (settings.num_samples - index) as u32;
            let progress = (index / settings.num_samples) as f32;

            tracing::trace!("Generating image {}/{}", index, settings.num_samples);

            let image = self
                .generate_image(&text_embeddings, &image_embeddings, b_size)
                .map(|val| DiffusionResult {
                    image: val,
                    height,
                    width,
                });

            let remaining_time = remaining_samples * iter_start_time.elapsed();

            let image = Image {
                sample_num: index,
                elapsed_time: start_time.elapsed(),
                remaining_time,
                progress,
                result: image,
            };

            if let Err(err) = result.send(image) {
                tracing::error!("Error sending segment: {err}");
                break;
            }
        }
    }
}
