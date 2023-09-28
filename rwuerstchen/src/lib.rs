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
use tokenizers::Tokenizer;

const PRIOR_GUIDANCE_SCALE: f64 = 4.0;
const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

pub struct WuerstchenBuilder {
    /// Run on CPU rather than on GPU.
    cpu: bool,

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
            cpu: false,
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
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }

    pub fn with_decoder_weights(mut self, decoder_weights: impl Into<String>) -> Self {
        self.decoder_weights = Some(decoder_weights.into());
        self
    }

    pub fn with_clip_weights(mut self, clip_weights: impl Into<String>) -> Self {
        self.clip_weights = Some(clip_weights.into());
        self
    }

    pub fn with_prior_clip_weights(mut self, prior_clip_weights: impl Into<String>) -> Self {
        self.prior_clip_weights = Some(prior_clip_weights.into());
        self
    }

    pub fn with_prior_weights(mut self, prior_weights: impl Into<String>) -> Self {
        self.prior_weights = Some(prior_weights.into());
        self
    }

    pub fn with_vqgan_weights(mut self, vqgan_weights: impl Into<String>) -> Self {
        self.vqgan_weights = Some(vqgan_weights.into());
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: impl Into<String>) -> Self {
        self.tokenizer = Some(tokenizer.into());
        self
    }

    pub fn with_prior_tokenizer(mut self, prior_tokenizer: impl Into<String>) -> Self {
        self.prior_tokenizer = Some(prior_tokenizer.into());
        self
    }

    pub fn build(self) -> Result<Wuerstchen> {
        Wuerstchen::new(self)
    }
}

pub struct InferenceSettings {
    /// The prompt to be used for image generation.
    prompt: String,

    uncond_prompt: String,

    /// The height in pixels of the generated image.
    height: usize,

    /// The width in pixels of the generated image.
    width: usize,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    n_steps: usize,

    /// The number of samples to generate.
    num_samples: i64,
}

impl InferenceSettings {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),

            uncond_prompt: String::new(),

            /// The height in pixels of the generated image.
            height: 1024,

            /// The width in pixels of the generated image.
            width: 1024,

            /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
            sliced_attention_size: None,

            /// The number of steps to run the diffusion for.
            n_steps: 30,

            /// The number of samples to generate.
            num_samples: 1,
        }
    }

    pub fn with_uncond_prompt(mut self, uncond_prompt: impl Into<String>) -> Self {
        self.uncond_prompt = uncond_prompt.into();
        self
    }

    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    pub fn with_sliced_attention_size(mut self, sliced_attention_size: usize) -> Self {
        self.sliced_attention_size = Some(sliced_attention_size);
        self
    }

    pub fn with_n_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }

    pub fn with_num_samples(mut self, num_samples: i64) -> Self {
        self.num_samples = num_samples;
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
    fn get(&self, filename: Option<String>) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo_main = "warp-ai/wuerstchen";
                let repo_prior = "warp-ai/wuerstchen-prior";
                let (repo, path) = match self {
                    Self::Tokenizer => (repo_main, "tokenizer/tokenizer.json"),
                    Self::PriorTokenizer => (repo_prior, "tokenizer/tokenizer.json"),
                    Self::Clip => (repo_main, "text_encoder/model.safetensors"),
                    Self::PriorClip => (repo_prior, "text_encoder/model.safetensors"),
                    Self::Decoder => (repo_main, "decoder/diffusion_pytorch_model.safetensors"),
                    Self::VqGan => (repo_main, "vqgan/diffusion_pytorch_model.safetensors"),
                    Self::Prior => (repo_prior, "prior/diffusion_pytorch_model.safetensors"),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

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
    pub fn builder() -> WuerstchenBuilder {
        WuerstchenBuilder::default()
    }

    fn new(settings: WuerstchenBuilder) -> Result<Self> {
        let WuerstchenBuilder {
            cpu,
            use_flash_attn,
            decoder_weights,
            clip_weights,
            prior_clip_weights,
            prior_weights,
            vqgan_weights,
            tokenizer,
            prior_tokenizer,
        } = settings;

        let prior_tokenizer_path = ModelFile::PriorTokenizer.get(prior_tokenizer)?;
        let prior_tokenizer = Tokenizer::from_file(prior_tokenizer_path).map_err(E::msg)?;
        let tokenizer_path = ModelFile::Tokenizer.get(tokenizer)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let device = device(cpu)?;

        let clip_weights = ModelFile::Clip.get(clip_weights)?;
        let clip_config = stable_diffusion::clip::Config::wuerstchen();
        let clip = stable_diffusion::build_clip_transformer(
            &clip_config,
            &clip_weights,
            &device,
            DType::F32,
        )?;
        let prior_clip_weights = ModelFile::PriorClip.get(prior_clip_weights)?;
        let prior_clip_config = stable_diffusion::clip::Config::wuerstchen_prior();
        let prior_clip = stable_diffusion::build_clip_transformer(
            &prior_clip_config,
            &prior_clip_weights,
            &device,
            DType::F32,
        )?;

        let decoder = {
            let file = ModelFile::Decoder.get(decoder_weights)?;
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
            let file = ModelFile::Prior.get(prior_weights)?;
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
            let file = ModelFile::VqGan.get(vqgan_weights)?;
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

    pub fn run(
        &self,
        settings: InferenceSettings,
    ) -> Result<Vec<ImageBuffer<image::Rgb<u8>, Vec<u8>>>> {
        let height = settings.height;
        let width = settings.width;

        let prior_text_embeddings = {
            self.encode_prompt(
                &settings.prompt,
                Some(&settings.uncond_prompt),
                &self.prior_tokenizer,
                &self.prior_clip,
                &self.prior_clip_config,
            )
            .unwrap()
        };

        let text_embeddings = {
            self.encode_prompt(
                &settings.prompt,
                None,
                &self.tokenizer,
                &self.clip,
                &self.clip_config,
            )
            .unwrap()
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
            )
            .unwrap();

            let prior_scheduler =
                wuerstchen::ddpm::DDPMWScheduler::new(60, Default::default()).unwrap();
            let timesteps = prior_scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            for &t in timesteps {
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0).unwrap();
                let ratio = (Tensor::ones(2, DType::F32, &self.device).unwrap() * t).unwrap();
                let noise_pred = self
                    .prior
                    .forward(&latent_model_input, &ratio, &prior_text_embeddings)
                    .unwrap();
                let noise_pred = noise_pred.chunk(2, 0).unwrap();
                let (noise_pred_text, noise_pred_uncond) = (&noise_pred[0], &noise_pred[1]);
                let noise_pred = (noise_pred_uncond
                    + ((noise_pred_text - noise_pred_uncond).unwrap() * PRIOR_GUIDANCE_SCALE)
                        .unwrap())
                .unwrap();
                latents = prior_scheduler.step(&noise_pred, t, &latents).unwrap();
            }
            ((latents * 42.).unwrap() - 1.).unwrap()
        };

        let mut images = Vec::new();
        for _ in 0..settings.num_samples {
            // https://huggingface.co/warp-ai/wuerstchen/blob/main/model_index.json
            let latent_height =
                (image_embeddings.dim(2).unwrap() as f64 * LATENT_DIM_SCALE) as usize;
            let latent_width =
                (image_embeddings.dim(3).unwrap() as f64 * LATENT_DIM_SCALE) as usize;

            let mut latents = Tensor::randn(
                0f32,
                1f32,
                (b_size, DECODER_CIN, latent_height, latent_width),
                &self.device,
            )
            .unwrap();

            let scheduler = wuerstchen::ddpm::DDPMWScheduler::new(12, Default::default()).unwrap();
            let timesteps = scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            for &t in timesteps {
                let ratio = (Tensor::ones(1, DType::F32, &self.device).unwrap() * t).unwrap();
                let noise_pred = self
                    .decoder
                    .forward(&latents, &ratio, &image_embeddings, Some(&text_embeddings))
                    .unwrap();
                latents = scheduler.step(&noise_pred, t, &latents).unwrap();
            }
            let img_tensor = self.vqgan.decode(&(&latents * 0.3764).unwrap()).unwrap();
            // TODO: Add the clamping between 0 and 1.
            let img_tensor = (img_tensor * 255.)
                .unwrap()
                .to_dtype(DType::U8)
                .unwrap()
                .i(0)
                .unwrap();
            let (channel, height, width) = img_tensor.dims3().unwrap();
            if channel != 3 {
                anyhow::bail!("image must have 3 channels");
            }
            let img = img_tensor
                .permute((1, 2, 0))
                .unwrap()
                .flatten_all()
                .unwrap();
            let pixels = img.to_vec1::<u8>().unwrap();
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
