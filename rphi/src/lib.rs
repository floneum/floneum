#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod source;

use anyhow::{Error as E, Result};

use candle_transformers::models::mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct Phi {
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
}

impl Default for Phi {
    fn default() -> Self {
        Phi::builder().build().unwrap()
    }
}

impl Phi {
    pub fn builder() -> PhiBuilder {
        PhiBuilder::default()
    }

    #[allow(clippy::too_many_arguments)]
    fn new(model: QMixFormer, tokenizer: Tokenizer, device: &Device) -> Self {
        Self {
            model,
            tokenizer,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, inference_settings: &InferenceSettings) -> Result<()> {
        use std::io::Write;

        let InferenceSettings {
            prompt,
            temperature,
            top_p,
            seed,
            sample_len,
            repeat_penalty,
            repeat_last_n,
            ..
        } = inference_settings;

        let mut logits_processor = LogitsProcessor::new(*seed, *temperature, *top_p);
        let mut tokens = self
            .tokenizer
            .encode(*prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..*sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if *repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(*repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    *repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{token}");
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{sample_len} tokens generated ({:.2} token/s)",
            *sample_len as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Default)]
pub struct PhiBuilder {
    /// Run on CPU rather than on GPU.
    cpu: bool,

    source: source::PhiSource,
}

impl PhiBuilder {
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn with_source(mut self, source: source::PhiSource) -> Self {
        self.source = source;
        self
    }

    pub fn build(self) -> anyhow::Result<Phi> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            self.source.model_id,
            RepoType::Model,
            self.source.revision,
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filename = api
            .model("lmz/candle-quantized-phi".to_string())
            .get("model-q4k.gguf")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let config = Config::v1_5();
        let (model, device) = {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&filename)?;
            let model = QMixFormer::new(&config, vb)?;

            (model, device(self.cpu)?)
        };

        Ok(Phi::new(model, tokenizer, &device))
    }
}

pub fn device(cpu: bool) -> anyhow::Result<Device> {
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

#[derive(Debug)]
pub struct InferenceSettings<'a> {
    prompt: &'a str,

    /// The temperature used to generate samples.
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    seed: u64,

    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

impl<'a> InferenceSettings<'a> {
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            temperature: None,
            top_p: None,
            seed: rand::random(),
            sample_len: 100,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_sample_len(mut self, sample_len: usize) -> Self {
        self.sample_len = sample_len;
        self
    }

    pub fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = repeat_penalty;
        self
    }

    pub fn with_repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = repeat_last_n;
        self
    }
}

#[test]
fn generate() -> Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let mut phi = Phi::default();

    phi.run(&InferenceSettings::new("The quick brown fox "))?;

    Ok(())
}
