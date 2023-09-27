#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod model;
mod source;

use anyhow::Error as E;

use candle_transformers::models::mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_core::Device;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::PhiInner;
use tokenizers::Tokenizer;

enum Task {
    Kill,
    Infer {
        settings: InferenceSettings,
        sender: tokio::sync::mpsc::UnboundedSender<String>,
    },
}

pub struct Phi {
    task_sender: tokio::sync::mpsc::UnboundedSender<Task>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Phi {
    fn drop(&mut self) {
        self.task_sender.send(Task::Kill).unwrap();
        self.thread_handle.take().unwrap().join().unwrap();
    }
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
    fn new(model: QMixFormer, tokenizer: Tokenizer, device: Device) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();

        let thread_handle = std::thread::spawn(move || {
            let mut inner = PhiInner::new(model, tokenizer, device);
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    while let Some(task) = task_receiver.recv().await {
                        match task {
                            Task::Kill => break,
                            Task::Infer { settings, sender } => {
                                inner._infer(&settings, sender).await.unwrap();
                            }
                        }
                    }
                })
        });
        Self {
            task_sender,
            thread_handle: Some(thread_handle),
        }
    }

    pub fn run(
        &mut self,
        settings: InferenceSettings,
    ) -> anyhow::Result<tokio::sync::mpsc::UnboundedReceiver<String>> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        self.task_sender
            .send(Task::Infer { settings, sender })
            .unwrap();
        Ok(receiver)
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

        Ok(Phi::new(model, tokenizer, device))
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
pub struct InferenceSettings {
    prompt: String,

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

impl InferenceSettings {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
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

#[tokio::test]
async fn generate() -> anyhow::Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let mut phi = Phi::default();

    phi.run(InferenceSettings::new("The quick brown fox "))?;

    Ok(())
}
