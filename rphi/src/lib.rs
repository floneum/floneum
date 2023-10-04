#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod model;
mod source;

use anyhow::Error as E;

use candle_core::Device;
use candle_transformers::models::mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use floneumin_sample::FasterHuggingFaceTokenizer;
use hf_hub::{api::sync::Api, Repo, RepoType};
use llm_samplers::prelude::Sampler;
use model::PhiInner;
use std::sync::Arc;
use std::sync::Mutex;
use tokenizers::Tokenizer;

enum Task {
    Kill,
    Infer {
        settings: InferenceSettings,
        sender: tokio::sync::mpsc::UnboundedSender<String>,
        sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    },
}

pub struct Phi {
    task_sender: tokio::sync::mpsc::UnboundedSender<Task>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    tokenizer: Arc<FasterHuggingFaceTokenizer>,
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

    pub fn downloaded() -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn new(model: QMixFormer, tokenizer: Tokenizer, device: Device) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();
        let arc_tokenizer = Arc::new(FasterHuggingFaceTokenizer::new(tokenizer.clone()));

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
                            Task::Infer {
                                settings,
                                sender,
                                sampler,
                            } => {
                                if let Err(err) = inner._infer(settings, sampler, sender) {
                                    tracing::trace!("Error in PhiInner::_infer: {}", err);
                                }
                            }
                        }
                    }
                })
        });
        Self {
            task_sender,
            thread_handle: Some(thread_handle),
            tokenizer: arc_tokenizer,
        }
    }

    pub fn get_tokenizer(&self) -> Arc<FasterHuggingFaceTokenizer> {
        self.tokenizer.clone()
    }

    pub fn run(
        &mut self,
        settings: InferenceSettings,
        sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    ) -> anyhow::Result<tokio::sync::mpsc::UnboundedReceiver<String>> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        self.task_sender
            .send(Task::Infer {
                settings,
                sender,
                sampler,
            })
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

    /// The seed to use when generating random samples.
    seed: u64,

    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The token to stop on.
    stop_on: Option<&'static str>,
}

impl InferenceSettings {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            seed: rand::random(),
            sample_len: 100,
            stop_on: None,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_sample_len(mut self, sample_len: usize) -> Self {
        self.sample_len = sample_len;
        self
    }

    pub fn with_stop_on(mut self, stop_on: Option<&'static str>) -> Self {
        self.stop_on = stop_on;
        self
    }
}
