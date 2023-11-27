use std::sync::{Arc, Mutex};

pub use crate::Mistral;
use crate::MistralModel;
use crate::{InferenceSettings, Task};
use kalosm_language_model::{CreateModel, GenerationParameters, Model, VectorSpace};
use kalosm_streams::ChannelTextStream;

#[async_trait::async_trait]
impl CreateModel for Mistral {
    async fn start() -> Self {
        Mistral::default()
    }

    fn requires_download() -> bool {
        !Mistral::downloaded()
    }
}

#[async_trait::async_trait]
impl Model for Mistral {
    type TextStream = ChannelTextStream<String>;
    type SyncModel = MistralModel;

    fn tokenizer(&self) -> Arc<dyn kalosm_sample::Tokenizer + Send + Sync> {
        self.get_tokenizer() as Arc<dyn kalosm_sample::Tokenizer + Send + Sync>
    }

    fn run_sync_raw(
        &mut self,
        f: Box<
            dyn for<'a> FnOnce(
                    &'a mut Self::SyncModel,
                )
                    -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
                + Send,
        >,
    ) -> anyhow::Result<()> {
        match self.task_sender.send(Task::RunSync { callback: f }) {
            Ok(_) => Ok(()),
            Err(_) => Err(anyhow::anyhow!("Failed to send task to Phi thread")),
        }
    }

    async fn stream_text_inner(
        &mut self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> anyhow::Result<Self::TextStream> {
        let max_length = generation_parameters.max_length();
        self.run(
            InferenceSettings::new(prompt)
                .with_sample_len(max_length as usize)
                .with_stop_on(generation_parameters.stop_on().map(|s| s.to_string())),
            Arc::new(Mutex::new(generation_parameters.sampler())),
        )
        .map(Into::into)
    }

    async fn stream_text_with_sampler(
        &mut self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn llm_samplers::prelude::Sampler>>,
    ) -> anyhow::Result<Self::TextStream> {
        let max_length = max_tokens.unwrap_or(64);
        self.run(
            InferenceSettings::new(prompt)
                .with_sample_len(max_length as usize)
                .with_stop_on(stop_on.map(|s| s.to_string())),
            sampler,
        )
        .map(Into::into)
    }
}

pub struct MistralSpace;

impl VectorSpace for MistralSpace {}
