use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::model::LlamaModelError;
pub use crate::Llama;
use crate::{InferenceSettings, LlamaSourceError, Task};
use crate::{LlamaBuilder, LlamaModel};
use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::ChatMarkers;
use kalosm_language_model::{GenerationParameters, Model, ModelBuilder};
use kalosm_streams::text_stream::ChannelTextStream;
use tokenizers::Tokenizer;

#[async_trait::async_trait]
impl ModelBuilder for LlamaBuilder {
    type Model = Llama;
    type Error = LlamaSourceError;

    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        !self.source.model.downloaded() || !self.source.tokenizer.downloaded()
    }
}

#[async_trait::async_trait]
impl Model for Llama {
    type TextStream = ChannelTextStream;
    type SyncModel = LlamaModel;
    type Error = LlamaModelError;

    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.get_tokenizer()
    }

    fn run_sync_raw(
        &self,
        f: Box<
            dyn for<'a> FnOnce(
                    &'a mut Self::SyncModel,
                )
                    -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
                + Send,
        >,
    ) -> Result<(), Self::Error> {
        _ = self.task_sender.send(Task::RunSync { callback: f });
        Ok(())
    }

    async fn stream_text_inner(
        &self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> Result<Self::TextStream, Self::Error> {
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
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn llm_samplers::prelude::Sampler>>,
    ) -> Result<Self::TextStream, Self::Error> {
        let max_length = max_tokens.unwrap_or(64);
        self.run(
            InferenceSettings::new(prompt)
                .with_sample_len(max_length as usize)
                .with_stop_on(stop_on.map(|s| s.to_string())),
            sampler,
        )
        .map(Into::into)
    }

    fn chat_markers(&self) -> Option<ChatMarkers> {
        self.chat_markers.deref().clone()
    }
}
