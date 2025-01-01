use std::future::Future;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::model::LlamaModelError;
pub use crate::Llama;
use crate::{InferenceSettings, LlamaSession, LlamaSourceError, Task};
use crate::{LlamaBuilder, LlamaModel};
use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::{
    ModelBuilder, ModelSession, StructuredTextCompletionModel, TextCompletionModel,
};
use kalosm_sample::Parser;
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokenizers::Tokenizer;

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

impl ModelSession for Llama {
    type Session = LlamaSession;
    type Error = LlamaModelError;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Ok(LlamaSession::new(&self.config))
    }
}

impl<S: Sampler + 'static> TextCompletionModel<S> for Llama {
    fn stream_text_with_callback(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: S,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        async {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
            let on_token = Box::new(on_token);
            self.task_sender
                .send(Task {
                    settings: InferenceSettings::new(text.to_string(), session.clone(), sampler),
                    on_token,
                    finished: tx,
                })
                .map_err(|_| LlamaModelError::ModelStopped)?;

            rx.await.map_err(|_| LlamaModelError::ModelStopped)??;

            Ok(())
        }
    }
}

impl<S: Sampler + 'static, Constraints: Parser> StructuredTextCompletionModel<S, Constraints>
    for Llama
{
    fn stream_text_with_callback_and_parser(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl std::future::Future<Output = Result<Constraints::Output, Self::Error>> + Send {
        async { todo!() }
    }
}
