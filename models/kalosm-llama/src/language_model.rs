use std::future::Future;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::model::LlamaModelError;
pub use crate::Llama;
use crate::{InferenceSettings, LlamaSession, LlamaSourceError, Task};
use crate::{LlamaBuilder, LlamaModel};
use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::{Model, ModelBuilder};
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

impl<S: Sampler, Constraints: Parser> Model<S, Constraints> for Llama {
    type Session = LlamaSession;
    type Error = LlamaModelError;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Ok(LlamaSession::new(&self.config))
    }

    fn stream_text_with_callback(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: S,
        on_token: impl FnMut(String) -> Result<(), Self::Error>,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        async { Ok(()) }
    }

    fn stream_text_with_callback_and_parser(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error>,
    ) -> impl std::future::Future<Output = Result<Constraints::Output, Self::Error>> + Send {
        async { todo!() }
    }
}
