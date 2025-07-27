use kalosm_language_model::{
    ContentChunk, CreateDefaultChatConstraintsForType, CreateDefaultCompletionConstraintsForType,
    CreateTextCompletionSession, GenerationParameters, MessageContent, ModelBuilder,
    StructuredTextCompletionModel, TextCompletionModel,
};
use kalosm_model_types::ModelLoadingProgress;
use kalosm_sample::{ArcParser, CreateParserState, Parse, Parser, ParserExt};
use llm_samplers::types::Sampler;
use std::any::Any;
use std::future::Future;

use crate::model::LlamaModelError;
use crate::structured::generate_structured;
pub use crate::Llama;
use crate::LlamaBuilder;
use crate::{
    InferenceSettings, LlamaSession, LlamaSourceError, StructuredGenerationTask, Task,
    UnstructuredGenerationTask,
};

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
        let cache = &self.source.cache;
        !self.source.model.iter().all(|m| cache.exists(m))
            || self
                .source
                .tokenizer
                .as_ref()
                .filter(|t| cache.exists(t))
                .is_none()
    }
}

impl CreateTextCompletionSession for Llama {
    type Session = LlamaSession;
    type Error = LlamaModelError;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Ok(LlamaSession::new(&self.config))
    }
}

impl<S: Sampler + 'static> TextCompletionModel<S> for Llama {
    async fn stream_text_with_callback<'a>(
        &'a self,
        session: &'a mut Self::Session,
        msg: MessageContent,
        sampler: S,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> Result<(), Self::Error> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (max_tokens, stop_on, seed) =
            match (&sampler as &dyn Any).downcast_ref::<GenerationParameters>() {
                Some(sampler) => (
                    sampler.max_length(),
                    sampler.stop_on().map(|s| s.to_string()),
                    sampler.seed(),
                ),
                None => (u32::MAX, None, None),
            };
        let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
        let on_token = Box::new(on_token);
        let text = msg.text();
        let msg = msg.resolve_media_sources().await?;
        let mut images = Vec::new();
        for chunk in msg.chunks() {
            if let ContentChunk::Media(media) = chunk {
                if let Some(bytes) = &media.source().as_bytes() {
                    // Decode the image from the bytes
                    images.push((image::load_from_memory(bytes)?, media.hints().clone()))
                }
            }
        }
        self.task_sender
            .send(Task::UnstructuredGeneration(UnstructuredGenerationTask {
                settings: InferenceSettings::new(
                    text,
                    images,
                    session.clone(),
                    sampler,
                    max_tokens,
                    stop_on,
                    seed,
                ),
                on_token,
                finished: tx,
            }))
            .map_err(|_| LlamaModelError::ModelStopped)?;

        rx.await.map_err(|_| LlamaModelError::ModelStopped)??;

        Ok(())
    }
}

impl<T: Parse + 'static> CreateDefaultChatConstraintsForType<T> for Llama {
    type DefaultConstraints = ArcParser<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        T::new_parser().boxed()
    }
}

impl<T: Parse + 'static> CreateDefaultCompletionConstraintsForType<T> for Llama {
    type DefaultConstraints = ArcParser<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        T::new_parser().boxed()
    }
}

impl<S, Constraints> StructuredTextCompletionModel<Constraints, S> for Llama
where
    <Constraints as Parser>::Output: Send,
    Constraints: CreateParserState + Send + 'static,
    S: Sampler + 'static,
{
    fn stream_text_with_callback_and_parser<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + Send + 'a {
        let mut session = session.clone();
        async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let seed = match (&sampler as &dyn Any).downcast_ref::<GenerationParameters>() {
                Some(sampler) => sampler.seed(),
                None => None,
            };
            let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
            let on_token = Box::new(on_token);
            let resolved_message = text.resolve_media_sources().await?;
            self.task_sender
                .send(Task::StructuredGeneration(StructuredGenerationTask {
                    runner: Box::new(move |model| {
                        let parser_state = parser.create_parser_state();
                        let result = generate_structured(
                            resolved_message,
                            model,
                            &mut session,
                            parser,
                            parser_state,
                            sampler,
                            on_token,
                            Some(64),
                            seed,
                        );
                        _ = tx.send(result);
                    }),
                }))
                .map_err(|_| LlamaModelError::ModelStopped)?;

            let result = rx.await.map_err(|_| LlamaModelError::ModelStopped)??;

            Ok(result)
        }
    }
}
