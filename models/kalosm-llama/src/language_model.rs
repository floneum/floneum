use kalosm_language_model::{
    CreateDefaultChatConstraintsForType, CreateDefaultCompletionConstraintsForType,
    CreateTextCompletionSession, GenerationParameters, ModelBuilder, StructuredTextCompletionModel,
    TextCompletionModel,
};
use kalosm_model_types::ModelLoadingProgress;
use kalosm_sample::{ArcParser, CreateParserState, Parse, Parser, ParserExt};
use llm_samplers::types::Sampler;
use std::any::Any;
use std::future::Future;
use std::sync::{Arc, RwLock};

use crate::model::LlamaModelError;
use crate::structured::{generate_structured, EvaluationTrie};
pub use crate::Llama;
use crate::{
    InferenceSettings, LlamaSession, LlamaSourceError, StructuredGenerationTask, Task,
    UnstructuredGenerationTask,
};
use crate::{LlamaBuilder, LlamaModel};

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
        !cache.exists(&self.source.model)
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
    fn stream_text_with_callback<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: &str,
        sampler: S,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        let text = text.to_string();
        async move {
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
            self.task_sender
                .send(Task::UnstructuredGeneration(UnstructuredGenerationTask {
                    settings: InferenceSettings::new(
                        text,
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

impl LlamaModel {
    pub fn new_session(&self) -> LlamaSession {
        let config = &self.model.config;
        LlamaSession::new(config)
    }

    pub fn generate_structured_with_trie<'a, S, Constraints>(
        &'a mut self,
        session: &'a mut LlamaSession,
        text: &str,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), LlamaModelError> + Send + Sync + 'static,
        trie: &mut EvaluationTrie,
        fast_case: bool,
        lazy: bool,
    ) -> Result<Constraints::Output, LlamaModelError>
    where
        <Constraints as Parser>::Output: Send,
        Constraints: CreateParserState + Send + 'static,
        S: Sampler + 'static,
    {
        let parser_state = parser.create_parser_state();
        let cache = session.cache.read().unwrap().clone();
        let mut session = LlamaSession {
            cache: Arc::new(RwLock::new(cache)),
        };
        let seed = match (&sampler as &dyn Any).downcast_ref::<GenerationParameters>() {
            Some(sampler) => sampler.seed(),
            None => None,
        };
        let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));

        generate_structured(
            text,
            self,
            &mut session,
            &parser,
            parser_state,
            sampler,
            on_token,
            None,
            seed,
            trie,
            fast_case,
            lazy,
        )
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
        text: &str,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + Send + 'a {
        let text = text.to_string();
        let mut session = session.clone();
        async {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let on_token = Box::new(on_token);
            self.task_sender
                .send(Task::StructuredGeneration(StructuredGenerationTask {
                    runner: Box::new(move |model| {
                        let mut trie = EvaluationTrie::new();
                        _ = tx.send(model.generate_structured_with_trie(
                            &mut session,
                            &text,
                            sampler,
                            parser,
                            on_token,
                            &mut trie,
                            true,
                            true,
                        ));
                    }),
                }))
                .map_err(|_| LlamaModelError::ModelStopped)?;

            let result = rx.await.map_err(|_| LlamaModelError::ModelStopped)??;

            Ok(result)
        }
    }
}
