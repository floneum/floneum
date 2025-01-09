use crate::model::LlamaModelError;
use crate::structured::generate_structured;
pub use crate::Llama;
use crate::LlamaBuilder;
use crate::{
    InferenceSettings, LlamaSession, LlamaSourceError, StructuredGenerationTask, Task,
    UnstructuredGenerationTask,
};
use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::{
    CreateTextCompletionSession, ModelBuilder, StructuredTextCompletionModel, TextCompletionModel,
};
use kalosm_sample::{CreateParserState, Parser};
use llm_samplers::types::Sampler;

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
        !self.source.model.downloaded()
            || !self
                .source
                .tokenizer
                .as_ref()
                .filter(|t| t.downloaded())
                .is_some()
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
    async fn stream_text_with_callback(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: S,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> Result<(), Self::Error> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
        let on_token = Box::new(on_token);
        self.task_sender
            .send(Task::UnstructuredGeneration(UnstructuredGenerationTask {
                settings: InferenceSettings::new(text.to_string(), session.clone(), sampler),
                on_token,
                finished: tx,
            }))
            .map_err(|_| LlamaModelError::ModelStopped)?;

        rx.await.map_err(|_| LlamaModelError::ModelStopped)??;

        Ok(())
    }
}

impl<S, Constraints> StructuredTextCompletionModel<Constraints, S> for Llama
where
    <Constraints as Parser>::Output: Send,
    Constraints: CreateParserState + Send + 'static,
    S: Sampler + 'static,
{
    fn stream_text_with_callback_and_parser(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl std::future::Future<Output = Result<Constraints::Output, Self::Error>> + Send {
        let text = text.to_string();
        let mut session = session.clone();
        async {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
            let on_token = Box::new(on_token);
            self.task_sender
                .send(Task::StructuredGeneration(StructuredGenerationTask {
                    runner: Box::new(move |model| {
                        let parser_state = parser.create_parser_state();
                        let result = generate_structured(
                            text,
                            model,
                            &mut session,
                            parser,
                            parser_state,
                            sampler,
                            on_token,
                            Some(64),
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
