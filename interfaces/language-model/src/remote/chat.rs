use super::{NoAPIKeyError, OpenAICompatibleClient};
use crate::{
    ChatModel, ChatSessionImpl, CreateChatSession, Embedder, Embedding, GenerationParameters,
    ModelBuilder,
};
use futures_util::StreamExt;
use kalosm_common::ModelLoadingProgress;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use std::future::Future;
use thiserror::Error;

/// An embedder that uses OpenAI's API for the a remote embedding model.
#[derive(Debug)]
pub struct OpenAICompatibleChatModel {
    model: String,
    client: OpenAICompatibleClient,
}

/// A builder for an openai compatible embedding model.
#[derive(Debug, Default)]
pub struct OpenAICompatibleChatModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    client: OpenAICompatibleClient,
}

impl OpenAICompatibleChatModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            client: Default::default(),
        }
    }
}

impl<const WITH_NAME: bool> OpenAICompatibleChatModelBuilder<WITH_NAME> {
    /// Set the name of the model to use.
    pub fn with_model(self, model: impl ToString) -> OpenAICompatibleChatModelBuilder<true> {
        OpenAICompatibleChatModelBuilder {
            model: Some(model.to_string()),
            client: self.client,
        }
    }

    /// Set the model to the latest version of gpt 4o
    pub fn with_gpt_4o(self) -> OpenAICompatibleChatModelBuilder<true> {
        self.with_model("gpt-4o")
    }

    /// Set the model to the latest version of chat gpt 4o used in ChatGPT
    pub fn with_chat_gpt_4o(self) -> OpenAICompatibleChatModelBuilder<true> {
        self.with_model("chatgpt-4o-latest")
    }

    /// Set the model to the latest version of gpt 4o mini
    pub fn with_gpt_4o_mini(self) -> OpenAICompatibleChatModelBuilder<true> {
        self.with_model("gpt-4o-mini")
    }

    /// Set the client used to make requests to the OpenAI API.
    pub fn with_client(mut self, client: OpenAICompatibleClient) -> Self {
        self.client = client;
        self
    }
}

impl OpenAICompatibleChatModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> OpenAICompatibleChatModel {
        OpenAICompatibleChatModel {
            model: self.model.unwrap(),
            client: self.client,
        }
    }
}

impl ModelBuilder for OpenAICompatibleChatModelBuilder<true> {
    type Model = OpenAICompatibleChatModel;
    type Error = std::convert::Infallible;

    async fn start_with_loading_handler(
        self,
        _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        Ok(self.build())
    }

    fn requires_download(&self) -> bool {
        false
    }
}

#[derive(Deserialize)]
struct CreateEmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Error, Debug)]
pub enum OpenAICompatibleChatModelError {
    #[error("Error resolving API key: {0}")]
    APIKeyError(#[from] NoAPIKeyError),
    #[error("Error making request: {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("Error receiving server side events: {0}")]
    EventSourceError(#[from] reqwest_eventsource::Error),
    #[error("OpenAI API returned no message choices in the response")]
    NoMessageChoices,
    #[error("Failed to deserialize OpenAI API response: {0}")]
    DeserializeError(#[from] serde_json::Error),
    #[error("Refusal from OpenAI API: {0}")]
    Refusal(String),
    #[error("Function calls are not yet supported in kalosm with the OpenAI API")]
    FunctionCallsNotSupported,
}

#[derive(Serialize, Deserialize)]
pub struct OpenAICompatibleChatSession {
    messages: Vec<crate::ChatHistoryItem>,
}

impl OpenAICompatibleChatSession {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

impl ChatSessionImpl for OpenAICompatibleChatSession {
    type Error = serde_json::Error;

    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error> {
        let json = serde_json::to_vec(self)?;
        into.extend_from_slice(&json);
        Ok(())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        let json = serde_json::from_slice(bytes)?;
        Ok(json)
    }

    fn history(&self) -> Vec<crate::ChatHistoryItem> {
        self.messages.clone()
    }
}

impl CreateChatSession for OpenAICompatibleChatModel {
    type ChatSession = OpenAICompatibleChatSession;
    type Error = OpenAICompatibleChatModelError;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        Ok(OpenAICompatibleChatSession::new())
    }
}

#[derive(Serialize, Deserialize)]
struct OpenAICompatibleChatResponse {
    choices: Vec<OpenAICompatibleChatResponseChoice>,
}

#[derive(Serialize, Deserialize)]
struct OpenAICompatibleChatResponseChoice {
    delta: OpenAICompatibleChatResponseChoiceMessage,
    finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Deserialize)]
enum FinishReason {
    #[serde(rename = "content_filter")]
    ContentFilter,
    #[serde(rename = "function_call")]
    FunctionCall,
    #[serde(rename = "length")]
    MaxTokens,
    #[serde(rename = "stop")]
    Stop,
}

#[derive(Serialize, Deserialize)]
struct OpenAICompatibleChatResponseChoiceMessage {
    content: Option<String>,
    refusal: Option<String>,
}

impl ChatModel<GenerationParameters> for OpenAICompatibleChatModel {
    fn add_messages_with_callback(
        &self,
        session: &mut Self::ChatSession,
        messages: &[crate::ChatHistoryItem],
        sampler: GenerationParameters,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        async move {
            let api_key = self.client.resolve_api_key()?;
            let mut event_source = self
                .client
                .reqwest_client
                .post(format!("{}/chat/completions", self.client.base_url))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&serde_json::json!({
                    "messages": messages,
                    "model": self.model,
                    "stream": true,
                    "top_p": sampler.top_p,
                    "temperature": sampler.temperature,
                    "frequency_penalty": sampler.repetition_penalty,
                }))
                .eventsource()
                .unwrap();

            let mut new_message_text = String::new();

            while let Some(event) = event_source.next().await {
                match event? {
                    Event::Open => {}
                    Event::Message(message) => {
                        let data =
                            serde_json::from_str::<OpenAICompatibleChatResponse>(&message.data)?;
                        let first_choice = data
                            .choices
                            .first()
                            .ok_or(OpenAICompatibleChatModelError::NoMessageChoices)?;
                        if let Some(content) = &first_choice.delta.refusal {
                            return Err(OpenAICompatibleChatModelError::Refusal(content.clone()));
                        }
                        if let Some(refusal) = &first_choice.finish_reason {
                            match refusal {
                                FinishReason::ContentFilter => {
                                    return Err(OpenAICompatibleChatModelError::Refusal(
                                        "ContentFilter".to_string(),
                                    ))
                                }
                                FinishReason::FunctionCall => {
                                    return Err(
                                        OpenAICompatibleChatModelError::FunctionCallsNotSupported,
                                    )
                                }
                                _ => return Ok(()),
                            }
                        }
                        if let Some(content) = &first_choice.delta.content {
                            on_token(content.clone())?;
                            new_message_text += content;
                        }
                    }
                }
            }

            let new_message =
                crate::ChatHistoryItem::new(crate::MessageType::UserMessage, new_message_text);

            session.messages.push(new_message);

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::{
        ChatModel, CreateChatSession, GenerationParameters, OpenAICompatibleChatModelBuilder,
    };

    #[tokio::test]
    async fn test_gpt_4o_mini() {
        let model = OpenAICompatibleChatModelBuilder::new()
            .with_gpt_4o_mini()
            .build();

        let mut session = model.new_chat_session().unwrap();

        let mut messages = Vec::new();
        messages.push(crate::ChatHistoryItem::new(
            crate::MessageType::UserMessage,
            "Hello, world!".to_string(),
        ));
        let all_text = Arc::new(RwLock::new(String::new()));
        model
            .add_messages_with_callback(&mut session, &messages, GenerationParameters::default(), {
                let all_text = all_text.clone();
                move |token| {
                    let mut all_text = all_text.write().unwrap();
                    all_text.push_str(&token);
                    Ok(())
                }
            })
            .await
            .unwrap();

        let all_text = all_text.read().unwrap();

        assert!(!all_text.is_empty());
    }
}
