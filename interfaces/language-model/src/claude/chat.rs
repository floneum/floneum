use super::{AnthropicCompatibleClient, NoAnthropicAPIKeyError};
use crate::{
    ChatMessage, ChatModel, ChatSession, ContentChunk, CreateChatSession, GenerationParameters,
    ModelBuilder,
};
use futures_util::StreamExt;
use kalosm_model_types::ModelLoadingProgress;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use std::{future::Future, sync::Arc};
use thiserror::Error;

#[derive(Debug)]
struct AnthropicCompatibleChatModelInner {
    model: String,
    max_tokens: u32,
    client: AnthropicCompatibleClient,
}

/// An chat model that uses Anthropic's API for the a remote chat model.
#[derive(Debug, Clone)]
pub struct AnthropicCompatibleChatModel {
    inner: Arc<AnthropicCompatibleChatModelInner>,
}

impl AnthropicCompatibleChatModel {
    /// Create a new builder for the Anthropic compatible chat model.
    pub fn builder() -> AnthropicCompatibleChatModelBuilder<false> {
        AnthropicCompatibleChatModelBuilder::new()
    }
}

/// A builder for an Anthropic compatible chat model.
#[derive(Debug, Default)]
pub struct AnthropicCompatibleChatModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    max_tokens: u32,
    client: AnthropicCompatibleClient,
}

impl AnthropicCompatibleChatModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            max_tokens: 8192,
            client: Default::default(),
        }
    }
}

impl<const WITH_NAME: bool> AnthropicCompatibleChatModelBuilder<WITH_NAME> {
    /// Set the name of the model to use.
    pub fn with_model(self, model: impl ToString) -> AnthropicCompatibleChatModelBuilder<true> {
        AnthropicCompatibleChatModelBuilder {
            model: Some(model.to_string()),
            max_tokens: self.max_tokens,
            client: self.client,
        }
    }

    /// Set the default max tokens to use when generating text.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the model to `claude-3-5-sonnet-20241022`
    pub fn with_claude_3_5_sonnet(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-5-sonnet-20241022")
    }

    /// Set the model to `claude-3-5-haiku-20241022`
    pub fn with_claude_3_5_haiku(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-5-haiku-20241022")
    }

    /// Set the model to `claude-3-opus-20240229`
    pub fn with_claude_3_opus(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-opus-20240229")
            .with_max_tokens(4096)
    }

    /// Set the model to `claude-3-sonnet-20240229`
    pub fn with_claude_3_sonnet(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-sonnet-20240229")
            .with_max_tokens(4096)
    }

    /// Set the model to `claude-3-haiku-20240307`
    pub fn with_claude_3_haiku(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-haiku-20240307")
            .with_max_tokens(4096)
    }

    /// Set the client used to make requests to the Anthropic API.
    pub fn with_client(mut self, client: AnthropicCompatibleClient) -> Self {
        self.client = client;
        self
    }
}

impl AnthropicCompatibleChatModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> AnthropicCompatibleChatModel {
        AnthropicCompatibleChatModel {
            inner: Arc::new(AnthropicCompatibleChatModelInner {
                model: self.model.unwrap(),
                max_tokens: self.max_tokens,
                client: self.client,
            }),
        }
    }
}

impl ModelBuilder for AnthropicCompatibleChatModelBuilder<true> {
    type Model = AnthropicCompatibleChatModel;
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

/// An error that can occur when running a [`AnthropicCompatibleChatModel`].
#[derive(Error, Debug)]
pub enum AnthropicCompatibleChatModelError {
    /// An error occurred while resolving the API key.
    #[error("Error resolving API key: {0}")]
    APIKeyError(#[from] NoAnthropicAPIKeyError),
    /// An error occurred while making a request to the Anthropic API.
    #[error("Error making request: {0}")]
    ReqwestError(#[from] reqwest::Error),
    /// An error occurred while receiving server side events from the Anthropic API.
    #[error("Error receiving server side events: {0}")]
    EventSourceError(#[from] reqwest_eventsource::Error),
    /// Failed to deserialize Anthropic API response.
    #[error("Failed to deserialize Anthropic API response: {0}")]
    DeserializeError(#[from] serde_json::Error),
    /// An error occurred while streaming the response from the Anthropic API.
    #[error("Error streaming response from Anthropic API: {0}")]
    StreamError(#[from] AnthropicCompatibleChatResponseError),
}

/// A chat session for the Anthropic compatible chat model.
#[derive(Serialize, Deserialize, Clone)]
pub struct AnthropicCompatibleChatSession {
    messages: Vec<crate::ChatMessage>,
}

impl AnthropicCompatibleChatSession {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

impl ChatSession for AnthropicCompatibleChatSession {
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

    fn history(&self) -> Vec<crate::ChatMessage> {
        self.messages.clone()
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

impl CreateChatSession for AnthropicCompatibleChatModel {
    type ChatSession = AnthropicCompatibleChatSession;
    type Error = AnthropicCompatibleChatModelError;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        Ok(AnthropicCompatibleChatSession::new())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicCompatibleChatResponse {
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta(AnthropicCompatibleChatResponseContentBlockDelta),
    #[serde(rename = "content_block_stop")]
    ContentBlockStop,
    #[serde(rename = "error")]
    Error(AnthropicCompatibleChatResponseError),
    #[serde(other)]
    Unknown,
}

/// An error that can occur when receiving a stream from the Anthropic API.
#[derive(Serialize, Deserialize, Error, Debug)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AnthropicCompatibleChatResponseError {
    /// The request was invalid.
    #[serde(rename = "invalid_request_error")]
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        /// The error message.
        message: String,
    },
    /// Authentication failed.
    #[serde(rename = "authentication_error")]
    #[error("Authentication error: {message}")]
    AuthenticationError {
        /// The error message.
        message: String,
    },
    /// An permission error occurred.
    #[serde(rename = "permission_error")]
    #[error("Permission error: {message}")]
    PermissionError {
        /// The error message.
        message: String,
    },
    /// The resource was not found.
    #[serde(rename = "not_found_error")]
    #[error("Not found error: {message}")]
    NotFoundError {
        /// The error message.
        message: String,
    },
    /// The request was too large.
    #[serde(rename = "request_too_large")]
    #[error("Request too large: {message}")]
    RequestTooLarge {
        /// The error message.
        message: String,
    },
    /// The rate limit was exceeded.
    #[serde(rename = "rate_limit_error")]
    #[error("Rate limit error: {message}")]
    RateLimitError {
        /// The error message.
        message: String,
    },
    /// An API error occurred.
    #[serde(rename = "api_error")]
    #[error("API error: {message}")]
    ApiError {
        /// The error message.
        message: String,
    },
    /// The server is overloaded.
    #[serde(rename = "overloaded_error")]
    #[error("Overloaded error: {message}")]
    OverloadedError {
        /// The error message.
        message: String,
    },
    /// An unknown error occurred.
    #[serde(other)]
    #[error("Unknown error")]
    Unknown,
}

#[derive(Serialize, Deserialize)]
struct AnthropicCompatibleChatResponseContentBlockDelta {
    index: u32,
    delta: AnthropicCompatibleChatResponseContentBlockDeltaMessage,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicCompatibleChatResponseContentBlockDeltaMessage {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(other)]
    Unknown,
}

impl ChatModel<GenerationParameters> for AnthropicCompatibleChatModel {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: GenerationParameters,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        let mut system_prompt = None;
        let messages: Vec<_> = messages
            .iter()
            .filter(|message| {
                if let crate::MessageType::SystemPrompt = message.role() {
                    system_prompt = message.content().as_str().map(ToString::to_string);
                    false
                } else {
                    true
                }
            })
            .collect();
        let messages = format_messages(&messages);
        let myself = &*self.inner;
        let mut json = serde_json::json!({
            "model": myself.model,
            "messages": messages,
            "stream": true,
            "top_p": sampler.top_p,
            "top_k": sampler.top_k,
            "temperature": sampler.temperature,
            "max_tokens": sampler.max_length.min(myself.max_tokens),
        });

        async move {
            let api_key = myself.client.resolve_api_key()?;
            if let Some(stop_on) = sampler.stop_on.as_ref() {
                json["stop"] = vec![stop_on.clone()].into();
            }
            if let Some(system) = system_prompt {
                json["system"] = system.into();
            }
            let mut event_source = myself
                .client
                .reqwest_client
                .post(format!("{}/messages", myself.client.base_url()))
                .header("Content-Type", "application/json")
                .header("x-api-key", api_key)
                .header("anthropic-version", myself.client.version())
                .json(&json)
                .eventsource()
                .unwrap();

            let mut new_message_text = String::new();

            while let Some(event) = event_source.next().await {
                match event? {
                    Event::Open => {}
                    Event::Message(message) => {
                        let data =
                            serde_json::from_str::<AnthropicCompatibleChatResponse>(&message.data)?;
                        match data {
                            AnthropicCompatibleChatResponse::ContentBlockDelta(
                                anthropic_compatible_chat_response_content_block_delta,
                            ) => {
                                match anthropic_compatible_chat_response_content_block_delta.delta {
                                AnthropicCompatibleChatResponseContentBlockDeltaMessage::TextDelta { text } => {
                                        new_message_text += &text;
                                        on_token(text)?;
                                },
                                AnthropicCompatibleChatResponseContentBlockDeltaMessage::Unknown => tracing::trace!("Unknown delta from Anthropic API: {:?}", message.data),
                            }
                            }
                            AnthropicCompatibleChatResponse::ContentBlockStop => {
                                break;
                            }
                            AnthropicCompatibleChatResponse::Error(
                                anthropic_compatible_chat_response_error,
                            ) => {
                                return Err(AnthropicCompatibleChatModelError::StreamError(
                                    anthropic_compatible_chat_response_error,
                                ))
                            }
                            AnthropicCompatibleChatResponse::Unknown => tracing::trace!(
                                "Unknown response from Anthropic API: {:?}",
                                message.data
                            ),
                        }
                    }
                }
            }

            let new_message =
                crate::ChatMessage::new(crate::MessageType::UserMessage, new_message_text);

            session.messages.push(new_message);

            Ok(())
        }
    }
}

fn format_messages(messages: &[&crate::ChatMessage]) -> serde_json::Value {
    messages
        .iter()
        .map(|m| {
            let content = m.content();
            let content: serde_json::Value = if let Some(string) = content.as_str() {
                string.into()
            } else {
                content
                    .chunks()
                    .iter()
                    .map(|chunk| match chunk {
                        ContentChunk::Text(text) => {
                            serde_json::json!({
                                "type": "text",
                                "text": text
                            })
                        }
                        ContentChunk::Media(image) => {
                            serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": image.as_url(),
                                }
                            })
                        }
                    })
                    .collect::<Vec<_>>()
                    .into()
            };

            serde_json::json!({
                "role": m.role(),
                "content": content,
            })
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::{
        AnthropicCompatibleChatModelBuilder, ChatModel, CreateChatSession, GenerationParameters,
    };

    #[tokio::test]
    async fn test_claude_3_5_haiku() {
        let model = AnthropicCompatibleChatModelBuilder::new()
            .with_claude_3_5_haiku()
            .build();

        let mut session = model.new_chat_session().unwrap();

        let messages = vec![
            crate::ChatMessage::new(
                crate::MessageType::SystemPrompt,
                "Respond like a pirate.".to_string(),
            ),
            crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                (
                    crate::MediaSource::url(
                        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    ),
                    "Describe this image".to_string(),
                ),
            ),
        ];
        let all_text = Arc::new(RwLock::new(String::new()));
        model
            .add_messages_with_callback(&mut session, &messages, GenerationParameters::default(), {
                let all_text = all_text.clone();
                move |token| {
                    let mut all_text = all_text.write().unwrap();
                    all_text.push_str(&token);
                    print!("{token}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    Ok(())
                }
            })
            .await
            .unwrap();

        let all_text = all_text.read().unwrap();
        println!("{all_text}");

        assert!(!all_text.is_empty());
    }
}
