use super::{NoOpenAIAPIKeyError, OpenAICompatibleClient};
use crate::{
    ChatModel, ChatSession, CreateChatSession, CreateDefaultChatConstraintsForType,
    GenerationParameters, ModelBuilder, ModelConstraints, StructuredChatModel,
};
use futures_util::StreamExt;
use kalosm_model_types::ModelLoadingProgress;
use kalosm_sample::Schema;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{future::Future, sync::Arc};
use thiserror::Error;

#[derive(Debug)]
struct OpenAICompatibleChatModelInner {
    model: String,
    client: OpenAICompatibleClient,
}

/// An chat model that uses OpenAI's API for the a remote chat model.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleChatModel {
    inner: Arc<OpenAICompatibleChatModelInner>,
}

impl OpenAICompatibleChatModel {
    /// Create a new builder for the OpenAI compatible chat model.
    pub fn builder() -> OpenAICompatibleChatModelBuilder<false> {
        OpenAICompatibleChatModelBuilder::new()
    }
}

/// A builder for an openai compatible chat model.
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
            inner: Arc::new(OpenAICompatibleChatModelInner {
                model: self.model.unwrap(),
                client: self.client,
            }),
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

/// An error that can occur when running a [`OpenAICompatibleChatModel`].
#[derive(Error, Debug)]
pub enum OpenAICompatibleChatModelError {
    /// An error occurred while resolving the API key.
    #[error("Error resolving API key: {0}")]
    APIKeyError(#[from] NoOpenAIAPIKeyError),
    /// An error occurred while making a request to the OpenAI API.
    #[error("Error making request: {0}")]
    ReqwestError(#[from] reqwest::Error),
    /// An error occurred while receiving server side events from the OpenAI API.
    #[error("Error receiving server side events: {0}")]
    EventSourceError(#[from] reqwest_eventsource::Error),
    /// OpenAI API returned no message choices in the response.
    #[error("OpenAI API returned no message choices in the response")]
    NoMessageChoices,
    /// Failed to deserialize OpenAI API response.
    #[error("Failed to deserialize OpenAI API response: {0}")]
    DeserializeError(#[from] serde_json::Error),
    /// Refusal from OpenAI API.
    #[error("Refusal from OpenAI API: {0}")]
    Refusal(String),
    /// Function calls are not yet supported in kalosm with the OpenAI API.
    #[error("Function calls are not yet supported in kalosm with the OpenAI API")]
    FunctionCallsNotSupported,
}

/// A chat session for the OpenAI compatible chat model.
#[derive(Serialize, Deserialize, Clone)]
pub struct OpenAICompatibleChatSession {
    messages: Vec<crate::ChatMessage>,
}

impl OpenAICompatibleChatSession {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

impl ChatSession for OpenAICompatibleChatSession {
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
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[crate::ChatMessage],
        sampler: GenerationParameters,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        let myself = &*self.inner;
        let json = serde_json::json!({
            "messages": messages,
            "model": myself.model,
            "stream": true,
            "top_p": sampler.top_p,
            "temperature": sampler.temperature,
            "frequency_penalty": sampler.repetition_penalty,
            "max_completion_tokens": if sampler.max_length == u32::MAX { None } else { Some(sampler.max_length) },
            "stop": sampler.stop_on.clone(),
        });
        async move {
            let api_key = myself.client.resolve_api_key()?;
            let mut event_source = myself
                .client
                .reqwest_client
                .post(format!("{}/chat/completions", myself.client.base_url()))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&json)
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
                            .into_iter()
                            .next()
                            .ok_or(OpenAICompatibleChatModelError::NoMessageChoices)?;
                        if let Some(content) = first_choice.delta.refusal {
                            return Err(OpenAICompatibleChatModelError::Refusal(content));
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
                        if let Some(content) = first_choice.delta.content {
                            new_message_text += &content;
                            on_token(content)?;
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

/// A parser for any type that implements the [`Schema`] trait and [`Deserialize`].
#[derive(Debug, Clone, Copy)]
pub struct SchemaParser<P> {
    phantom: std::marker::PhantomData<P>,
}

impl<P> Default for SchemaParser<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> SchemaParser<P> {
    /// Create a new parser for the given schema.
    pub const fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<P> ModelConstraints for SchemaParser<P> {
    type Output = P;
}

impl<T: Schema + DeserializeOwned> CreateDefaultChatConstraintsForType<T>
    for OpenAICompatibleChatModel
{
    type DefaultConstraints = SchemaParser<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        SchemaParser::new()
    }
}

impl<P> StructuredChatModel<SchemaParser<P>> for OpenAICompatibleChatModel
where
    P: Schema + DeserializeOwned,
{
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[crate::ChatMessage],
        sampler: GenerationParameters,
        _: SchemaParser<P>,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<P, Self::Error>> + Send + 'a {
        let schema = P::schema();
        let mut schema: serde_json::Result<serde_json::Value> =
            serde_json::from_str(&schema.to_string());
        fn remove_unsupported_properties(schema: &mut serde_json::Value) {
            match schema {
                serde_json::Value::Null => {}
                serde_json::Value::Bool(_) => {}
                serde_json::Value::Number(_) => {}
                serde_json::Value::String(_) => {}
                serde_json::Value::Array(array) => {
                    for item in array {
                        remove_unsupported_properties(item);
                    }
                }
                serde_json::Value::Object(map) => {
                    map.retain(|key, value| {
                        const OPEN_AI_UNSUPPORTED_PROPERTIES: [&str; 19] = [
                            "minLength",
                            "maxLength",
                            "pattern",
                            "format",
                            "minimum",
                            "maximum",
                            "multipleOf",
                            "patternProperties",
                            "unevaluatedProperties",
                            "propertyNames",
                            "minProperties",
                            "maxProperties",
                            "unevaluatedItems",
                            "contains",
                            "minContains",
                            "maxContains",
                            "minItems",
                            "maxItems",
                            "uniqueItems",
                        ];
                        if OPEN_AI_UNSUPPORTED_PROPERTIES.contains(&key.as_str()) {
                            return false;
                        }

                        remove_unsupported_properties(value);
                        true
                    });
                }
            }
        }
        if let Ok(schema) = &mut schema {
            remove_unsupported_properties(schema);
        }

        let myself = &*self.inner;
        let json = schema.map(|schema| serde_json::json!({
            "messages": messages,
            "model": myself.model,
            "stream": true,
            "top_p": sampler.top_p,
            "temperature": sampler.temperature,
            "frequency_penalty": sampler.repetition_penalty,
            "max_completion_tokens": if sampler.max_length == u32::MAX { None } else { Some(sampler.max_length) },
            "stop": sampler.stop_on.clone(),
            "seed": sampler.seed(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }
        }));
        async move {
            let json = json?;
            let api_key = myself.client.resolve_api_key()?;
            let mut event_source = myself
                .client
                .reqwest_client
                .post(format!("{}/chat/completions", myself.client.base_url()))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&json)
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
                                _ => break,
                            }
                        }
                        if let Some(content) = &first_choice.delta.content {
                            on_token(content.clone())?;
                            new_message_text += content;
                        }
                    }
                }
            }

            let result = serde_json::from_str::<P>(&new_message_text)?;

            let new_message =
                crate::ChatMessage::new(crate::MessageType::UserMessage, new_message_text);

            session.messages.push(new_message);

            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use serde::Deserialize;

    use super::{
        ChatModel, CreateChatSession, GenerationParameters, OpenAICompatibleChatModelBuilder,
        SchemaParser, StructuredChatModel,
    };

    #[tokio::test]
    async fn test_gpt_4o_mini() {
        let model = OpenAICompatibleChatModelBuilder::new()
            .with_gpt_4o_mini()
            .build();

        let mut session = model.new_chat_session().unwrap();

        let messages = vec![crate::ChatMessage::new(
            crate::MessageType::UserMessage,
            "Hello, world!".to_string(),
        )];
        let all_text = Arc::new(RwLock::new(String::new()));
        model
            .add_messages_with_callback(
                &mut session,
                &messages,
                GenerationParameters::default().with_seed(1234),
                {
                    let all_text = all_text.clone();
                    move |token| {
                        let mut all_text = all_text.write().unwrap();
                        all_text.push_str(&token);
                        print!("{token}");
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                        Ok(())
                    }
                },
            )
            .await
            .unwrap();

        let all_text = all_text.read().unwrap();
        println!("{all_text}");

        assert!(!all_text.is_empty());
    }

    #[tokio::test]
    async fn test_gpt_4o_mini_constrained() {
        let model = OpenAICompatibleChatModelBuilder::new()
            .with_gpt_4o_mini()
            .build();

        let mut session = model.new_chat_session().unwrap();

        let messages = vec![crate::ChatMessage::new(
            crate::MessageType::UserMessage,
            "Give me a list of 5 primes.".to_string(),
        )];
        let all_text = Arc::new(RwLock::new(String::new()));

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        struct Constraints {
            primes: Vec<u8>,
        }

        let response: Constraints = model
            .add_message_with_callback_and_constraints(
                &mut session,
                &messages,
                GenerationParameters::default(),
                SchemaParser::new(),
                {
                    let all_text = all_text.clone();
                    move |token| {
                        let mut all_text = all_text.write().unwrap();
                        all_text.push_str(&token);
                        print!("{token}");
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                        Ok(())
                    }
                },
            )
            .await
            .unwrap();
        println!("{response:?}");

        let all_text = all_text.read().unwrap();
        println!("{all_text}");

        assert!(!all_text.is_empty());

        assert!(!response.primes.is_empty());
    }

    #[tokio::test]
    async fn test_gpt_4o_mini_constrained_with_option() {
        use kalosm_sample::Schema;

        let model = OpenAICompatibleChatModelBuilder::new()
            .with_gpt_4o_mini()
            .build();

        let mut session = model.new_chat_session().unwrap();

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        struct Constraints {
            name: Option<String>,
        }

        println!("schema: {}", Constraints::schema());

        {
            let all_text = Arc::new(RwLock::new(String::new()));
            let messages = vec![crate::ChatMessage::new(
            crate::MessageType::UserMessage,
            "What name, if any does this sentence contain: Evan is one of the developers of Kalosm".to_string(),
        )];

            let response: Constraints = model
                .add_message_with_callback_and_constraints(
                    &mut session,
                    &messages,
                    GenerationParameters::default(),
                    SchemaParser::new(),
                    {
                        let all_text = all_text.clone();
                        move |token| {
                            let mut all_text = all_text.write().unwrap();
                            all_text.push_str(&token);
                            print!("{token}");
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            Ok(())
                        }
                    },
                )
                .await
                .unwrap();
            println!("{response:?}");

            let all_text = all_text.read().unwrap();
            println!("{all_text}");

            assert!(!all_text.is_empty());

            assert!(response.name.is_some());
        }
        {
            let all_text = Arc::new(RwLock::new(String::new()));
            let messages = vec![crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                "What name, if any does this sentence contain: The earth is round".to_string(),
            )];

            let response: Constraints = model
                .add_message_with_callback_and_constraints(
                    &mut session,
                    &messages,
                    GenerationParameters::default(),
                    SchemaParser::new(),
                    {
                        let all_text = all_text.clone();
                        move |token| {
                            let mut all_text = all_text.write().unwrap();
                            all_text.push_str(&token);
                            print!("{token}");
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            Ok(())
                        }
                    },
                )
                .await
                .unwrap();
            println!("{response:?}");

            let all_text = all_text.read().unwrap();
            println!("{all_text}");

            assert!(!all_text.is_empty());

            assert!(response.name.is_none());
        }
    }

    #[tokio::test]
    async fn test_gpt_4o_mini_constrained_with_unit_enum() {
        use kalosm_sample::Schema;

        let model = OpenAICompatibleChatModelBuilder::new()
            .with_gpt_4o_mini()
            .build();

        let mut session = model.new_chat_session().unwrap();

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        struct Constraints {
            contains_name: ContainsName,
        }

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        enum ContainsName {
            Yes,
            No,
        }

        println!("schema: {}", Constraints::schema());

        {
            let all_text = Arc::new(RwLock::new(String::new()));
            let messages = vec![crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                "Does this sentence contain a name: Evan is one of the developers of Kalosm"
                    .to_string(),
            )];

            let response: Constraints = model
                .add_message_with_callback_and_constraints(
                    &mut session,
                    &messages,
                    GenerationParameters::default(),
                    SchemaParser::new(),
                    {
                        let all_text = all_text.clone();
                        move |token| {
                            let mut all_text = all_text.write().unwrap();
                            all_text.push_str(&token);
                            print!("{token}");
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            Ok(())
                        }
                    },
                )
                .await
                .unwrap();
            println!("{response:?}");

            let all_text = all_text.read().unwrap();
            println!("{all_text}");

            assert!(!all_text.is_empty());

            assert!(matches!(response.contains_name, ContainsName::Yes));
        }
        {
            let all_text = Arc::new(RwLock::new(String::new()));
            let messages = vec![crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                "Does this sentence contain a name: The earth is round".to_string(),
            )];

            let response: Constraints = model
                .add_message_with_callback_and_constraints(
                    &mut session,
                    &messages,
                    GenerationParameters::default(),
                    SchemaParser::new(),
                    {
                        let all_text = all_text.clone();
                        move |token| {
                            let mut all_text = all_text.write().unwrap();
                            all_text.push_str(&token);
                            print!("{token}");
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            Ok(())
                        }
                    },
                )
                .await
                .unwrap();
            println!("{response:?}");

            let all_text = all_text.read().unwrap();
            println!("{all_text}");

            assert!(!all_text.is_empty());

            assert!(matches!(response.contains_name, ContainsName::No));
        }
    }
}
