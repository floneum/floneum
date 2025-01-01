//! A chat interface that builds on top of [`crate::Model`]

mod local;

use crate::{Model, ModelConstraints};
use crate::{NoConstraintsSupported, Session};
use futures_util::Future;
use kalosm_sample::{ParserExt, SendCreateParserState};
use llm_samplers::types::Sampler;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

trait ChatModel<Constraints: ModelConstraints = NoConstraintsSupported> {
    type ChatSession: ChatSessionImpl;
    type Error;

    fn new_chat_session(&self) -> Self::ChatSession;

    fn add_message_with_callback(
        &self,
        session: &mut Self::ChatSession,
        message: ChatHistoryItem,
        on_token: impl FnMut(String) -> Result<(), Self::Error>,
    ) -> impl Future<Output = Result<(), Self::Error>>;

    fn add_message_with_callback_and_constraints(
        &self,
        session: &mut Self::ChatSession,
        message: ChatHistoryItem,
        constraints: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error>,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>>;
}

trait ChatSessionImpl {
    type Error;

    fn save_session(&self) -> Result<Vec<u8>, Self::Error>;

    fn load_session(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: Sized;

    fn history(&self) -> Vec<ChatHistoryItem>;
}

/// A simple helper function for prompting the user for input.
pub fn prompt_input(prompt: impl Display) -> Result<String, std::io::Error> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    input.pop();
    Ok(input)
}

/// The type of a chat message
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// A system prompt.
    #[serde(rename = "system")]
    SystemPrompt,
    /// A user message.
    #[serde(rename = "user")]
    UserMessage,
    /// A model answer.
    #[serde(rename = "assistant")]
    ModelAnswer,
}

/// A single item in the chat history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatHistoryItem {
    role: MessageType,
    content: String,
}

impl ChatHistoryItem {
    /// Creates a new chat history item.
    pub fn new(role: MessageType, contents: impl ToString) -> Self {
        Self {
            role,
            content: contents.to_string(),
        }
    }

    /// Returns the type of the item.
    pub fn role(&self) -> MessageType {
        self.role
    }

    /// Returns the content of the item.
    pub fn content(&self) -> &str {
        &self.content
    }
}
