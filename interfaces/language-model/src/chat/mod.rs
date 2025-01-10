use crate::GenerationParameters;
use crate::ModelConstraints;
use futures_util::Future;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

mod ext;
pub use ext::*;

pub trait CreateChatSession {
    /// The type of error the chat session may return during operations.
    type Error: Send + Sync + 'static;

    /// The type of chat session.
    type ChatSession: ChatSessionImpl;

    /// Create a new chat session for this model.
    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error>;
}

pub trait ChatModel<Sampler = GenerationParameters>: CreateChatSession {
    /// Add messages to the chat session with a callback that is called for each token.
    fn add_messages_with_callback(
        &self,
        session: &mut Self::ChatSession,
        messages: &[ChatHistoryItem],
        sampler: Sampler,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

pub trait StructuredChatModel<Constraints: ModelConstraints, Sampler = GenerationParameters>:
    ChatModel<Sampler>
{
    /// Add messages to the chat session with a callback that is called for each token and a constraints the response must follow.
    fn add_message_with_callback_and_constraints(
        &self,
        session: &mut Self::ChatSession,
        messages: &[ChatHistoryItem],
        sampler: Sampler,
        constraints: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + Send;
}

pub trait ChatSessionImpl {
    /// The type of error the chat session may return during operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Serialize the session into bytes.
    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error>;

    /// Write the session to bytes.
    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error> {
        let mut bytes = Vec::new();
        self.write_to(&mut bytes)?;
        Ok(bytes)
    }

    /// Load the session from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;

    /// Get the history of the session.
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

/// A trait for types that can be converted into a chat message.
///
/// # Example
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// // Displayable types are converted into a user chat message
/// let user_message = "Hello, world!";
/// let chat_message = user_message.into_chat_message();
/// assert_eq!(chat_message.role(), MessageType::UserMessage);
///
/// // Or you can create a chat message manually
/// let chat_message = ChatHistoryItem::new(MessageType::ModelAnswer, "Hello, world!".to_string());
/// assert_eq!(chat_message.role(), MessageType::ModelAnswer);
/// # }
/// ```
pub trait IntoChatMessage {
    /// Convert the type into a chat message.
    fn into_chat_message(self) -> ChatHistoryItem;
}

impl<S: ToString> IntoChatMessage for S {
    fn into_chat_message(self) -> ChatHistoryItem {
        ChatHistoryItem::new(MessageType::UserMessage, self.to_string())
    }
}

impl IntoChatMessage for ChatHistoryItem {
    fn into_chat_message(self) -> ChatHistoryItem {
        self
    }
}
