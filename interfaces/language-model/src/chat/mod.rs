use crate::GenerationParameters;
use crate::ModelConstraints;
use futures_util::Future;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

mod ext;
pub use ext::*;
mod task;
pub use task::*;
mod chat_builder;
pub use chat_builder::*;
mod boxed;
pub use boxed::*;

/// A trait for creating a chat session. While it the core trait
/// every chat session implementation implements, most methods to use models that implement
/// this trait are implemented in the [`ChatModelExt`] trait.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateChatSession trait
///     let llm = Llama::new_chat().await.unwrap();
///     // Create a new chat for the model
///     let mut chat = llm.chat();
///     // Add a message to the chat session
///     chat("Hello, world!").to_std_out().await.unwrap();
/// }
/// ```
pub trait CreateChatSession {
    /// The type of error the chat session may return during operations.
    type Error: Send + Sync + 'static;

    /// The type of chat session.
    type ChatSession: ChatSession;

    /// Create a new chat session for this model.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     // Create a new model which implements the CreateChatSession trait
    ///     let llm = Llama::new_chat().await.unwrap();
    ///     // Create a new chat session for the model
    ///     let mut chat_session = llm.new_chat_session().unwrap();
    /// }
    /// ```
    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error>;
}

/// A trait for unstructured chat models. This trait is required for any chat models
/// even if they do not support structured generation. While this trait is implemented for
/// all chat models, most methods to use models that implement this trait are implemented
/// in the [`ChatModelExt`] trait.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateChatSession trait
///     let llm = Llama::new_chat().await.unwrap();
///     // Create a new chat session for the model
///     let mut chat_session = llm.new_chat_session().unwrap();
///     // Add a message to the chat session
///     llm.add_messages_with_callback(
///         &mut chat_session,
///         &[ChatMessage::new(MessageType::UserMessage, "Hello, world!")],
///         GenerationParameters::new(),
///         |token| {
///             println!("{token}");
///             Ok(())
///         },
///     )
///     .await
///     .unwrap();
/// }
/// ```
pub trait ChatModel<Sampler = GenerationParameters>: CreateChatSession {
    /// Add messages to the chat session with a callback that is called for each token.
    ///
    /// See [`Chat::add_message`] for nicer API with examples
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: Sampler,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a;
}

/// A trait for unstructured chat models that support structured generation. While this trait is implemented for
/// all structured chat models, most methods to use models that implement this trait are implemented
/// in the [`ChatModelExt`] trait.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateChatSession trait
///     let llm = Llama::new_chat().await.unwrap();
///     // Create a new chat session for the model
///     let mut chat_session = llm.new_chat_session().unwrap();
///     // Create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
///     let parser = i32::new_parser();
///     // Add a message to the chat session with the given constraints
///     let mut result: i32 = llm.add_message_with_callback_and_constraints(&mut chat_session, &[ChatMessage::new(MessageType::UserMessage, "5 + 5")], GenerationParameters::new(), parser, |token| {
///         println!("{token}");
///         Ok(())
///     }).await.unwrap();
///     println!("{result}");
/// }
/// ```
pub trait StructuredChatModel<Constraints: ModelConstraints, Sampler = GenerationParameters>:
    ChatModel<Sampler>
{
    /// Add messages to the chat session with a callback that is called for each token and a constraints the response must follow.
    ///
    /// See [`ChatResponseBuilder::with_constraints`] for nicer API with examples
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: Sampler,
        constraints: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + Send + 'a;
}

/// A trait that defines the default constraints for a type with this chat model.
pub trait CreateDefaultChatConstraintsForType<T>:
    StructuredChatModel<Self::DefaultConstraints>
{
    /// The default constraints for this type that work with this chat model.
    type DefaultConstraints: ModelConstraints<Output = T>;

    /// Create [`Self::DefaultConstraints`] which parse the type `T` for this chat model.
    fn create_default_constraints() -> Self::DefaultConstraints;
}

#[doc = include_str!("../../docs/chat_session.md")]
pub trait ChatSession {
    /// The type of error the chat session may return during operations.
    type Error: Send + Sync + 'static;

    /// Serialize the session into bytes.
    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error>;

    /// # Loading sessions
    ///
    /// Sessions can be deserialized to and from bytes using the [`ChatSession::from_bytes`] method.
    /// Caching a session avoids re-processing the text again when the session is resumed.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new_chat().await.unwrap();
    ///     let mut chat = llm.chat();
    ///
    ///     // Feed some text into the session
    ///     chat("What is the capital of France?").await.unwrap();
    ///
    ///     // Save the session to bytes
    ///     let session = chat.session().unwrap();
    ///     let session_as_bytes = session.to_bytes().unwrap();
    ///
    ///     // And write those bytes to a file
    ///     std::fs::write("session.bin", session_as_bytes).unwrap();
    /// }
    /// ```
    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error> {
        let mut bytes = Vec::new();
        self.write_to(&mut bytes)?;
        Ok(bytes)
    }

    /// # Loading sessions
    ///
    /// Sessions can be deserialized to and from bytes using the [`ChatSession::from_bytes`] method.
    /// Caching a session avoids re-processing the text again when the session is resumed.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new_chat().await.unwrap();
    ///     // Load a chat session from a file
    ///     let session =
    ///         LlamaChatSession::from_bytes(std::fs::read("session.bin").unwrap().as_slice()).unwrap();
    ///     let mut chat = llm.chat().with_session(session);
    ///
    ///     // Feed some more text into the session
    ///     chat("What was my first question?")
    ///         .to_std_out()
    ///         .await
    ///         .unwrap();
    /// }
    /// ```
    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;

    /// # Session History
    ///
    /// Get the history of the session. The history is a list of messages that have been sent to the model.
    ///
    /// ## Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut llm = Llama::new_chat().await.unwrap();
    /// let mut chat = llm.chat();
    /// // Add a message to the session
    /// chat("Hello, world!");
    /// // Get the history of the session
    /// let history = chat.session().unwrap().history();
    /// assert_eq!(history.len(), 1);
    /// assert_eq!(history[0].role(), MessageType::UserMessage);
    /// assert_eq!(history[0].content(), "Hello, world!");
    /// # }
    /// ```
    fn history(&self) -> Vec<ChatMessage>;

    /// # Cloning Sessions
    ///
    /// Not all chat models support cloning sessions, but if a model does support
    /// cloning sessions, you can clone a session using the [`ChatSession::try_clone`] method
    /// to clone a session state while retaining the original session.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new_chat().await.unwrap();
    ///     let mut chat = llm.chat();
    ///
    ///     // Feed some text into the session
    ///     chat("What is the capital of France?").await.unwrap();
    ///     let mut session = chat.session().unwrap();
    ///
    ///     // Clone the session
    ///     let cloned_session = session.try_clone().unwrap();
    ///
    ///     // Feed some more text into the cloned session
    ///     let mut chat = llm.chat().with_session(cloned_session);
    ///     chat("What was my first question?").await.unwrap();
    /// }
    /// ```
    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

/// A simple helper function for prompting the user for input.
pub fn prompt_input(prompt: impl Display) -> Result<String, std::io::Error> {
    use std::io::Write;
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    input.pop();
    Ok(input)
}

/// The type of a chat message
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// A system prompt message. System prompts should always be the first message in a chat session.
    #[serde(rename = "developer")]
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
pub struct ChatMessage {
    role: MessageType,
    content: String,
}

impl ChatMessage {
    /// Creates a new chat history item.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let llm = Llama::new_chat().await.unwrap();
    /// let mut chat = llm.chat();
    /// chat.add_message(ChatMessage::new(MessageType::UserMessage, "Hello, world!"));
    /// # }
    /// ```
    pub fn new(role: MessageType, contents: impl ToString) -> Self {
        Self {
            role,
            content: contents.to_string(),
        }
    }

    /// Returns the type of the chat message.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let message = ChatMessage::new(MessageType::UserMessage, "Hello, world!");
    /// assert_eq!(message.role(), MessageType::UserMessage);
    /// # }
    /// ```
    pub fn role(&self) -> MessageType {
        self.role
    }

    /// Returns the content of the item.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let message = ChatMessage::new(MessageType::UserMessage, "Hello, world!");
    /// assert_eq!(message.content(), "Hello, world!");
    /// # }
    /// ```
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
/// let chat_message = ChatMessage::new(MessageType::ModelAnswer, "Hello, world!".to_string());
/// assert_eq!(chat_message.role(), MessageType::ModelAnswer);
/// # }
/// ```
pub trait IntoChatMessage {
    /// Convert the type into a chat message.
    fn into_chat_message(self) -> ChatMessage;
}

impl<S: ToString> IntoChatMessage for S {
    fn into_chat_message(self) -> ChatMessage {
        ChatMessage::new(MessageType::UserMessage, self.to_string())
    }
}

impl IntoChatMessage for ChatMessage {
    fn into_chat_message(self) -> ChatMessage {
        self
    }
}
