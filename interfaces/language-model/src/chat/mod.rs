//! A chat interface that builds on top of [`crate::Model`]

use crate::GenerationParameters;
use crate::ModelConstraints;
use crate::NoConstraints;
use futures_util::Future;
use futures_util::FutureExt;
use futures_util::Stream;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Display;
use std::future::IntoFuture;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::task::Poll;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::oneshot::Receiver;
use tokio::sync::RwLock as AsyncRwLock;

/// A trait for creating a chat session.
pub trait CreateChatSession {
    /// The type of error the chat session may return during operations.
    type Error: Send + Sync + 'static;

    /// The type of chat session.
    type ChatSession: ChatSessionImpl;

    /// Create a new chat session for this model.
    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error>;
}

/// A trait for unstructured chat models.
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

/// A trait for chat models that work with constraints.
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

/// A trait for chat sessions.
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

/// [`Chat`] is a chat interface that builds on top of [`crate::Model`]. It makes it easy to create a chat session with streaming responses, and constraints.
///
/// Let's start with a simple chat application:
///
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// // Before you create a chat session, you need a model. Llama::new_chat will create a good default chat model.
/// let model = Llama::new_chat().await.unwrap();
/// // Then you can build a chat session that uses that model
/// let mut chat = Chat::builder(model)
///     // The builder exposes methods for settings like the system prompt and constraints the bot response must follow
///     .with_system_prompt("The assistant will act like a pirate")
///     // Once you are done setting up the chat session, you can build it
///     .build();
///
/// loop {
///     // To use the chat session, you need to add messages to it
///     let mut response_stream = chat.add_message(prompt_input("\n> ").unwrap());
///     // And then display the response stream to the user
///     response_stream.to_std_out().await.unwrap();
/// }
/// # }
/// ```
///
/// If you run the application, you may notice that it takes more time for the assistant to start responding to long prompts.
/// The LLM needs to read and transform the prompt into a format it understands before it can start generating a response.
/// Kalosm stores that state in a chat session, which can be saved and loaded from the filesystem to make loading existing chat sessions faster.
///
/// You can save and load chat sessions from the filesystem using the [`Self::save_session`] and [`ChatBuilder::with_try_session_path`] methods:
///
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// // First, create a model to chat with
/// let model = Llama::new_chat().await.unwrap();
/// // Then try to load the chat session from the filesystem
/// let save_path = std::path::PathBuf::from("./chat.llama");
/// let mut chat = Chat::builder(model)
///     // You can try to load the chat session from the filesystem with the `with_try_session_path` method
///     .with_try_session_path(&save_path)
///     .build();
///
/// // Then you can add messages to the chat session as usual
/// let mut response_stream = chat.add_message(prompt_input("\n> ").unwrap());
/// // And then display the response stream to the user
/// response_stream.to_std_out().await.unwrap();
///
/// // After you are done, you can save the chat session to the filesystem
/// chat.save_session(&save_path).await.unwrap();
/// # }
/// ```
///
/// LLMs are powerful because of their generality, but sometimes you need more control over the output. For example, you might want the assistant to start with a certain phrase, or to follow a certain format.
///
/// In kalosm, you can use constraints to guide the model's response. Constraints are a way to specify the format of the output. When generating with constraints, the model will always respond with the specified format.
///
///
/// Let's create a chat application that uses constraints to guide the assistant's response to always start with "Yes!":
///
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// let model = Llama::new_chat().await.unwrap();
/// // Create constraints that parses Yes! and then stops on the end of the assistant's response
/// let constraints = LiteralParser::new("Yes!")
///     .then(model.default_assistant_constraints().unwrap());
/// // Create a chat session with the model and the constraints
/// let mut chat = Chat::builder(model)
///     .constrain_response(move |_history| constraints.clone())
///     .build();
///
/// // Chat with the user
/// loop {
///     let mut output_stream = chat.add_message(prompt_input("\n> ").unwrap());
///     output_stream.to_std_out().await.unwrap();
/// }
/// # }
/// ```
pub struct Chat<M: CreateChatSession> {
    model: M,
    session: OnceCell<Result<Arc<AsyncRwLock<M::ChatSession>>, M::Error>>,
    queued_messages: Vec<ChatHistoryItem>,
}

impl<M: CreateChatSession> Chat<M> {
    /// Create a new chat session with the default settings.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// // Before you create a chat session, you need to create a model. Llama::new_chat will create a good default chat model.
    /// let model = Llama::new_chat().await.unwrap();
    /// // If you don't need to customize the chat session, you can use the `new` method to create a chat session with the default settings
    /// let mut chat = Chat::new(model);
    /// # }
    /// ```
    pub fn new(model: M) -> Chat<M>
    where
        M: ChatModel,
    {
        Self {
            model,
            session: OnceCell::new(),
            queued_messages: Vec::new(),
        }
    }

    /// Adds a system prompt to the chat. The system prompt guides the model to respond in a certain way.
    /// If no system prompt is added, the model will use a default system prompt that instructs the model to respond in a way that is safe and respectful.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::builder(Llama::new_chat().await.unwrap())
    ///     .with_system_prompt("The assistant will act like a pirate.")
    ///     .build();
    /// # }
    /// ```
    pub fn with_system_prompt(mut self, system_prompt: impl ToString) -> Self {
        #[cfg(debug_assertions)]
        if let Some(Ok(session)) = self.session.get() {
            let session = session.blocking_read();
            let mut existing_history = session.history();
            existing_history.extend_from_slice(&self.queued_messages);
            if !existing_history.is_empty() {
                tracing::error!("System prompt should be the first message in the history. System prompt was added to the end of the history: {:?}", existing_history);
            }
        }

        // Add the system prompt to the queue
        self.queued_messages.push(ChatHistoryItem::new(
            MessageType::SystemPrompt,
            system_prompt.to_string(),
        ));

        self
    }

    /// Starts the chat instance with the given model session. This can be useful for resuming a chat session with a long context that has already been processed.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # use kalosm_llama::LlamaSession;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Load the model session from the filesystem
    /// let session = LlamaSession::load_from(std::path::PathBuf::from("./chat.llama")).unwrap();
    /// // Start the chat session with the cached session
    /// let mut chat = Chat::builder(Llama::new_chat().await.unwrap())
    ///     .with_session(session)
    ///     .build();
    /// # }
    /// ```
    pub fn with_session(mut self, session: M::ChatSession) -> Self {
        // Copy the history from the current chat session into the queue for the new chat session
        let mut existing_history = session.history();
        existing_history.extend_from_slice(&self.queued_messages);
        self.queued_messages = existing_history;

        // Set the new chat session
        self.session
            .set(Ok(Arc::new(AsyncRwLock::new(session))))
            .unwrap_or_else(|_| panic!("Chat session already set"));

        self
    }

    /// Adds a user message to the chat session and streams the bot response.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::new(Llama::new_chat().await.unwrap());
    /// let prompt = prompt_input("\n> ").unwrap();
    ///
    /// // You can add the user message to the chat session with the `add_message` method
    /// let mut response_stream = chat.add_message(prompt);
    /// // And then stream the result to std out
    /// response_stream.to_std_out().await.unwrap();
    /// # }
    /// ```
    pub fn add_message(&mut self, message: impl IntoChatMessage) -> ChatResponseBuilder<'_, M> {
        // First push the message to the queue
        self.queued_messages.push(message.into_chat_message());

        // Then create the builder that will respond to the message if it is awaited
        ChatResponseBuilder {
            chat_session: self,
            constraints: None,
            sampler: Some(GenerationParameters::default()),
            task: OnceLock::new(),
            queued_tokens: None,
            result: None,
        }
    }

    fn session_clone(&mut self) -> Result<Arc<AsyncRwLock<M::ChatSession>>, M::Error> {
        let session = self.session.get_or_init(|| {
            self.model
                .new_chat_session()
                .map(|session| Arc::new(AsyncRwLock::new(session)))
        });

        match session {
            Ok(session) => Ok(session.clone()),
            Err(_) => {
                let session_owned = self.session.take().unwrap();
                match session_owned {
                    Ok(_) => unreachable!(),
                    Err(session_err) => Err(session_err),
                }
            }
        }
    }

    /// Get a reference to the chat session or an error if the session failed to load.
    ///
    /// You can use the session to save the chat for later:
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::new(Llama::new_chat().await.unwrap());
    /// let save_path = std::path::PathBuf::from("./chat.llama");
    /// let session = chat.session();
    /// session.save_session(&save_path).await.unwrap();
    /// # }
    /// ```
    ///
    /// Or get the chat history:
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::new(Llama::new_chat().await.unwrap());
    /// // Add a message to the chat history
    /// chat.add_message("Hello, world!").to_std_out().await.unwrap();
    /// // Get the chat history
    /// let history = chat.history();
    /// println!("{:?}", history);
    /// # }
    /// ````
    pub fn session(&self) -> Result<impl Deref<Target = M::ChatSession> + use<'_, M>, &M::Error> {
        self.session
            .get_or_init(|| {
                self.model
                    .new_chat_session()
                    .map(|session| Arc::new(AsyncRwLock::new(session)))
            })
            .as_ref()
            .map(|session| session.blocking_read())
    }
}

pub struct ChatResponseBuilder<
    'a,
    M: CreateChatSession,
    Constraints = NoConstraints,
    Sampler = GenerationParameters,
> {
    chat_session: &'a mut Chat<M>,
    constraints: Option<Constraints>,
    sampler: Option<Sampler>,
    task: OnceLock<RwLock<Pin<Box<dyn Future<Output = ()> + Send>>>>,
    result: Option<Receiver<Result<Box<dyn Any + Send>, M::Error>>>,
    queued_tokens: Option<UnboundedReceiver<String>>,
}

impl<'a, M: CreateChatSession, Constraints, Sampler>
    ChatResponseBuilder<'a, M, Constraints, Sampler>
{
    /// Constrains the model's response to the given parser. This can be used to make the model start with a certain phrase, or to make the model respond in a certain way.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Create constraints that parses Yes! and then stops on the end of the assistant's response
    /// let constraints = LiteralParser::new("Yes!")
    ///     .then(model.default_assistant_constraints().unwrap());
    /// // Create a chat session with the model and the constraints
    /// let mut chat = Chat::new(model);
    ///
    /// // Chat with the user
    /// loop {
    ///     let mut output_stream = chat.add_message(prompt_input("\n> ").unwrap()).with_constraints(constraints.clone());
    ///     output_stream.to_std_out().await.unwrap();
    /// }
    /// # }
    /// ```
    pub fn with_constraints<NewConstraints: ModelConstraints>(
        self,
        constraints: NewConstraints,
    ) -> ChatResponseBuilder<'a, M, NewConstraints, Sampler> {
        ChatResponseBuilder {
            chat_session: self.chat_session,
            constraints: Some(constraints),
            sampler: self.sampler,
            queued_tokens: None,
            result: None,
            task: OnceLock::new(),
        }
    }

    /// Sets the sampler to use for generating responses. The sampler determines how tokens are choosen from the probability distribution
    /// the model generates. They can be used to make the model more or less predictable and prevent repetition.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Create the sampler to use for the chat session
    /// let sampler = GenerationParameters::default().sampler();
    ///
    /// // Create a chat session with the model and the constraints
    /// let mut chat = Chat::new(model);
    ///
    /// // Chat with the user
    /// loop {
    ///     let mut output_stream = chat.add_message(prompt_input("\n> ").unwrap()).with_sampler(sampler);
    ///     output_stream.to_std_out().await.unwrap();
    /// }
    /// # }
    /// ```
    pub fn with_sampler<NewSampler>(
        self,
        sampler: NewSampler,
    ) -> ChatResponseBuilder<'a, M, Constraints, NewSampler> {
        ChatResponseBuilder {
            chat_session: self.chat_session,
            constraints: self.constraints,
            sampler: Some(sampler),
            queued_tokens: None,
            result: None,
            task: OnceLock::new(),
        }
    }
}

impl<'a, M, Sampler> ChatResponseBuilder<'a, M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: ChatModel<Sampler> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Send + Sync + Unpin + 'static,
{
    fn ensure_unstructured_task_started(&mut self) {
        if self.task.get().is_none() {
            let messages = std::mem::take(&mut self.chat_session.queued_messages);
            let sampler = self
                .sampler
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let all_text = Arc::new(Mutex::new(String::new()));
            let on_token = {
                let all_text = all_text.clone();
                move |tok: String| {
                    all_text.lock().unwrap().push_str(&tok);
                    _ = tx.send(tok);
                    Ok(())
                }
            };
            let session = self.chat_session.session_clone();
            let model = self.chat_session.model.clone();
            let future = async move {
                let session = session?;
                let mut session = session.write().await;
                model
                    .add_messages_with_callback(&mut session, &messages, sampler, on_token)
                    .await?;
                let mut all_text = all_text.lock().unwrap();
                let all_text = std::mem::take(&mut *all_text);
                Ok(Box::new(all_text) as Box<dyn Any + Send>)
            };
            let wrapped = async move {
                let result: Result<Box<dyn Any + Send>, M::Error> = future.await;
                _ = result_tx.send(result);
            };
            let task = Box::pin(wrapped);
            self.task
                .set(RwLock::new(task))
                .unwrap_or_else(|_| panic!("Task already set"));
        }
    }
}

impl<'a, M, Sampler> Stream for ChatResponseBuilder<'a, M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: ChatModel<Sampler> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Send + Sync + Unpin + 'static,
{
    type Item = String;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = Pin::get_mut(self);
        myself.ensure_unstructured_task_started();
        {
            if let Some(token) = &mut myself.queued_tokens {
                if let Poll::Ready(Some(token)) = token.poll_recv(cx) {
                    return Poll::Ready(Some(token));
                }
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        task.poll_unpin(cx).map(|_| None)
    }
}

impl<'a, M, Sampler> IntoFuture for ChatResponseBuilder<'a, M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: ChatModel<Sampler> + Send + Sync + Unpin + Clone + 'static,
    M::ChatSession: Clone + Send + Sync + Unpin + 'static,
{
    type Output = Result<String, M::Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(mut self) -> Self::IntoFuture {
        self.ensure_unstructured_task_started();

        Box::pin(async move {
            if !self.result.is_some() {
                self.task.into_inner().unwrap().into_inner().unwrap().await;
            }
            let result = self.result.take().unwrap().await.unwrap();
            let result = result.map(|boxed| *boxed.downcast::<String>().unwrap());
            result
        })
    }
}

impl<'a, M, Constraints, Sampler> ChatResponseBuilder<'a, M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredChatModel<Constraints, Sampler> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    fn ensure_structured_task_started(&mut self) {
        if self.task.get().is_none() {
            let messages = std::mem::take(&mut self.chat_session.queued_messages);
            let sampler = self
                .sampler
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let constraints = self
                .constraints
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let on_token = move |tok: String| {
                _ = tx.send(tok);
                Ok(())
            };
            let session = self.chat_session.session_clone();
            let model = self.chat_session.model.clone();
            let future = async move {
                let session = session?;
                let mut session = session.write().await;
                model
                    .add_message_with_callback_and_constraints(
                        &mut session,
                        &messages,
                        sampler,
                        constraints,
                        on_token,
                    )
                    .await
                    .map(|value| Box::new(value) as Box<dyn Any + Send>)
            };
            let wrapped = async move {
                let result: Result<Box<dyn Any + Send>, M::Error> = future.await;
                _ = result_tx.send(result);
            };
            let task = Box::pin(wrapped);
            self.task
                .set(RwLock::new(task))
                .unwrap_or_else(|_| panic!("Task already set"));
        }
    }
}

impl<'a, M, Constraints, Sampler> Stream for ChatResponseBuilder<'a, M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredChatModel<Constraints, Sampler> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    type Item = String;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = Pin::get_mut(self);
        myself.ensure_structured_task_started();
        {
            if let Some(token) = &mut myself.queued_tokens {
                if let Poll::Ready(Some(token)) = token.poll_recv(cx) {
                    return Poll::Ready(Some(token));
                }
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        task.poll_unpin(cx).map(|_| None)
    }
}

impl<'a, M, Constraints, Sampler> IntoFuture for ChatResponseBuilder<'a, M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredChatModel<Constraints, Sampler> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    type Output = Result<Constraints::Output, M::Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(mut self) -> Self::IntoFuture {
        self.ensure_structured_task_started();

        Box::pin(async move {
            if !self.result.is_some() {
                self.task.into_inner().unwrap().into_inner().unwrap().await;
            }
            let result = self.result.take().unwrap().await.unwrap();
            let result = result.map(|boxed| *boxed.downcast::<Constraints::Output>().unwrap());
            result
        })
    }
}
