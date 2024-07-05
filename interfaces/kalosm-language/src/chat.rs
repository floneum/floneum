//! A chat interface that builds on top of [`kalosm_language_model::Model`]

use std::{
    fmt::Display,
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
};

use anyhow::Result;
use futures_util::Future;
use kalosm_language_model::ChatMarkers;
use kalosm_language_model::Session;
use kalosm_language_model::{GenerationParameters, Model, ModelExt, SyncModel, SyncModelExt};
use kalosm_sample::{ArcParser, CreateParserState, ParserExt};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokio::sync::{mpsc::unbounded_channel, oneshot};

type ResponseConstraintGenerator =
    Arc<Mutex<Box<dyn FnMut(&[ChatHistoryItem]) -> ArcParser<()> + Send + Sync>>>;

/// A simple helper function for prompting the user for input.
pub fn prompt_input(prompt: impl Display) -> Result<String> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    input.pop();
    Ok(input)
}

/// The type of a chat message
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageType {
    /// A system prompt.
    SystemPrompt,
    /// A user message.
    UserMessage,
    /// A model answer.
    ModelAnswer,
}

/// A single item in the chat history.
#[derive(Clone)]
pub struct ChatHistoryItem {
    ty: MessageType,
    contents: String,
}

impl ChatHistoryItem {
    /// Creates a new chat history item.
    pub fn new(ty: MessageType, contents: impl Into<String>) -> Self {
        Self {
            ty,
            contents: contents.into(),
        }
    }

    /// Returns the type of the item.
    pub fn ty(&self) -> MessageType {
        self.ty
    }

    /// Returns the contents of the item.
    pub fn contents(&self) -> &str {
        &self.contents
    }
}

/// The history of a chat session.
struct ChatSession<Model: SyncModel> {
    user_marker: String,
    end_user_marker: String,
    assistant_marker: String,
    end_assistant_marker: String,
    history: Arc<RwLock<Vec<ChatHistoryItem>>>,
    session: Model::Session,
    unfed_text: String,
    bot_constraints: Option<ResponseConstraintGenerator>,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
}

impl<Model: SyncModel> ChatSession<Model> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    fn new(
        model: &mut Model,
        system_prompt_marker: String,
        end_system_prompt_marker: String,
        user_marker: String,
        end_user_marker: String,
        assistant_marker: String,
        end_assistant_marker: String,
        system_prompt: String,
        bot_constraints: Option<ResponseConstraintGenerator>,
        sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
        session: Option<Model::Session>,
        initial_history: Vec<ChatHistoryItem>,
        shared_history: Arc<RwLock<Vec<ChatHistoryItem>>>,
    ) -> Self {
        let feed_initial_messages = session.is_none();
        let session = session.unwrap_or_else(|| model.new_session().unwrap());
        let unfed_text = if feed_initial_messages {
            let mut unfed_text = String::new();
            unfed_text += &system_prompt_marker;
            unfed_text += &system_prompt;
            unfed_text += &end_system_prompt_marker;
            unfed_text
        } else {
            String::new()
        };
        *shared_history.write().unwrap() = vec![ChatHistoryItem {
            ty: MessageType::SystemPrompt,
            contents: system_prompt,
        }];

        let mut myself = Self {
            user_marker,
            end_user_marker,
            assistant_marker,
            end_assistant_marker,
            session,
            unfed_text,
            history: shared_history,
            bot_constraints,
            sampler,
        };

        if feed_initial_messages {
            for item in initial_history {
                match item.ty() {
                    MessageType::SystemPrompt => {
                        panic!("Initial history cannot contain a system prompt");
                    }
                    MessageType::UserMessage => {
                        myself.add_user_message(item.contents);
                    }
                    MessageType::ModelAnswer => {
                        myself.add_bot_message(item.contents);
                    }
                }
            }
        }

        myself
    }

    /// Adds a message to the history.
    fn add_message(
        &mut self,
        message: String,
        model: &mut Model,
        stream: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        self.add_user_message(message);
        let mut bot_response = String::new();
        self.unfed_text += &self.assistant_marker;
        let prompt = std::mem::take(&mut self.unfed_text);
        let bot_constraints = &self.bot_constraints;
        loop {
            bot_response.clear();

            let mut on_token = |tok: String| {
                let tok = tok
                    .strip_suffix(&self.end_assistant_marker)
                    .unwrap_or(&tok)
                    .to_string();
                bot_response += &tok;
                // Send the new token to the stream
                stream.send(tok)?;
                Ok(())
            };

            match bot_constraints {
                Some(constraints) => {
                    let mut constraints = constraints.lock().unwrap();
                    let constraints = constraints(&self.history.read().unwrap());
                    let state = constraints.create_parser_state();
                    model.generate_structured(
                        &mut self.session,
                        &prompt,
                        constraints,
                        state,
                        self.sampler.clone(),
                        on_token,
                        Some(32),
                    )?;
                }
                None => {
                    model.stream_text_with_sampler(
                        &mut self.session,
                        &prompt,
                        None,
                        Some(&self.end_assistant_marker),
                        self.sampler.clone(),
                        |tok| {
                            on_token(tok)?;
                            Ok(kalosm_language_model::ModelFeedback::Continue)
                        },
                    )?;
                }
            }
        }
    }

    fn add_user_message(&mut self, message: String) {
        self.unfed_text += &self.user_marker;
        self.unfed_text += &message;
        self.unfed_text += &self.end_user_marker;
        self.history.write().unwrap().push(ChatHistoryItem {
            ty: MessageType::UserMessage,
            contents: message,
        });
    }

    fn add_bot_message(&mut self, message: String) {
        self.unfed_text += &self.assistant_marker;
        self.unfed_text += &message;
        self.unfed_text += &self.end_assistant_marker;
        self.history.write().unwrap().push(ChatHistoryItem {
            ty: MessageType::ModelAnswer,
            contents: message,
        });
    }
}

/// A builder for [`Chat`].
pub struct ChatBuilder<M: Model> {
    model: M,
    chat_markers: ChatMarkers,
    session: Option<<M::SyncModel as kalosm_language_model::SyncModel>::Session>,
    system_prompt: String,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    bot_constraints: Option<ResponseConstraintGenerator>,
    initial_history: Vec<ChatHistoryItem>,
}

impl<M: Model> ChatBuilder<M> {
    fn new(model: M) -> ChatBuilder<M> {
        let chat_markers = model.chat_markers().expect("Model does not support chat");

        ChatBuilder {
            model,
            chat_markers,
            session: None,
            system_prompt: "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.".into(),
            sampler: Arc::new(Mutex::new(GenerationParameters::default().sampler())),
            bot_constraints: None,
            initial_history: Vec::new(),
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
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Sets the [`Sampler`] to use for generating responses.
    pub fn with_sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Arc::new(Mutex::new(sampler));
        self
    }

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
    /// let mut chat = Chat::builder(model)
    ///     .with_system_prompt(character_description)
    ///     .constrain_response(move |_history| constraints.clone())
    ///     .build();
    ///
    /// // Chat with the user
    /// loop {
    ///     let output_stream = chat.add_message(prompt_input("\n> ").unwrap());
    ///     output_stream.to_std_out().await.unwrap();
    /// }
    /// # }
    /// ```
    pub fn constrain_response<P>(
        self,
        mut bot_constraints: impl FnMut(&[ChatHistoryItem]) -> P + Send + Sync + 'static,
    ) -> ChatBuilder<M>
    where
        P: CreateParserState + Sized + Send + Sync + 'static,
        P::Output: Send + Sync + 'static,
        P::PartialState: Send + Sync + 'static,
    {
        ChatBuilder {
            model: self.model,
            chat_markers: self.chat_markers,
            session: self.session,
            system_prompt: self.system_prompt,
            sampler: self.sampler,
            bot_constraints: Some(Arc::new(Mutex::new(Box::new(
                move |history: &[ChatHistoryItem]| {
                    bot_constraints(history).map_output(|_| ()).boxed()
                },
            )
                as Box<dyn FnMut(&[ChatHistoryItem]) -> ArcParser + Send + Sync>))),
            initial_history: self.initial_history,
        }
    }

    /// Starts the chat instance with the given model session. This can be useful for resuming a chat session with a long context that has already been processed.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Load the model session from the filesystem
    /// let session = LlamaModel::load_from(std::path::PathBuf::from("./chat.llama")).unwrap();
    /// // Start the chat session with the cached session
    /// let mut chat = Chat::builder(Llama::new_chat().await.unwrap())
    ///     .with_session(session)
    ///     .build();
    /// # }
    /// ```
    pub fn with_session(
        mut self,
        session: <M::SyncModel as kalosm_language_model::SyncModel>::Session,
    ) -> Self {
        self.session = Some(session);
        self
    }

    /// Try to load the chat session from the given path. If the session is not found, the default session will be used.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::builder(Llama::new_chat().await.unwrap())
    ///     .with_try_session_path("./chat.llama")
    ///     .build();
    /// # }
    /// ```
    pub fn with_try_session_path(self, path: impl AsRef<std::path::Path>) -> Self {
        let session = <M::SyncModel as kalosm_language_model::SyncModel>::Session::load_from(path);
        if let Ok(session) = session {
            self.with_session(session)
        } else {
            self
        }
    }

    /// Set the initial history of the chat. Each message in the original history will be added to the chat history, and the model will be fed the user messages.
    ///
    /// > **Note**: The system prompt automatically added to the chat history and should not be included in the initial history.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::builder(Llama::new_chat().await.unwrap())
    ///     .with_initial_history(vec![
    ///         ChatHistoryItem::new(MessageType::SystemPrompt, "The assistant will act like a pirate."),
    ///         ChatHistoryItem::new(MessageType::UserMessage, "Hello!"),
    ///         ChatHistoryItem::new(MessageType::ModelAnswer, "Arrr matey, how ar ya?"),
    ///     ])
    ///     .build();
    /// # }
    /// ```
    pub fn with_initial_history(mut self, initial_history: Vec<ChatHistoryItem>) -> Self {
        self.initial_history = initial_history;
        self
    }

    /// Builds a [`Chat`] instance.
    pub fn build(self) -> Chat
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        let Self {
            model,
            chat_markers,
            system_prompt,
            sampler,
            bot_constraints,
            session,
            initial_history,
        } = self;
        let system_prompt_marker = chat_markers.system_prompt_marker.to_string();
        let end_system_prompt_marker = chat_markers.end_system_prompt_marker.to_string();
        let user_marker = chat_markers.user_marker.to_string();
        let end_user_marker = chat_markers.end_user_marker.to_string();
        let assistant_marker = chat_markers.assistant_marker.to_string();
        let end_assistant_marker = chat_markers.end_assistant_marker.to_string();
        let (sender_tx, mut sender_rx) = unbounded_channel();
        let shared_history = Arc::new(RwLock::new(Vec::new()));
        {
            let shared_history = shared_history.clone();

            tokio::spawn(async move {
                let (tx, rx) = oneshot::channel();
                {
                    model
                        .run_sync(move |model| {
                            Box::pin(async move {
                                let _ = tx.send(ChatSession::new(
                                    model,
                                    system_prompt_marker,
                                    end_system_prompt_marker,
                                    user_marker,
                                    end_user_marker,
                                    assistant_marker,
                                    end_assistant_marker,
                                    system_prompt,
                                    bot_constraints,
                                    sampler,
                                    session,
                                    initial_history,
                                    shared_history,
                                ));
                            })
                        })
                        .unwrap();
                }

                let Ok(session) = rx.await else {
                    tracing::error!("Error loading session");
                    return;
                };
                let chat_session = Arc::new(Mutex::new(session));

                while let Some(message) = sender_rx.recv().await {
                    match message {
                        Message::AddMessage {
                            message,
                            response_tx,
                        } => {
                            let chat_session = chat_session.clone();
                            model
                                .run_sync(move |model| {
                                    Box::pin(async move {
                                        let mut chat_session = chat_session.lock().unwrap();
                                        if let Err(err) =
                                            chat_session.add_message(message, model, response_tx)
                                        {
                                            tracing::error!("Error adding message: {}", err);
                                        }
                                    })
                                })
                                .unwrap();
                        }
                        Message::SaveSession { path, resolve } => {
                            let chat_session = chat_session.lock().unwrap();
                            resolve.send(chat_session.session.save_to(path)).unwrap();
                        }
                    }
                }
            });
        }

        Chat {
            sender: sender_tx,
            shared_history,
        }
    }
}

enum Message {
    AddMessage {
        message: String,
        response_tx: tokio::sync::mpsc::UnboundedSender<String>,
    },
    SaveSession {
        path: PathBuf,
        resolve: tokio::sync::oneshot::Sender<Result<()>>,
    },
}

/// [`Chat`] is a chat interface that builds on top of [`kalosm_language_model::Model`]. It makes it easy to create a chat session with streaming responses, and constraints.
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
///     let response_stream = chat.add_message(prompt_input("\n> ").unwrap());
///     // And then display the response stream to the user
///     response_stream.to_std_out().await.unwrap();
/// }
/// #}
/// ```
///
/// If you run the application, you may notice that it takes more time for the assistant to start responding to long prompts.
/// The LLM needs to read and transform the prompt into a format it understands before it can start generating a response.
/// Kalosm stores that state in a chat session, which can be saved and loaded from the filesystem to make loading existing chat sessions faster.
///
/// You can save and load chat sessions from the filesystem using the [`Self::save_session`] and [`Self::load_session`] methods:
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
/// let response_stream = chat.add_message(prompt_input("\n> ").unwrap());
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
/// /// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// let model = Llama::new_chat().await.unwrap();
/// // Create constraints that parses Yes! and then stops on the end of the assistant's response
/// let constraints = LiteralParser::new("Yes!")
///     .then(model.default_assistant_constraints().unwrap());
/// // Create a chat session with the model and the constraints
/// let mut chat = Chat::builder(model)
///     .with_system_prompt(character_description)
///     .constrain_response(move |_history| constraints.clone())
///     .build();
///
/// // Chat with the user
/// loop {
///     let output_stream = chat.add_message(prompt_input("\n> ").unwrap());
///     output_stream.to_std_out().await.unwrap();
/// }
/// # }
/// ```
pub struct Chat {
    sender: tokio::sync::mpsc::UnboundedSender<Message>,
    shared_history: Arc<RwLock<Vec<ChatHistoryItem>>>,
}

impl Chat {
    /// Creates a new builder for a chat session.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// // Before you create a chat session, you need to create a model. Llama::new_chat will create a good default chat model.
    /// let model = Llama::new_chat().await.unwrap();
    /// // If you don't need to customize the chat session, you can use the `new` method to create a chat session with the default settings
    /// let mut chat = Chat::builder(model)
    ///     // The builder exposes methods for settings like the system prompt and constraints the bot response must follow
    ///     .with_system_prompt("The assistant will act like a pirate")
    ///     // Once you are done setting up the chat session, you can build it
    ///     .build();
    /// #}
    /// ```
    pub fn builder<M: Model>(model: M) -> ChatBuilder<M>
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        ChatBuilder::new(model)
    }

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
    /// #}
    /// ```
    pub fn new<M: Model>(model: M) -> Chat
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        Self::builder(model).build()
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
    /// let response_stream = chat.add_message(prompt);
    /// // And then read the stream as the response is generated
    /// while let Some(token) = response_stream.next().await {
    ///     print!("{token}");
    ///     std::io::stdout().flush().unwrap();
    /// }
    /// #}
    /// ```
    pub fn add_message(&mut self, message: impl ToString) -> ChannelTextStream {
        let (tx, rx) = unbounded_channel();

        let message = message.to_string();
        let message = message.trim().to_string();
        let _ = self.sender.send(Message::AddMessage {
            message,
            response_tx: tx,
        });
        ChannelTextStream::from(rx)
    }

    /// Saves the session to the given path.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut chat = Chat::new(Llama::new_chat().await.unwrap());
    /// let save_path = std::path::PathBuf::from("./chat.llama");
    /// chat.save_session(&save_path).await.unwrap();
    /// # }
    /// ```
    pub fn save_session(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> impl Future<Output = Result<()>> {
        let (tx, rx) = oneshot::channel();
        let result = self.sender.send(Message::SaveSession {
            path: path.as_ref().to_path_buf(),
            resolve: tx,
        });
        async move {
            result.map_err(|_| anyhow::anyhow!("Model stopped"))?;
            rx.await.map_err(|_| anyhow::anyhow!("Model stopped"))?
        }
    }

    /// Get the current chat history.
    ///
    /// # Example
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
    /// ```
    pub fn history(&self) -> Vec<ChatHistoryItem> {
        self.shared_history.read().unwrap().clone()
    }
}
