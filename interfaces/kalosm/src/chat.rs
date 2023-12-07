use kalosm_language::Session;
use std::{
    fmt::Display,
    sync::{Arc, Mutex}, path::PathBuf,
};

use anyhow::Result;
use kalosm_language::{
    kalosm_sample::{ArcParser, CreateParserState, ParserExt},
    ChatModel, GenerationParameters, ModelExt, SyncModel, SyncModelExt,
};
use kalosm_streams::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokio::sync::mpsc::unbounded_channel;

type MessageFilter<M> =
    Arc<Mutex<Box<dyn for<'a> FnMut(&'a str, &mut M) -> Option<&'a str> + Send + Sync>>>;
type UserMessageMapping<M> = Arc<Mutex<Box<dyn FnMut(&str, &mut M) -> String + Send + Sync>>>;
type ResponseConstraintGenerator<M> =
    Arc<Mutex<Box<dyn FnMut(&[ChatHistoryItem], &mut M) -> ArcParser + Send + Sync>>>;

/// A simple helper function for prompting the user for input.
pub fn prompt_input(prompt: impl Display) -> Result<String> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
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
pub struct ChatHistoryItem {
    ty: MessageType,
    contents: String,
}

impl ChatHistoryItem {
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
struct ChatSession<Session, Model: SyncModel<Session = Session>> {
    user_marker: String,
    end_user_marker: String,
    assistant_marker: String,
    end_assistant_marker: String,
    history: Vec<ChatHistoryItem>,
    session: Session,
    unfed_text: String,
    eos: String,
    map_user_message_prompt: Option<UserMessageMapping<Model>>,
    bot_constraints: Option<ResponseConstraintGenerator<Model>>,
    filter_map_bot_response: Option<MessageFilter<Model>>,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
}

impl<Session, Model: SyncModel<Session = Session>> ChatSession<Session, Model> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    pub(crate) fn new(
        model: &Model,
        system_prompt_marker: String,
        end_system_prompt_marker: String,
        user_marker: String,
        end_user_marker: String,
        assistant_marker: String,
        end_assistant_marker: String,
        system_prompt: String,
        map_user_message_prompt: Option<UserMessageMapping<Model>>,
        bot_constraints: Option<ResponseConstraintGenerator<Model>>,
        filter_map_bot_response: Option<MessageFilter<Model>>,
        sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
        session: Option<Session>,
    ) -> Self {
        let (unfed_text, session) = match session {
            Some(session) => (String::new(), session),
            None => {
                let session = model.new_session().unwrap();
                let mut unfed_text = String::new();
                unfed_text += &system_prompt_marker;
                unfed_text += &system_prompt;
                unfed_text += &end_system_prompt_marker;
                (unfed_text, session)
            }
        };
        let history = vec![ChatHistoryItem {
            ty: MessageType::SystemPrompt,
            contents: system_prompt,
        }];

        Self {
            user_marker,
            end_user_marker,
            assistant_marker,
            end_assistant_marker,
            eos: model
                .tokenizer()
                .decode(&[model.stop_token().unwrap()])
                .unwrap()
                .to_string(),
            session,
            unfed_text,
            history,
            map_user_message_prompt,
            bot_constraints,
            filter_map_bot_response,
            sampler,
        }
    }

    /// Adds a message to the history.
    pub fn add_message(
        &mut self,
        message: String,
        model: &mut Model,
        stream: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        match &self.map_user_message_prompt {
            Some(map) => {
                let mut map = map.lock().unwrap();
                let message = map(&message, model);
                let new_text = format!("{}{}{}", self.user_marker, message, self.end_user_marker);
                self.unfed_text += &new_text;
            }
            None => {
                let new_text = format!("{}{}{}", self.user_marker, message, self.end_user_marker);
                self.unfed_text += &new_text;
            }
        };
        self.history.push(ChatHistoryItem {
            ty: MessageType::UserMessage,
            contents: message,
        });
        let mut bot_response = String::new();
        self.unfed_text += &self.assistant_marker;
        let prompt = std::mem::take(&mut self.unfed_text);
        let bot_constraints = &self.bot_constraints;
        match &self.filter_map_bot_response {
            Some(filter) => {
                let mut filter = filter.lock().unwrap();
                loop {
                    bot_response.clear();

                    match bot_constraints {
                        Some(constraints) => {
                            let mut constraints = constraints.lock().unwrap();
                            let constraints = constraints(&self.history, model);
                            let state = constraints.create_parser_state();
                            let on_token = |tok: String| {
                                bot_response += &tok;
                                Ok(())
                            };
                            model.generate_structured(
                                &mut self.session,
                                &prompt,
                                constraints,
                                state,
                                self.sampler.clone(),
                                on_token,
                            )?;
                        }
                        None => {
                            let on_token = |tok: String| {
                                bot_response += &tok;
                                Ok(kalosm_language::ModelFeedback::Continue)
                            };
                            model.stream_text_with_sampler(
                                &mut self.session,
                                &prompt,
                                None,
                                Some(&self.eos),
                                self.sampler.clone(),
                                on_token,
                            )?;
                        }
                    }

                    if let Some(bot_response) = filter(&bot_response, model) {
                        stream.send(bot_response.to_string())?;
                        break;
                    } else {
                        tracing::trace!("Filtered out: {}", bot_response);
                    }
                }
            }
            None => match bot_constraints {
                Some(constraints) => {
                    let mut constraints = constraints.lock().unwrap();
                    let constraints = constraints(&self.history, model);
                    let state = constraints.create_parser_state();
                    let on_token = |tok: String| {
                        bot_response += &tok;
                        stream.send(tok)?;
                        Ok(())
                    };
                    model.generate_structured(
                        &mut self.session,
                        &prompt,
                        constraints,
                        state,
                        self.sampler.clone(),
                        on_token,
                    )?;
                }
                None => {
                    let on_token = |tok: String| {
                        bot_response += &tok;
                        stream.send(tok)?;
                        Ok(kalosm_language::ModelFeedback::Continue)
                    };
                    model.stream_text_with_sampler(
                        &mut self.session,
                        &prompt,
                        None,
                        Some(&self.eos),
                        self.sampler.clone(),
                        on_token,
                    )?;
                }
            },
        }

        self.unfed_text += &self.end_assistant_marker;
        self.history.push(ChatHistoryItem {
            ty: MessageType::ModelAnswer,
            contents: bot_response,
        });
        Ok(())
    }
}

/// A builder for [`Chat`].
pub struct ChatBuilder<'a, M: ChatModel> {
    model: &'a mut M,
    session: Option<<M::SyncModel as kalosm_language::SyncModel>::Session>,
    system_prompt: String,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    map_user_message_prompt: Option<UserMessageMapping<M::SyncModel>>,
    bot_constraints: Option<ResponseConstraintGenerator<M::SyncModel>>,
    filter_map_bot_response: Option<MessageFilter<M::SyncModel>>,
}

impl<'a, M: ChatModel> ChatBuilder<'a, M> {
    fn new(model: &'a mut M) -> ChatBuilder<M> {
        ChatBuilder {
            model,
            session: None,
            system_prompt: "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.".into(),
            sampler: Arc::new(Mutex::new(GenerationParameters::default().sampler())),
            map_user_message_prompt: None,
            bot_constraints: None,
            filter_map_bot_response: None,
        }
    }

    /// Adds a system prompt to the chat. The system prompt guides the model to respond in a certain way.
    /// If no system prompt is added, the model will use a default system prompt that instructs the model to respond in a way that is safe and respectful.
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Sets the [`Sampler`] to use for generating responses.
    pub fn with_sampler(mut self, sampler: impl Sampler + Send + Sync + 'static) -> Self {
        self.sampler = Arc::new(Mutex::new(sampler));
        self
    }

    /// Filters out bot responses that do not match the given filter.
    ///
    /// > **Note**: This setting will disable streaming responses.
    pub fn filter_bot_response(
        self,
        mut filter: impl FnMut(&str, &mut M::SyncModel) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.filter_map_bot_response(move |message, model| {
            (filter(message, model)).then_some(message)
        })
    }

    /// Map user prompts before they are sent to the model. This will not effect the chat history.
    pub fn map_user_message_prompt(
        mut self,
        map_user_message_prompt: impl FnMut(&str, &mut M::SyncModel) -> String + Send + Sync + 'static,
    ) -> Self {
        self.map_user_message_prompt =
            Some(Arc::new(Mutex::new(Box::new(map_user_message_prompt)
                as Box<
                    dyn FnMut(&str, &mut M::SyncModel) -> String + Send + Sync,
                >)));
        self
    }

    /// Constrains the model's response to the given parser. This can be used to make the model start with a certain phrase, or to make the model respond in a certain way.
    pub fn constrain_response<P>(
        mut self,
        mut bot_constraints: impl FnMut(&[ChatHistoryItem], &mut M::SyncModel) -> P
            + Send
            + Sync
            + 'static,
    ) -> Self
    where
        P: CreateParserState + Sized + Send + Sync + 'static,
        P::Error: std::error::Error + Send + Sync + 'static,
        P::Output: Send + Sync + 'static,
        P::PartialState: Send + Sync + 'static,
    {
        self.bot_constraints = Some(Arc::new(Mutex::new(Box::new(
            move |history: &[ChatHistoryItem], model: &mut M::SyncModel| {
                bot_constraints(history, model).boxed()
            },
        )
            as Box<
                dyn FnMut(&[ChatHistoryItem], &mut M::SyncModel) -> ArcParser + Send + Sync,
            >)));
        self
    }

    /// Filters out bot responses that do not match the given filter, and maps the bot response before it is sent to the stream.
    ///
    /// > **Note**: This setting will disable streaming responses.
    pub fn filter_map_bot_response(
        mut self,
        filter_map_bot_response: impl for<'b> FnMut(&'b str, &mut M::SyncModel) -> Option<&'b str>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.filter_map_bot_response =
            Some(Arc::new(Mutex::new(Box::new(filter_map_bot_response)
                as Box<
                    dyn for<'b> FnMut(&'b str, &mut M::SyncModel) -> Option<&'b str> + Send + Sync,
                >)));
        self
    }

    /// Maps the bot response before it is sent to the stream.
    ///
    /// > **Note**: This setting will disable streaming responses.
    pub fn map_bot_response(
        self,
        mut map_bot_response: impl for<'b> FnMut(&'b str, &mut M::SyncModel) -> &'b str
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.filter_map_bot_response(move |message, model| Some(map_bot_response(message, model)))
    }

    /// Starts the chat instance with the given session. This can be useful for resuming a chat session.
    pub fn with_session(
        mut self,
        session: <M::SyncModel as kalosm_language::SyncModel>::Session,
    ) -> Self {
        self.session = Some(session);
        self
    }

    /// Starts the chat instance with a session from the given path. This can be useful for resuming a chat session.
    pub fn with_session_path(self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Ok(self
            .with_session(<M::SyncModel as kalosm_language::SyncModel>::Session::load_from(path)?))
    }

    /// Builds a [`Chat`] instance.
    pub fn build(self) -> Chat
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        let Self {
            model,
            system_prompt,
            sampler,
            map_user_message_prompt,
            bot_constraints,
            filter_map_bot_response,
            session,
        } = self;
        let system_prompt_marker = model.system_prompt_marker().to_string();
        let end_system_prompt_marker = model.end_system_prompt_marker().to_string();
        let user_marker = model.user_marker().to_string();
        let end_user_marker = model.end_user_marker().to_string();
        let assistant_marker = model.assistant_marker().to_string();
        let end_assistant_marker = model.end_assistant_marker().to_string();
        let (sender_tx, mut sender_rx) = unbounded_channel();
        let (result_tx, result_rx) = unbounded_channel();
        model
            .run_sync(move |model| {
                Box::pin(async move {
                    let mut session = ChatSession::new(
                        model,
                        system_prompt_marker,
                        end_system_prompt_marker,
                        user_marker,
                        end_user_marker,
                        assistant_marker,
                        end_assistant_marker,
                        system_prompt,
                        map_user_message_prompt,
                        bot_constraints,
                        filter_map_bot_response,
                        sampler,
                        session,
                    );

                    while let Some(message) = sender_rx.recv().await {
                        match message {
                            Message::AddMessage(message) => {
                                let (tx, rx) = unbounded_channel();
                                result_tx.send(Response::AddMessage(rx.into())).unwrap();
                                session.add_message(message, model, tx).unwrap();
                            }
                            Message::SaveSession(path) => {
                                session.session.save_to(path).unwrap();
                                result_tx.send(Response::SaveSession).unwrap();
                            }
                        }
                    }
                })
            })
            .unwrap();

        Chat {
            sender: sender_tx,
            channel: result_rx,
        }
    }
}


enum Message {
    AddMessage(String),
    SaveSession(PathBuf),
}

enum Response {
    AddMessage(ChannelTextStream<String>),
    SaveSession,
}

/// A chat session.
pub struct Chat {
    sender: tokio::sync::mpsc::UnboundedSender<Message>,
    channel: tokio::sync::mpsc::UnboundedReceiver<Response>,
}

impl Chat {
    /// Creates a new builder for a chat session.
    pub fn builder<M: ChatModel>(model: &mut M) -> ChatBuilder<'_, M> {
        ChatBuilder::new(model)
    }

    /// Adds a message to the history.
    pub async fn add_message(
        &mut self,
        message: impl Into<String>,
    ) -> Result<ChannelTextStream<String>> {
        let message = message.into();
        let message = message.trim().to_string();
        self.sender
            .send(Message::AddMessage(message))
            .map_err(|_| anyhow::anyhow!("Model stopped"))?;
        self.channel
            .recv()
            .await
            .map(|c| match c {
                Response::AddMessage(c) => c,
                _ => unreachable!(),
            })
            .ok_or(anyhow::anyhow!("Model stopped"))
    }

    /// Saves the session to the given path.
    pub async fn save_session(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self
            .sender
            .send(Message::SaveSession(path.as_ref().to_path_buf()))
            .map_err(|_| anyhow::anyhow!("Model stopped"))?;
        self.channel
            .recv()
            .await
            .map(|c| match c {
                Response::SaveSession => (),
                _ => unreachable!(),
            })
            .ok_or(anyhow::anyhow!("Model stopped"))?;
        Ok(())
    }
}
