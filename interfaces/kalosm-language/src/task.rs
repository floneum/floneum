//! A chat interface that builds on top of [`kalosm_language_model::ChatModel`]

use kalosm_language_model::Session;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use kalosm_language_model::{ChatModel, GenerationParameters, ModelExt, SyncModel, SyncModelExt};
use kalosm_sample::{ArcParser, CreateParserState, ParserExt};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokio::sync::mpsc::unbounded_channel;

/// A task session
struct TaskSession<Session> {
    user_marker: String,
    end_user_marker: String,
    assistant_marker: String,
    end_assistant_marker: String,
    session: Session,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
}

impl<Session> TaskSession<Session> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new [`TaskSession`].
    pub(crate) fn new<Model: SyncModel<Session = Session>>(
        model: &mut Model,
        system_prompt_marker: String,
        end_system_prompt_marker: String,
        user_marker: String,
        end_user_marker: String,
        assistant_marker: String,
        end_assistant_marker: String,
        system_prompt: String,
        sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    ) -> Self {
        let session = session.unwrap_or_else(|| model.new_session().unwrap());

        Self {
            user_marker,
            end_user_marker,
            assistant_marker,
            end_assistant_marker,
            session,
            sampler,
        }
    }

    /// Adds a message to the history.
    pub fn add_message<Model: SyncModel<Session = Session>>(
        &mut self,
        message: String,
        model: &mut Model,
        stream: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let mut bot_response = String::new();
        match constraints {
                Some(constraints) => {
                    let mut constraints = constraints.lock().unwrap();
                    let constraints = constraints(&self.history, model);
                    let state = constraints.create_parser_state();
                    let on_token = |tok: String| {
                        let tok = tok
                            .strip_suffix(&self.end_assistant_marker)
                            .unwrap_or(&tok)
                            .to_string();
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
                        let tok = tok
                            .strip_suffix(&self.end_assistant_marker)
                            .unwrap_or(&tok)
                            .to_string();
                        bot_response += &tok;
                        stream.send(tok)?;
                        Ok(kalosm_language_model::ModelFeedback::Continue)
                    };
                    model.stream_text_with_sampler(
                        &mut self.session,
                        &prompt,
                        None,
                        Some(&self.end_assistant_marker),
                        self.sampler.clone(),
                        on_token,
                    )?;
                }
        }

        Ok(())
    }
}

/// A builder for [`Task`].
pub struct ChatBuilder<'a, M: ChatModel> {
    model: &'a mut M,
    session: Option<<M::SyncModel as kalosm_language_model::SyncModel>::Session>,
    system_prompt: String,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    
}

impl<'a, M: ChatModel> ChatBuilder<'a, M> {
    fn new(model: &'a mut M) -> ChatBuilder<M> {
        ChatBuilder {
            model,
            session: None,
            system_prompt: "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.".into(),
            sampler: Arc::new(Mutex::new(GenerationParameters::default().sampler())),
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


    /// Starts the chat instance with the given session. This can be useful for resuming a chat session.
    pub fn with_session(
        mut self,
        session: <M::SyncModel as kalosm_language_model::SyncModel>::Session,
    ) -> Self {
        self.session = Some(session);
        self
    }

    /// Starts the chat instance with a session from the given path. This can be useful for resuming a chat session.
    pub fn with_session_path(self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Ok(self.with_session(
            <M::SyncModel as kalosm_language_model::SyncModel>::Session::load_from(path)?,
        ))
    }

    /// Builds a [`Task`] instance.
    pub fn build(self) -> Task
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        let Self {
            model,
            system_prompt,
            sampler,
            session
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
                    let mut session = TaskSession::new(
                        model,
                        system_prompt_marker,
                        end_system_prompt_marker,
                        user_marker,
                        end_user_marker,
                        assistant_marker,
                        end_assistant_marker,
                        system_prompt,
                        sampler,
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

        Task {
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
pub struct Task {
    sender: tokio::sync::mpsc::UnboundedSender<Message>,
    channel: tokio::sync::mpsc::UnboundedReceiver<Response>,
}

impl Task {
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
        self.sender
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
