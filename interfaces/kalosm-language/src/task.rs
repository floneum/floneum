//! A task interface that builds on top of [`kalosm_language_model::ChatModel`]

use std::sync::{Arc, Mutex};

use anyhow::Result;
use kalosm_language_model::{ChatModel, GenerationParameters, ModelExt, SyncModel, SyncModelExt};
use kalosm_sample::{ArcParser, CreateParserState};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokio::sync::mpsc::unbounded_channel;

/// A task session
struct TaskSession<Session> {
    assistant_marker: String,
    end_assistant_marker: String,
    end_user_marker: String,
    cached_prompt: String,
    session: Option<Session>,
    constraints: Option<Arc<Mutex<dyn Fn() -> ArcParser + Send + Sync>>>,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
}

impl<Session: kalosm_language_model::Session> TaskSession<Session> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new [`TaskSession`].
    pub(crate) fn new(
        system_prompt_marker: String,
        end_system_prompt_marker: String,
        user_marker: String,
        end_user_marker: String,
        assistant_marker: String,
        end_assistant_marker: String,
        system_prompt: String,
        constraints: Option<Arc<Mutex<dyn Fn() -> ArcParser + Send + Sync>>>,
        sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    ) -> Self {
        Self {
            assistant_marker,
            end_assistant_marker,
            end_user_marker,
            cached_prompt: format!(
                "{}{}{}{}",
                system_prompt_marker, system_prompt, end_system_prompt_marker, user_marker
            ),
            session: None,
            constraints,
            sampler,
        }
    }

    fn create_new_session(
        &mut self,
        model: &mut impl SyncModel<Session = Session>,
    ) -> Result<Session> {
        let mut session = model.new_session()?;
        model.feed_text(&mut session, &self.cached_prompt, Some(0))?;

        self.session = session.try_clone().ok();

        Ok(session)
    }

    /// Create a session with the task's system prompt.
    fn create_session(&mut self, model: &mut impl SyncModel<Session = Session>) -> Result<Session> {
        match &self.session {
            Some(cache) => match cache.try_clone() {
                Ok(cache) => Ok(cache),
                Err(err) => {
                    tracing::error!("Failed to clone session: {}", err);
                    Ok(self.create_new_session(model)?)
                }
            },
            None => Ok(self.create_new_session(model)?),
        }
    }

    /// Run the task with a message.
    pub fn run<Model: SyncModel<Session = Session>>(
        &mut self,
        message: String,
        model: &mut Model,
        stream: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let mut bot_response = String::new();
        let mut session = self.create_session(model)?;
        let Self {
            end_user_marker,
            constraints,
            sampler,
            ..
        } = self;

        // Feed the message to the model.
        model.feed_text(
            &mut session,
            &format!("{}{}{}", message, end_user_marker, self.assistant_marker),
            Some(0)
        )?;

        // Generate a response.
        match constraints {
            Some(constraints) => {
                let constraints = constraints.lock().unwrap();
                let constraints = constraints();
                let state = constraints.create_parser_state();
                let on_token = |tok: String| {
                    bot_response += &tok;
                    stream.send(tok)?;
                    Ok(())
                };
                model.generate_structured(
                    &mut session,
                    &message,
                    constraints,
                    state,
                    sampler.clone(),
                    on_token,
                )?;
            }
            None => {
                let on_token = |tok: String| {
                    bot_response += &tok;
                    stream.send(tok)?;
                    Ok(kalosm_language_model::ModelFeedback::Continue)
                };
                model.stream_text_with_sampler(
                    &mut session,
                    &message,
                    None,
                    Some(&self.end_assistant_marker),
                    sampler.clone(),
                    on_token,
                )?;
            }
        }

        Ok(())
    }
}

/// A builder for [`Task`].
pub struct TaskBuilder<'a, M: ChatModel> {
    model: &'a mut M,
    system_prompt: String,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    constraints: Option<Arc<Mutex<dyn Fn() -> ArcParser + Send + Sync>>>,
}

impl<'a, M: ChatModel> TaskBuilder<'a, M> {
    fn new(model: &'a mut M, description: impl Into<String>) -> TaskBuilder<M> {
        TaskBuilder {
            model,
            system_prompt: description.into(),
            sampler: Arc::new(Mutex::new(GenerationParameters::default().sampler())),
            constraints: None,
        }
    }

    /// Sets the [`Sampler`] to use for generating responses.
    pub fn with_sampler(mut self, sampler: impl Sampler + Send + Sync + 'static) -> Self {
        self.sampler = Arc::new(Mutex::new(sampler));
        self
    }

    /// Set the constraints for the task. The response generated by the model will follow the constraints.
    pub fn with_constraints(
        mut self,
        constraints: impl Fn() -> ArcParser + Send + Sync + 'static,
    ) -> Self {
        self.constraints = Some(Arc::new(Mutex::new(constraints)));
        self
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
            constraints,
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
                        system_prompt_marker,
                        end_system_prompt_marker,
                        user_marker,
                        end_user_marker,
                        assistant_marker,
                        end_assistant_marker,
                        system_prompt,
                        constraints,
                        sampler,
                    );

                    while let Some(message) = sender_rx.recv().await {
                        let (tx, rx) = unbounded_channel();
                        result_tx.send(rx.into()).unwrap();
                        session.run(message, model, tx).unwrap();
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

/// A task session lets you efficiently run a task with a model. The task session will reuse the model's cache to avoid re-feeding the task prompt repeatedly.
///
/// # Example
/// ```rust
/// use kalosm_language::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let mut llm = Llama::new_chat();
///     let mut task = Task::new(&mut llm, "You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.");
///
///     println!("question 1");
///     // The first time we use the task, it will load the model and prompt.
///     task.run("What is 2 + 2?")
///         .await
///         .unwrap()
///         .to_std_out()
///         .await
///         .unwrap();
///     
///     println!("question 2");
///     // After the first time, the model and prompt are cached.
///     task.run("What is 4 + 4?")
///         .await
///         .unwrap()
///         .to_std_out()
///         .await
///         .unwrap();
/// }
/// ```
pub struct Task {
    sender: tokio::sync::mpsc::UnboundedSender<String>,
    channel: tokio::sync::mpsc::UnboundedReceiver<ChannelTextStream<String>>,
}

impl Task {
    /// Create a new task with no constraints and the default sampler. See [`Task::builder`] for more options.
    pub fn new<M: ChatModel>(model: &mut M, description: impl Into<String>) -> Self
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        Self::builder(model, description).build()
    }

    /// Creates a new builder for a task session.
    pub fn builder<M: ChatModel>(
        model: &mut M,
        description: impl Into<String>,
    ) -> TaskBuilder<'_, M>
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        TaskBuilder::new(model, description)
    }

    /// Run the task with a message.
    pub async fn run(&mut self, message: impl Into<String>) -> Result<ChannelTextStream<String>> {
        let message = message.into();
        let message = message.trim().to_string();
        self.sender
            .send(message)
            .map_err(|_| anyhow::anyhow!("Model stopped"))?;
        self.channel
            .recv()
            .await
            .ok_or(anyhow::anyhow!("Model stopped"))
    }
}
