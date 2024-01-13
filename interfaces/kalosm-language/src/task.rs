//! A task interface that builds on top of [`kalosm_language_model::Model`]

use anyhow::Result;
use kalosm_language_model::ChatMarkers;
use kalosm_language_model::StructureParserResult;
use kalosm_language_model::{GenerationParameters, Model, ModelExt, SyncModel, SyncModelExt};
use kalosm_sample::{CreateParserState, Parser};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::{mpsc::unbounded_channel, oneshot};

/// A task session
struct TaskSession<Session> {
    cached_prompt: String,
    after_input: String,
    session: Option<Session>,
}

impl<Session: kalosm_language_model::Session> TaskSession<Session> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new [`TaskSession`].
    pub(crate) fn new(
        markers: Option<ChatMarkers>,
        system_prompt: String,
        examples: Vec<TaskExample>,
    ) -> Self {
        let (cached_prompt, after_input) = match markers {
            Some(markers) => {
                let mut cached_prompt = markers.system_prompt_marker.to_string() + &system_prompt;
                cached_prompt += markers.end_system_prompt_marker;

                for example in examples {
                    cached_prompt += markers.user_marker;
                    cached_prompt += &example.input;
                    cached_prompt += markers.end_user_marker;
                    cached_prompt += markers.assistant_marker;
                    cached_prompt += &example.output;
                    cached_prompt += markers.end_assistant_marker;
                }

                cached_prompt += markers.user_marker;
                (
                    cached_prompt,
                    markers.end_user_marker.to_string() + markers.assistant_marker,
                )
            }
            None => {
                let mut cached_prompt = "# Instruction\n".to_string();
                cached_prompt += &system_prompt;
                if !system_prompt.ends_with('\n') {
                    cached_prompt += "\n";
                }

                for example in examples {
                    cached_prompt += "# Input\n";
                    cached_prompt += &example.input;
                    if !example.input.ends_with('\n') {
                        cached_prompt += "\n";
                    }
                    cached_prompt += "# Output\n";
                    cached_prompt += &example.output;
                    if !example.output.ends_with('\n') {
                        cached_prompt += "\n";
                    }
                }

                cached_prompt += "# Input\n";
                (cached_prompt, "# Output\n".to_string())
            }
        };

        Self {
            cached_prompt,
            after_input,
            session: None,
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

    fn start_session(
        &mut self,
        message: &str,
        model: &mut impl SyncModel<Session = Session>,
    ) -> Result<Session> {
        let mut session = self.create_session(model)?;

        let prompt = message.to_string() + &self.after_input;

        // Feed the message to the model.
        model.feed_text(&mut session, &prompt, Some(0))?;

        Ok(session)
    }
}

#[derive(Debug)]
struct TaskExample {
    input: String,
    output: String,
}

/// A marker for no parser.
pub struct NoParser;

/// A builder for [`Task`].
pub struct TaskBuilder<'a, M: Model, P = NoParser> {
    model: &'a mut M,
    system_prompt: String,
    sampler: Arc<std::sync::Mutex<dyn Sampler + Send + Sync>>,
    constraints: P,
    examples: Vec<TaskExample>,
}

impl<'a, M: Model> TaskBuilder<'a, M> {
    fn new(model: &'a mut M, description: impl Into<String>) -> TaskBuilder<'a, M> {
        TaskBuilder {
            model,
            system_prompt: description.into(),
            sampler: Arc::new(std::sync::Mutex::new(
                GenerationParameters::default().sampler(),
            )),
            constraints: NoParser,
            examples: Vec::new(),
        }
    }
}

impl<'a, M: Model, P: TaskBuilderReturn + Send + Sync + 'static> TaskBuilder<'a, M, P>
where
    <M::SyncModel as SyncModel>::Session: Send,
{
    /// Sets the [`Sampler`] to use for generating responses.
    pub fn with_sampler(mut self, sampler: impl Sampler + Send + Sync + 'static) -> Self {
        self.sampler = Arc::new(std::sync::Mutex::new(sampler));
        self
    }

    /// Set the constraints for the task. The response generated by the model will follow the constraints.
    pub fn with_constraints<Parser, Builder: FnMut() -> Parser + 'static>(
        self,
        constraints: Builder,
    ) -> TaskBuilder<'a, M, Builder> {
        TaskBuilder {
            constraints,
            model: self.model,
            system_prompt: self.system_prompt,
            sampler: self.sampler,
            examples: self.examples,
        }
    }

    /// Add an example to the task.
    pub fn with_example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        let input = input.into();
        let output = output.into();
        self.examples.push(TaskExample { input, output });
        self
    }

    /// Add multiple examples to the task.
    pub fn with_examples(
        mut self,
        examples: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        for (input, output) in examples {
            let input = input.into();
            let output = output.into();
            self.examples.push(TaskExample { input, output });
        }
        self
    }

    /// Build a [`Task`] from a [`TaskBuilder`].
    pub fn build(self) -> <P as TaskBuilderReturn>::Output {
        <P as TaskBuilderReturn>::build(self)
    }
}

/// A trait for returning the output of a [`TaskBuilder`].
pub trait TaskBuilderReturn {
    /// The output of the [`TaskBuilder`].
    type Output;

    /// Build the output of the [`TaskBuilder`].
    fn build<M: Model>(task_builder: TaskBuilder<M, Self>) -> Self::Output
    where
        <M::SyncModel as SyncModel>::Session: Send,
        Self: Sized;
}

impl TaskBuilderReturn for NoParser {
    type Output = Task<ChannelTextStream<String>>;

    fn build<M: Model>(task_builder: TaskBuilder<M, Self>) -> Self::Output {
        let TaskBuilder {
            model,
            system_prompt,
            sampler,
            examples,
            ..
        } = task_builder;
        let chat_markers = model.chat_markers();
        let stop_on = chat_markers
            .as_ref()
            .map(|m| m.end_assistant_marker.to_string())
            .unwrap_or_else(|| "# Input".to_string());
        let (sender_tx, mut sender_rx) = unbounded_channel::<String>();
        let (result_tx, result_rx) = unbounded_channel();
        model
            .run_sync(move |model| {
                Box::pin(async move {
                    let mut session = TaskSession::new(chat_markers, system_prompt, examples);

                    while let Some(message) = sender_rx.recv().await {
                        let (tx, rx) = unbounded_channel();
                        result_tx.send(rx.into()).unwrap();

                        let mut session = match session.start_session(message.as_str(), model) {
                            Ok(session) => session,
                            Err(err) => {
                                tracing::error!("Failed to start session: {}", err);
                                return;
                            }
                        };
                        let on_token = |tok: String| {
                            tx.send(tok)?;
                            Ok(kalosm_language_model::ModelFeedback::Continue)
                        };
                        if let Err(err) = model.stream_text_with_sampler(
                            &mut session,
                            &message,
                            None,
                            Some(&stop_on),
                            sampler.clone(),
                            on_token,
                        ) {
                            tracing::error!("Failed to stream text: {}", err);
                            return;
                        }
                    }
                })
            })
            .unwrap();

        Task {
            sender: sender_tx,
            channel: Mutex::new(result_rx),
        }
    }
}

impl<P: Parser + CreateParserState, B: FnMut() -> P + Send + 'static> TaskBuilderReturn for B
where
    <P as Parser>::Output: Send + 'static,
    <P as Parser>::PartialState: Sync + Send,
{
    type Output = Task<StructureParserResult<ChannelTextStream<String>, P::Output>>;

    fn build<M: Model>(task_builder: TaskBuilder<M, Self>) -> Self::Output {
        let TaskBuilder {
            model,
            system_prompt,
            sampler,
            mut constraints,
            examples,
        } = task_builder;

        // check if the examples are valid
        #[cfg(debug_assertions)]
        {
            let parser = constraints();
            for example in &examples {
                let state = parser.create_parser_state();
                let result = parser.parse(&state, example.output.as_bytes());
                if result.is_err() {
                    tracing::error!("Example: {:?} does not fit the constraints you provided to the task. Examples tend to perform better when they follow the same format as the model output.", example);
                }
            }
        }

        let chat_markers = model.chat_markers();
        let (sender_tx, mut sender_rx) = unbounded_channel::<String>();
        let (result_tx, result_rx) = unbounded_channel();
        model
            .run_sync(move |model| {
                Box::pin(async move {
                    let mut session = TaskSession::new(chat_markers, system_prompt, examples);

                    let span = tracing::span!(tracing::Level::TRACE, "Task session");
                    let _span = span.enter();

                    while let Some(message) = sender_rx.recv().await {
                        let (tx, rx) = unbounded_channel();
                        let (parsed_tx, parsed_rx) = oneshot::channel();
                        result_tx
                            .send(StructureParserResult::new(rx.into(), parsed_rx))
                            .unwrap();

                        let mut session = match session.start_session(message.as_str(), model) {
                            Ok(session) => session,
                            Err(err) => {
                                tracing::error!("Failed to start session: {}", err);
                                return;
                            }
                        };

                        let constraints = constraints();
                        let state = constraints.create_parser_state();
                        let on_token = |tok: String| {
                            tracing::trace!("Task generated token: {}", tok);
                            tx.send(tok)?;
                            Ok(())
                        };
                        let result = model.generate_structured(
                            &mut session,
                            &message,
                            constraints,
                            state,
                            sampler.clone(),
                            on_token,
                        );
                        if parsed_tx.send(result).is_err() {
                            tracing::error!("Failed to send parsed result");
                            return;
                        }
                    }
                })
            })
            .unwrap();

        Task {
            sender: sender_tx,
            channel: Mutex::new(result_rx),
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
pub struct Task<R = ChannelTextStream<String>> {
    sender: tokio::sync::mpsc::UnboundedSender<String>,
    channel: Mutex<tokio::sync::mpsc::UnboundedReceiver<R>>,
}

impl Task<ChannelTextStream<String>> {
    /// Create a new task with no constraints and the default sampler. See [`Task::builder`] for more options.
    pub fn new<M: Model>(model: &mut M, description: impl Into<String>) -> Self
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        Self::builder(model, description).build()
    }
}

impl Task {
    /// Creates a new builder for a task session.
    pub fn builder<M: Model>(model: &mut M, description: impl Into<String>) -> TaskBuilder<'_, M>
    where
        <M::SyncModel as SyncModel>::Session: Send,
    {
        TaskBuilder::new(model, description)
    }
}

impl<R> Task<R> {
    /// Run the task with a message.
    pub async fn run(&self, message: impl Into<String>) -> Result<R> {
        let message = message.into();
        let message = message.trim().to_string();
        self.sender
            .send(message)
            .map_err(|_| anyhow::anyhow!("Model stopped"))?;
        self.channel
            .lock()
            .await
            .recv()
            .await
            .ok_or(anyhow::anyhow!("Model stopped"))
    }
}
