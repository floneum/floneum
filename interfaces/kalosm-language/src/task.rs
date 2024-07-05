//! A task interface that builds on top of [`kalosm_language_model::Model`]

use anyhow::Result;
use futures_util::Stream;
use kalosm_language_model::ChatMarkers;
use kalosm_language_model::Session;
use kalosm_language_model::StructureParserResult;
use kalosm_language_model::{GenerationParameters, Model, ModelExt, SyncModel, SyncModelExt};
use kalosm_sample::{CreateParserState, Parser};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use rustc_hash::FxHashMap;
use std::any::Any;
use std::any::TypeId;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::sync::{mpsc::unbounded_channel, oneshot};

struct TaskSessionEntry<S> {
    cached_prompt: String,
    after_input: String,
    session: Option<S>,
}

impl<S: Session> TaskSessionEntry<S> {
    pub(crate) fn new(
        markers: Option<ChatMarkers>,
        system_prompt: String,
        examples: &[TaskExample],
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

    fn create_new_session(&mut self, model: &mut impl SyncModel<Session = S>) -> Result<S> {
        let mut session = model.new_session()?;
        model.feed_text(&mut session, &self.cached_prompt)?;

        self.session = session.try_clone().ok();

        Ok(session)
    }

    /// Create a session with the task's system prompt.
    fn create_session(&mut self, model: &mut impl SyncModel<Session = S>) -> Result<S> {
        let read = &self.session;
        match read {
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

    fn task_prompt(&self, message: &str) -> String {
        message.to_string() + &self.after_input
    }
}

/// A task session
struct TaskSessions {
    sessions: RwLock<FxHashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    system_prompt: String,
    examples: Vec<TaskExample>,
}

impl TaskSessions {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new [`TaskSessions`].
    pub(crate) fn new(system_prompt: String, examples: Vec<TaskExample>) -> Self {
        Self {
            sessions: RwLock::new(FxHashMap::default()),
            system_prompt,
            examples,
        }
    }
}

#[derive(Debug, Clone)]
struct TaskExample {
    input: String,
    output: String,
}

/// A marker for no parser.
#[derive(Debug, Clone)]
pub struct NoParser;

/// A builder for [`Task`].
#[derive(Debug, Clone)]
pub struct TaskBuilder<P = NoParser> {
    system_prompt: String,
    sampler: Arc<std::sync::Mutex<dyn Sampler + Send + Sync>>,
    constraints: P,
    examples: Vec<TaskExample>,
}

impl TaskBuilder {
    fn new(description: impl Into<String>) -> TaskBuilder {
        TaskBuilder {
            system_prompt: description.into(),
            sampler: Arc::new(std::sync::Mutex::new(
                GenerationParameters::default().sampler(),
            )),
            constraints: NoParser,
            examples: Vec::new(),
        }
    }
}

impl<P: TaskBuilderReturn + Send + Sync + 'static> TaskBuilder<P> {
    /// Sets the [`Sampler`] to use for generating responses.
    pub fn with_sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Arc::new(std::sync::Mutex::new(sampler));
        self
    }

    /// Set the constraints for the task. The response generated by the model will follow the constraints.
    pub fn with_constraints<Parser>(self, constraints: Parser) -> TaskBuilder<Parser>
    where
        Parser: kalosm_sample::Parser + CreateParserState + Sync + Send + 'static,
    {
        TaskBuilder {
            constraints,
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
    pub fn build(self) -> Task<<P as TaskBuilderReturn>::Output> {
        let inner = <P as TaskBuilderReturn>::build(self);
        Task { runner: inner }
    }
}

/// A trait for returning the output of a [`TaskBuilder`].
pub trait TaskBuilderReturn
where
    Self: Sized,
{
    /// The output of the [`TaskBuilder`].
    type Output: TaskRunner;

    /// Build the output of the [`TaskBuilder`].
    fn build(task_builder: TaskBuilder<Self>) -> Self::Output;
}

impl TaskBuilderReturn for NoParser {
    type Output = UnstructuredRunner;

    fn build(task_builder: TaskBuilder<Self>) -> Self::Output {
        let TaskBuilder {
            system_prompt,
            sampler,
            examples,
            ..
        } = task_builder;

        let sessions = TaskSessions::new(system_prompt.clone(), examples.clone());
        UnstructuredRunner {
            sessions: Arc::new(sessions),
            sampler,
        }
    }
}

/// A task runner for a task that does not follow constraints.
pub struct UnstructuredRunner {
    sessions: Arc<TaskSessions>,
    sampler: Arc<std::sync::Mutex<dyn Sampler + Send + Sync>>,
}

impl TaskRunner for UnstructuredRunner {
    type Output = ChannelTextStream;

    fn run<M: Model>(&self, input: String, model: & M) -> Self::Output  where <<M as kalosm_language_model::Model>::SyncModel as kalosm_language_model::SyncModel>::Session: Send + Sync{
        let chat_markers = model.chat_markers();

        let stop_on = chat_markers
            .as_ref()
            .map(|m| m.end_assistant_marker.to_string())
            .unwrap_or_else(|| "# Input".to_string());

        let (tx, rx) = unbounded_channel();

        let sampler = self.sampler.clone();
        let stop_on = stop_on.clone();
        let sessions = self.sessions.clone();

        model.run_sync(move |model| {
            Box::pin(async move {
                let mut sessions_write = sessions.sessions.write().unwrap();
                let session_entry: &mut TaskSessionEntry<<M::SyncModel as SyncModel>::Session> = {
                    sessions_write
                        .entry(TypeId::of::<M>())
                        .or_insert_with(|| {
                            Box::new(
                                TaskSessionEntry::<<M::SyncModel as SyncModel>::Session>::new(
                                    chat_markers.clone(),
                                    sessions.system_prompt.clone(),
                                    &sessions.examples,
                                ),
                            )
                        })
                        .downcast_mut()
                        .unwrap()
                };
                let mut session = match session_entry.create_session(model) {
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
                let prompt = session_entry.task_prompt(&input);
                if let Err(err) = model.stream_text_with_sampler(
                    &mut session,
                    &prompt,
                    None,
                    Some(&stop_on),
                    sampler,
                    on_token,
                ) {
                    tracing::error!("Failed to stream text: {}", err);
                }
            })
        }).unwrap();

        rx.into()
    }
}

impl<P: Parser + CreateParserState + Sync + Send + 'static> TaskBuilderReturn for P
where
    <P as Parser>::Output: Clone + Send + 'static,
    <P as Parser>::PartialState: Sync + Send,
{
    type Output = StructuredRunner<P>;

    fn build(task_builder: TaskBuilder<Self>) -> Self::Output {
        let TaskBuilder {
            system_prompt,
            sampler,
            constraints,
            examples,
        } = task_builder;

        let arc_parser = Arc::new(constraints);

        // check if the examples are valid
        #[cfg(debug_assertions)]
        {
            for example in &examples {
                let state = arc_parser.create_parser_state();
                let result = arc_parser.parse(&state, example.output.as_bytes());
                if result.is_err() {
                    tracing::error!("Example: {:?} does not fit the constraints you provided to the task. Examples tend to perform better when they follow the same format as the model output.", example);
                }
            }
        }

        let sessions = TaskSessions::new(system_prompt, examples);

        StructuredRunner {
            sessions: Arc::new(sessions),
            sampler,
            parser: arc_parser,
        }
    }
}

/// A task runner for a task that follows constraints.
pub struct StructuredRunner<P> {
    sessions: Arc<TaskSessions>,
    sampler: Arc<std::sync::Mutex<dyn Sampler + Send + Sync>>,
    parser: Arc<P>,
}

impl<P> TaskRunner for StructuredRunner<P>
where
    P: Parser + CreateParserState + Sync + Send + 'static,
    <P as Parser>::Output: Clone + Send + 'static,
    <P as Parser>::PartialState: Sync + Send,
{
    type Output = StructureParserResult<ChannelTextStream, P::Output>;

    fn run<M: Model>(&self, input: String, model: &M) -> Self::Output where <<M as kalosm_language_model::Model>::SyncModel as kalosm_language_model::SyncModel>::Session: Send + Sync{
        let (tx, rx) = unbounded_channel();
        let (parsed_tx, parsed_rx) = oneshot::channel();
        let arc_parser = self.parser.clone();
        let sampler = self.sampler.clone();
        let sessions = self.sessions.clone();
        let chat_markers = model.chat_markers();

        model.run_sync(move |model| {
            Box::pin(async move {
                let mut sessions_write = sessions.sessions.write().unwrap();
                let session_entry: &mut TaskSessionEntry<<M::SyncModel as SyncModel>::Session> = {
                    sessions_write
                        .entry(TypeId::of::<M>())
                        .or_insert_with(|| {
                            Box::new(
                                TaskSessionEntry::<<M::SyncModel as SyncModel>::Session>::new(
                                    chat_markers,
                                    sessions.system_prompt.clone(),
                                    &sessions.examples,
                                ),
                            )
                        })
                        .downcast_mut()
                        .unwrap()
                };
                let span = tracing::span!(tracing::Level::TRACE, "Task session");
                let _span = span.enter();

                let mut session = match session_entry.create_session( model) {
                    Ok(session) => session,
                    Err(err) => {
                        tracing::error!("Failed to start session: {}", err);
                        return;
                    }
                };

                let state = arc_parser.create_parser_state();
                let on_token = |tok: String| {
                    tracing::trace!("Task generated token: {}", tok);
                    tx.send(tok)?;
                    Ok(())
                };
                let prompt = session_entry.task_prompt(&input);
                let result = model.generate_structured(
                    &mut session,
                    &prompt,
                    arc_parser,
                    state,
                    sampler,
                    on_token,
                    Some(32),
                );
                if parsed_tx.send(result).is_err() {
                    tracing::error!("Failed to send parsed result");
                }
            })
        }).unwrap();

        StructureParserResult::new(rx.into(), parsed_rx)
    }
}

// This is essentially a manual implementation of a closure so you can name the type
/// Something that can run a task.
pub trait TaskRunner {
    /// The output of the task.
    type Output: Stream<Item = String> + Send + Sync + Unpin + 'static;

    /// Run the task with a input and a model.
    fn run<M: Model>(&self, input: String, model: & M) -> Self::Output where <<M as kalosm_language_model::Model>::SyncModel as kalosm_language_model::SyncModel>::Session: Send + Sync;
}

/// A task session lets you efficiently run a task with a model. The task session will reuse the model's cache to avoid re-feeding the task prompt repeatedly.
///
/// # Example
/// ```rust, no_run
/// use kalosm_language::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let mut llm = Llama::new_chat().await.unwrap();
///     let mut task = Task::new("You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.");
///
///     println!("question 1");
///     // The first time we use the task, it will load the model and prompt.
///     task.run("What is 2 + 2?", &llm)
///         .to_std_out()
///         .await
///         .unwrap();
///     
///     println!("question 2");
///     // After the first time, the model and prompt are cached.
///     task.run("What is 4 + 4?", &llm)
///         .to_std_out()
///         .await
///         .unwrap();
/// }
/// ```
pub struct Task<R = UnstructuredRunner> {
    runner: R,
}

impl Task {
    /// Create a new task with no constraints and the default sampler. See [`Task::builder`] for more options.
    pub fn new(description: impl Into<String>) -> Self {
        Self::builder(description).build()
    }

    /// Creates a new builder for a task session.
    pub fn builder(description: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(description)
    }
}

impl<R: TaskRunner> Task<R> {
    /// Run the task with a message.
    pub fn run<M>(&self, message: impl Into<String>, model: &M) -> R::Output
    where
        M: Model,
         <<M as kalosm_language_model::Model>::SyncModel as kalosm_language_model::SyncModel>::Session: Send + Sync
    {
        let message = message.into();
        let message = message.trim().to_string();
        self.runner.run(message, model)
    }
}
