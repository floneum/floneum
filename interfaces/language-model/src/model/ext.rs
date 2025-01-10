use crate::GenerationParameters;
use crate::ModelConstraints;
use crate::NoConstraints;
use futures_channel::mpsc::UnboundedReceiver;
use futures_channel::oneshot::Receiver;
use futures_util::Future;
use futures_util::FutureExt;
use futures_util::Stream;
use futures_util::StreamExt;
use std::any::Any;
use std::future::IntoFuture;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::task::Poll;

use super::CreateTextCompletionSession;
use super::StructuredTextCompletionModel;
use super::TextCompletionModel;

/// An extension trait for text generation models with a builder api for streaming
/// structured or unstructured text.
pub trait TextCompletionModelExt: CreateTextCompletionSession {
    #[doc = include_str!("../../docs/complete.md")]
    fn complete(&self, text: &str) -> TextCompletionBuilder<Self>
    where
        Self: Clone,
    {
        // Then create the builder that will respond to the message if it is awaited
        TextCompletionBuilder {
            text: text.to_string(),
            model: Some(self.clone()),
            constraints: None,
            sampler: Some(GenerationParameters::default()),
            task: OnceLock::new(),
            queued_tokens: None,
            result: None,
        }
    }
}

impl<M: CreateTextCompletionSession> TextCompletionModelExt for M {}

/// A builder for a chat response. This is returned by [`Chat::add_message`] and can be modified until you start awaiting the response.
///
/// # Example
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// let mut chat = Chat::builder(Llama::new_chat().await.unwrap())
///     .with_system_prompt("The assistant will act like a pirate.")
///     .build();
///
/// // Add a message to the chat session with the given message
/// let mut response = chat.add_message(prompt_input("\n> ").unwrap());
/// // Before you start streaming the response, you can add constraints to the response
/// let response = response.with_constraints(String::new_parser());
/// // Once you start streaming the response, the generation starts
/// response.to_std_out().await.unwrap();
/// // The response can be awaited to get the final (typed) result
/// let all_text: String = response.await.unwrap();
/// println!("{all_text}");
/// # }
/// ```
pub struct TextCompletionBuilder<
    M: CreateTextCompletionSession,
    Constraints = NoConstraints,
    Sampler = GenerationParameters,
> {
    text: String,
    model: Option<M>,
    constraints: Option<Constraints>,
    sampler: Option<Sampler>,
    task: OnceLock<RwLock<Pin<Box<dyn Future<Output = ()> + Send>>>>,
    result: Option<Receiver<Result<Box<dyn Any + Send>, M::Error>>>,
    queued_tokens: Option<UnboundedReceiver<String>>,
}

impl<M: CreateTextCompletionSession, Constraints, Sampler>
    TextCompletionBuilder<M, Constraints, Sampler>
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
    ) -> TextCompletionBuilder<M, NewConstraints, Sampler> {
        TextCompletionBuilder {
            text: self.text,
            model: self.model,
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
    ) -> TextCompletionBuilder<M, Constraints, NewSampler> {
        TextCompletionBuilder {
            text: self.text,
            model: self.model,
            constraints: self.constraints,
            sampler: Some(sampler),
            queued_tokens: None,
            result: None,
            task: OnceLock::new(),
        }
    }
}

impl<M, Sampler> TextCompletionBuilder<M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: TextCompletionModel<Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Send + Sync + Unpin + 'static,
{
    fn ensure_unstructured_task_started(&mut self) {
        if self.task.get().is_none() {
            let text = std::mem::take(&mut self.text);
            let model = self
                .model
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let sampler = self
                .sampler
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let (mut tx, rx) = futures_channel::mpsc::unbounded();
            let (result_tx, result_rx) = futures_channel::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let all_text = Arc::new(Mutex::new(String::new()));
            let on_token = {
                let all_text = all_text.clone();
                move |tok: String| {
                    all_text.lock().unwrap().push_str(&tok);
                    _ = tx.start_send(tok);
                    Ok(())
                }
            };
            let future = async move {
                let mut session = model.new_session()?;
                model
                    .stream_text_with_callback(&mut session, &text, sampler, on_token)
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

impl<M, Sampler> Stream for TextCompletionBuilder<M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: TextCompletionModel<Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Send + Sync + Unpin + 'static,
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
                if let Poll::Ready(Some(token)) = token.poll_next_unpin(cx) {
                    return Poll::Ready(Some(token));
                }
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        task.poll_unpin(cx).map(|_| None)
    }
}

impl<M, Sampler> IntoFuture for TextCompletionBuilder<M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: TextCompletionModel<Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
{
    type Output = Result<String, M::Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

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

impl<M, Constraints, Sampler> TextCompletionBuilder<M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredTextCompletionModel<Constraints, Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    fn ensure_structured_task_started(&mut self) {
        if self.task.get().is_none() {
            let text = std::mem::take(&mut self.text);
            let model = self
                .model
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let sampler = self
                .sampler
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let constraints = self
                .constraints
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let (mut tx, rx) = futures_channel::mpsc::unbounded();
            let (result_tx, result_rx) = futures_channel::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let on_token = move |tok: String| {
                _ = tx.start_send(tok);
                Ok(())
            };
            let future = async move {
                let mut session = model.new_session()?;
                model
                    .stream_text_with_callback_and_parser(
                        &mut session,
                        &text,
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

impl<M, Constraints, Sampler> Stream for TextCompletionBuilder<M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredTextCompletionModel<Constraints, Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
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
                if let Poll::Ready(Some(token)) = token.poll_next_unpin(cx) {
                    return Poll::Ready(Some(token));
                }
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        task.poll_unpin(cx).map(|_| None)
    }
}

impl<M, Constraints, Sampler> IntoFuture for TextCompletionBuilder<M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredTextCompletionModel<Constraints, Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    type Output = Result<Constraints::Output, M::Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

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
