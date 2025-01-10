//! A task interface that builds on top of [`crate::Chat`]

use std::mem::MaybeUninit;
use std::ops::Deref;

use super::Chat;
use super::ChatMessage;
use super::ChatResponseBuilder;
use super::CreateChatSession;
use super::MessageType;

/// A task session lets you efficiently run a task with a model. The task session will reuse the model's cache to avoid re-feeding the task prompt repeatedly.
///
/// # Example
/// ```rust, no_run
/// use kalosm_language::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let mut llm = Llama::new_chat().await.unwrap();
///     let mut task = llm.task("You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.");
///
///     println!("question 1");
///     // The first time we use the task, it will load the model and prompt.
///     task.run("What is 2 + 2?")
///         .to_std_out()
///         .await
///         .unwrap();
///     
///     println!("question 2");
///     // After the first time, the model and prompt are cached.
///     task.run("What is 4 + 4?")
///         .to_std_out()
///         .await
///         .unwrap();
/// }
/// ```
pub struct Task<M: CreateChatSession> {
    chat: Chat<M>,
}

impl<M: CreateChatSession> Task<M> {
    /// Create a new task with no constraints and the default sampler. See [`Task::builder`] for more options.
    pub fn new(model: M, description: impl ToString) -> Self {
        let chat = Chat::new(model).with_system_prompt(description);
        Self { chat }
    }

    /// Add an example to the task.
    pub fn with_example(mut self, input: impl ToString, output: impl ToString) -> Self {
        self.chat
            .add_message(ChatMessage::new(MessageType::UserMessage, input));
        self.chat
            .add_message(ChatMessage::new(MessageType::ModelAnswer, output));
        self
    }

    /// Add multiple examples to the task.
    pub fn with_examples(
        mut self,
        examples: impl IntoIterator<Item = (impl ToString, impl ToString)>,
    ) -> Self {
        for (input, output) in examples {
            self = self.with_example(input, output);
        }
        self
    }

    /// Run the task with a message.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm_language::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new_chat().await.unwrap();
    ///     let task = llm.task("You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.");
    ///
    ///     let result = task.run("What is 2 + 2?").all_text().await;
    ///     println!("{result}");
    /// }
    /// ```
    pub fn run(&self, message: impl ToString) -> ChatResponseBuilder<'static, M>
    where
        M: Clone,
    {
        self.chat.clone().into_add_message(message)
    }
}

impl<M: CreateChatSession + Clone + 'static> Deref for Task<M> {
    type Target = dyn Fn(&str) -> ChatResponseBuilder<'static, M>;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure =
            move |input: &str| Self::run(unsafe { &*uninit_callable.as_ptr() }, input);

        // Make sure the layout of the closure and Self is the same.
        let size_of_closure = std::alloc::Layout::for_value(&uninit_closure);
        assert_eq!(size_of_closure, std::alloc::Layout::new::<Self>());

        // Then cast the lifetime of the closure to the lifetime of &self.
        fn cast_lifetime<'a, T>(_a: &T, b: &'a T) -> &'a T {
            b
        }
        let reference_to_closure = cast_lifetime(
            {
                // The real closure that we will never use.
                &uninit_closure
            },
            #[allow(clippy::missing_transmute_annotations)]
            // We transmute self into a reference to the closure. This is safe because we know that the closure has the same memory layout as Self so &Closure == &Self.
            unsafe {
                std::mem::transmute(self)
            },
        );

        // Cast the closure to a trait object.
        reference_to_closure as &_
    }
}
