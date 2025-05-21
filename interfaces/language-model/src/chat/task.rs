use std::mem::MaybeUninit;
use std::ops::Deref;

use crate::ModelConstraints;
use crate::NoConstraints;

use super::Chat;
use super::ChatMessage;
use super::ChatResponseBuilder;
use super::CreateChatSession;
use super::CreateDefaultChatConstraintsForType;
use super::IntoChatMessage;
use super::MessageContent;
use super::MessageType;
use super::ToChatMessage;

/// A task session lets you efficiently run a task with a model. The task session will reuse the model's cache to avoid re-feeding the task prompt repeatedly.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
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
#[derive(Debug)]
pub struct Task<M: CreateChatSession, Constraints = NoConstraints> {
    chat: Chat<M>,
    constraints: Constraints,
}

impl<M: CreateChatSession, Constraints: Clone> Clone for Task<M, Constraints> {
    fn clone(&self) -> Self {
        Self {
            chat: self.chat.clone(),
            constraints: self.constraints.clone(),
        }
    }
}

impl<M: CreateChatSession> Task<M> {
    /// Create a new task with no constraints and the default sampler.
    pub fn new(model: M, description: impl ToString) -> Self {
        let chat = Chat::new(model).with_system_prompt(description);
        Self {
            chat,
            constraints: NoConstraints,
        }
    }

    /// Create a new task from an existing chat session.
    pub fn from_chat(chat: Chat<M>) -> Self {
        Self {
            chat,
            constraints: NoConstraints,
        }
    }
}

impl<M: CreateChatSession, Constraints> Task<M, Constraints> {
    /// Add an example to the task. Examples help the model perform better by allowing it to mimic the format of the examples.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let model = Llama::new_chat().await.unwrap();
    ///     let task = model.task("You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.")
    ///         .with_example("What is 1 + 2?", "Step 1: 1 + 2 = 3\nOutput: 3");
    ///     let mut stream = task(&"What is 2 + 2?");
    ///     stream.to_std_out().await.unwrap();
    /// }
    /// ```
    pub fn with_example(mut self, input: impl Into<MessageContent>, output: impl ToString) -> Self {
        self.chat
            .add_message(ChatMessage::new(MessageType::UserMessage, input));
        self.chat.add_message(ChatMessage::new(
            MessageType::ModelAnswer,
            output.to_string(),
        ));
        self
    }

    /// Add multiple examples to the task. Examples help the model perform better by allowing it to mimic the format of the examples.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let model = Llama::new_chat().await.unwrap();
    ///     let task = model.task("You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.")
    ///         .with_examples([
    ///             ("What is 1 + 2?", "Step 1: 1 + 2 = 3\nOutput: 3"),
    ///             ("What is 3 + 4?", "Step 1: 3 + 4 = 7\nOutput: 7"),
    ///             ("What is (4 + 8) / 3?", "Step 1: 4 + 8 = 12\nStep 2: 12 / 3 = 4\nOutput: 4"),
    ///         ]);
    ///     let mut stream = task(&"What is 3 + 4?");
    ///     stream.to_std_out().await.unwrap();
    /// }
    /// ```
    pub fn with_examples(
        mut self,
        examples: impl IntoIterator<Item = (impl Into<MessageContent>, impl ToString)>,
    ) -> Self {
        for (input, output) in examples {
            self = self.with_example(input, output);
        }
        self
    }

    /// Set the constraints for the task. The constraints force the format of all outputs of the task to fit
    /// the constraints. This can be used to make the model return a specific type. This method does the same thing
    /// as [`ChatResponseBuilder::with_constraints`] except it is called once on the task instead of any time you
    /// run the task.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let model = Llama::new_chat().await.unwrap();
    ///     let task = model
    ///         .task("You are a math assistant. Respond with just the number answer and nothing else.")
    ///         .with_constraints(Arc::new(i32::new_parser()));
    ///     let mut stream = task(&"What is 2 + 2?");
    ///     stream.to_std_out().await.unwrap();
    ///     let result: i32 = stream.await.unwrap();
    ///     println!("{result}");
    /// }
    /// ```
    pub fn with_constraints<NewConstraints>(
        self,
        constraints: NewConstraints,
    ) -> Task<M, NewConstraints> {
        Task {
            chat: self.chat,
            constraints,
        }
    }

    /// Create a task with the default constraints for the given type. This is the same as calling [`Task::with_constraints`] with the default constraints for the given type.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let model = Llama::new_chat().await.unwrap();
    ///     let task = model
    ///         .task("You are a math assistant. Respond with just the number answer and nothing else.")
    ///         .typed();
    ///     let mut stream = task(&"What is 2 + 2?");
    ///     stream.to_std_out().await.unwrap();
    ///     let result: i32 = stream.await.unwrap();
    ///     println!("{result}");
    /// }
    /// ```
    pub fn typed<T>(
        self,
    ) -> Task<M, <M as CreateDefaultChatConstraintsForType<T>>::DefaultConstraints>
    where
        M: CreateDefaultChatConstraintsForType<T>,
    {
        self.with_constraints(M::create_default_constraints())
    }

    /// Get a reference to the underlying chat session.
    pub fn chat(&self) -> &Chat<M> {
        &self.chat
    }
}

impl<M: CreateChatSession, Constraints: Clone> Task<M, Constraints> {
    /// Run the task with a message.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new_chat().await.unwrap();
    ///     let task = llm.task("You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.");
    ///
    ///     let result = task(&"What is 2 + 2?").await.unwrap();
    ///     println!("{result}");
    /// }
    /// ```
    pub fn run<Msg: IntoChatMessage>(
        &self,
        message: Msg,
    ) -> ChatResponseBuilder<'static, M, Constraints> {
        self.chat
            .clone()
            .into_add_message(message)
            .with_constraints(self.constraints.clone())
    }
}

impl<M: CreateChatSession + 'static, Constraints: ModelConstraints + Clone + 'static> Deref
    for Task<M, Constraints>
{
    type Target = dyn Fn(&dyn ToChatMessage) -> ChatResponseBuilder<'static, M, Constraints>;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure = move |input: &dyn ToChatMessage| {
            Self::run(
                unsafe { &*uninit_callable.as_ptr() },
                input.to_chat_message(),
            )
        };

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

impl<M: CreateChatSession + 'static> Deref for Task<M> {
    type Target = dyn Fn(&dyn ToChatMessage) -> ChatResponseBuilder<'static, M>;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure = move |input: &dyn ToChatMessage| {
            Self::run(
                unsafe { &*uninit_callable.as_ptr() },
                input.to_chat_message(),
            )
        };

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
