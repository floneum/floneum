use super::Chat;
use super::CreateChatSession;
use super::Task;

/// An extension trait for chat models with helpers for handling chat sessions. This trait is implemented automatically for all [`crate::ChatModel`]s.
pub trait ChatModelExt: CreateChatSession {
    /// Create a new chat session with the model.
    #[doc = include_str!("../../docs/chat.md")]
    fn chat(&self) -> Chat<Self>
    where
        Self: Clone,
    {
        Chat::new(self.clone())
    }

    /// Create a new task with the model.
    #[doc = include_str!("../../docs/task.md")]
    fn task(&self, description: impl ToString) -> Task<Self>
    where
        Self: Clone,
    {
        Task::new(self.clone(), description)
    }
}

impl<M: CreateChatSession> ChatModelExt for M {}
