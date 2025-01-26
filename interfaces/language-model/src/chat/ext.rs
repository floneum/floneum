use std::error::Error;

use super::BoxedChatModel;
use super::BoxedStructuredChatModel;
use super::Chat;
use super::ChatModel;
use super::ChatSession;
use super::CreateChatSession;
use super::CreateDefaultChatConstraintsForType;
use super::StructuredChatModel;
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

    fn boxed_chat_model(self) -> BoxedChatModel
    where
        Self: ChatModel<
                Error: Send + Sync + std::error::Error + 'static,
                ChatSession: ChatSession<Error: std::error::Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + Sized
            + Send
            + Sync
            + 'static,
    {
        BoxedChatModel::new(self)
    }

    fn boxed_typed_chat_model<T>(self) -> BoxedStructuredChatModel<T>
    where
        Self: StructuredChatModel<
                Self::DefaultConstraints,
                Error: Send + Sync + Error + 'static,
                ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + CreateDefaultChatConstraintsForType<T>
            + Sized
            + Send
            + Sync
            + 'static,
        T: 'static,
    {
        BoxedStructuredChatModel::new(self)
    }
}

impl<M: CreateChatSession> ChatModelExt for M {}
