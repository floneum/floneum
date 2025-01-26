use crate::ModelConstraints;

use super::{
    ChatMessage, ChatModel, ChatSession, CreateChatSession, CreateDefaultChatConstraintsForType,
    StructuredChatModel,
};
use std::{error::Error, future::Future, pin::Pin, sync::Arc};

/// A boxed [`ChatModel`].
#[derive(Clone)]
pub struct BoxedChatModel {
    model: Arc<dyn DynChatModel + Send + Sync>,
}

impl BoxedChatModel {
    pub(crate) fn new(
        model: impl ChatModel<
                Error: Send + Sync + Error + 'static,
                ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            model: Arc::new(model),
        }
    }
}

impl CreateChatSession for BoxedChatModel {
    type ChatSession = BoxedChatSession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        self.model.new_chat_session_boxed()
    }
}

impl ChatModel for BoxedChatModel {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: crate::GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        self.model
            .add_messages_with_callback_boxed(session, messages, sampler, Box::new(on_token))
    }
}

/// A boxed [`StructuredChatModel`].
#[derive(Clone)]
pub struct BoxedStructuredChatModel<T> {
    model: Arc<dyn DynStructuredChatModel<T> + Send + Sync>,
}

impl<T> BoxedStructuredChatModel<T> {
    pub(crate) fn new<S>(model: S) -> Self
    where
        S: StructuredChatModel<
                S::DefaultConstraints,
                Error: Send + Sync + Error + 'static,
                ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + CreateDefaultChatConstraintsForType<T>
            + Send
            + Sync
            + 'static,
        T: 'static,
    {
        Self {
            model: Arc::new(model),
        }
    }
}

impl<T> CreateChatSession for BoxedStructuredChatModel<T> {
    type ChatSession = BoxedChatSession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        self.model.new_chat_session_boxed()
    }
}

impl<T> ChatModel for BoxedStructuredChatModel<T> {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: crate::GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        self.model
            .add_messages_with_callback_boxed(session, messages, sampler, Box::new(on_token))
    }
}

impl<T: 'static> StructuredChatModel<BoxedChatConstraintsForType<T>>
    for BoxedStructuredChatModel<T>
{
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: crate::GenerationParameters,
        constraints: BoxedChatConstraintsForType<T>,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<T, Self::Error>> + Send + 'a {
        self.model.add_message_with_callback_and_constraints_boxed(
            session,
            messages,
            sampler,
            constraints,
            Box::new(on_token),
        )
    }
}

impl<T> CreateDefaultChatConstraintsForType<T> for BoxedStructuredChatModel<T>
where
    T: 'static,
{
    type DefaultConstraints = BoxedChatConstraintsForType<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        BoxedChatConstraintsForType {
            phantom: std::marker::PhantomData,
        }
    }
}

/// A boxed [`ChatSession`].
pub struct BoxedChatSession {
    session: Box<dyn DynChatSession + Send + Sync>,
}

impl Clone for BoxedChatSession {
    fn clone(&self) -> Self {
        DynChatSession::clone_(&*self.session)
    }
}

impl ChatSession for BoxedChatSession {
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error> {
        self.session.write_to_boxed(into)
    }

    fn from_bytes(_: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        #[derive(Debug)]
        struct FromBytesNotSupported;

        impl std::error::Error for FromBytesNotSupported {}

        impl std::fmt::Display for FromBytesNotSupported {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "FromBytesNotSupported")
            }
        }

        Err(Box::new(FromBytesNotSupported))
    }

    fn history(&self) -> Vec<super::ChatMessage> {
        self.session.history_boxed()
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        self.session.try_clone_boxed()
    }

    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error> {
        self.session.to_bytes_boxed()
    }
}

#[derive(Debug)]
struct MismatchedSessionType;

impl std::error::Error for MismatchedSessionType {}

impl std::fmt::Display for MismatchedSessionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MismatchedSessionType")
    }
}

trait DynCreateChatSession {
    fn new_chat_session_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync>>;
}

impl<S> DynCreateChatSession for S
where
    S: CreateChatSession<
        Error: Send + Sync + Error,
        ChatSession: ChatSession<Error: Error> + Clone + Send + Sync + 'static,
    >,
{
    fn new_chat_session_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync>> {
        let session = self
            .new_chat_session()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        let session = Box::new(session) as Box<dyn DynChatSession + Send + Sync>;
        Ok(BoxedChatSession { session })
    }
}

trait DynChatSession {
    fn write_to_boxed(
        &self,
        into: &mut Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>;

    fn history_boxed(&self) -> Vec<super::ChatMessage>;

    fn try_clone_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync + 'static>>;

    fn to_bytes_boxed(&self)
        -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync + 'static>>;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn clone_(&self) -> BoxedChatSession;
}

impl<S: ChatSession<Error: Error> + Send + Sync + Clone + 'static> DynChatSession for S {
    fn write_to_boxed(
        &self,
        into: &mut Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.write_to(into)
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
    }

    fn history_boxed(&self) -> Vec<super::ChatMessage> {
        self.history()
    }

    fn try_clone_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let session = self
            .try_clone()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        let session = Box::new(session) as Box<dyn DynChatSession + Send + Sync>;
        Ok(BoxedChatSession { session })
    }

    fn to_bytes_boxed(
        &self,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.to_bytes()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_(&self) -> BoxedChatSession {
        BoxedChatSession {
            session: Box::new(Clone::clone(self)),
        }
    }
}

trait DynChatModel: DynCreateChatSession {
    fn add_messages_with_callback_boxed<'a>(
        &'a self,
        session: &'a mut BoxedChatSession,
        messages: &[super::ChatMessage],
        sampler: crate::GenerationParameters,
        on_token: Box<
            dyn FnMut(String) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>
                + Send
                + Sync
                + 'static,
        >,
    ) -> ::core::pin::Pin<
        Box<
            dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>>
                + Send
                + 'a,
        >,
    >;
}

impl<S> DynChatModel for S
where
    S: ChatModel<
        Error: Send + Sync + Error + 'static,
        ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                         + Clone
                         + Send
                         + Sync
                         + 'static,
    >,
{
    fn add_messages_with_callback_boxed<'a>(
        &'a self,
        session: &'a mut BoxedChatSession,
        messages: &[super::ChatMessage],
        sampler: crate::GenerationParameters,
        mut on_token: Box<
            dyn FnMut(String) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>
                + Send
                + Sync
                + 'static,
        >,
    ) -> ::core::pin::Pin<
        Box<
            dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>>
                + Send
                + 'a,
        >,
    > {
        let session = session.session.as_any_mut();

        let Some(session) = session.downcast_mut::<S::ChatSession>() else {
            return Box::pin(async move {
                Err(Box::new(MismatchedSessionType) as Box<dyn Error + Send + Sync>)
            });
        };
        let on_token = move |token: String| {
            if let Err(err) = on_token(token) {
                tracing::error!("Error running on_token callback: {}", err);
            }
            Ok(())
        };
        let future = self.add_messages_with_callback(session, messages, sampler, on_token);
        // Double box prevents a rust compiler error with lifetimes. See https://github.com/rust-lang/rust/issues/102211
        let future: Pin<Box<dyn Future<Output = Result<(), _>> + Send>> = Box::pin(future);
        Box::pin(async move {
            future
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync + 'static>)
        })
    }
}

/// A constraints for [`CreateDefaultChatConstraintsForType`] that work with boxed [`StructuredChatModel`]s.
pub struct BoxedChatConstraintsForType<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T> ModelConstraints for BoxedChatConstraintsForType<T> {
    type Output = T;
}

trait DynStructuredChatModel<T>: DynChatModel {
    fn add_message_with_callback_and_constraints_boxed<'a>(
        &'a self,
        session: &'a mut BoxedChatSession,
        messages: &[ChatMessage],
        sampler: crate::GenerationParameters,
        constraints: BoxedChatConstraintsForType<T>,
        on_token: Box<
            dyn FnMut(String) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>
                + Send
                + Sync
                + 'static,
        >,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>>
                + Send
                + 'a,
        >,
    >;
}

impl<S, T> DynStructuredChatModel<T> for S
where
    S: StructuredChatModel<
            S::DefaultConstraints,
            Error: Send + Sync + Error + 'static,
            ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                             + Clone
                             + Send
                             + Sync
                             + 'static,
        > + CreateDefaultChatConstraintsForType<T>,
    T: 'static,
{
    fn add_message_with_callback_and_constraints_boxed<'a>(
        &'a self,
        session: &'a mut BoxedChatSession,
        messages: &[ChatMessage],
        sampler: crate::GenerationParameters,
        _: BoxedChatConstraintsForType<T>,
        mut on_token: Box<
            dyn FnMut(String) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>
                + Send
                + Sync
                + 'static,
        >,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>>
                + Send
                + 'a,
        >,
    > {
        let constraints =
            <S as CreateDefaultChatConstraintsForType<T>>::create_default_constraints();
        let session = session.session.as_any_mut();

        let Some(session) = session.downcast_mut::<S::ChatSession>() else {
            return Box::pin(async move {
                Err(Box::new(MismatchedSessionType) as Box<dyn Error + Send + Sync>)
            });
        };

        let on_token = move |token: String| {
            if let Err(err) = on_token(token) {
                tracing::error!("Error running on_token callback: {}", err);
            }
            Ok(())
        };

        let future = self.add_message_with_callback_and_constraints(
            session,
            messages,
            sampler,
            constraints,
            on_token,
        );
        // Double box prevents a rust compiler error with lifetimes. See https://github.com/rust-lang/rust/issues/102211
        let future: Pin<Box<dyn Future<Output = Result<T, _>> + Send>> = Box::pin(future);
        Box::pin(async move {
            future
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync + 'static>)
        })
    }
}
