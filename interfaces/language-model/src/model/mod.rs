use kalosm_sample::Parser;
use std::{convert::Infallible, future::Future};

mod generation_parameters;
pub use generation_parameters::*;
// mod ext;

/// A session for a model.
pub trait Session {
    /// The type of error this model may return during operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Serialize the session into bytes.
    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error>;

    /// Write the session to bytes.
    fn to_bytes(&self) -> Result<Vec<u8>, Self::Error> {
        let mut bytes = Vec::new();
        self.write_to(&mut bytes)?;
        Ok(bytes)
    }

    /// Load the session from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;

    /// Try to clone the session.
    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

impl Session for () {
    type Error = Infallible;

    fn write_to(&self, _into: &mut Vec<u8>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn from_bytes(_bytes: &[u8]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn try_clone(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// No parser is supported for this chat model
pub struct NoConstraintsSupported;

impl ModelConstraints for NoConstraintsSupported {
    type Output = ();
}

/// A type that can constrain the output of a model into a specific output type.
pub trait ModelConstraints {
    /// The type of the output of the constraints.
    type Output;
}

impl<P> ModelConstraints for P
where
    P: Parser,
{
    type Output = <P as Parser>::Output;
}

pub trait ModelSession: Send + Sync + 'static {
    /// The type of error this model may return during operations.
    type Error: Send + Sync + 'static;

    /// The type of the session that this model uses.
    type Session: Session;

    /// Create a new session for this model.
    fn new_session(&self) -> Result<Self::Session, Self::Error>;
}

pub trait TextCompletionModel<Sampler>: ModelSession {
    /// Generate text with the given prompt.
    ///
    /// See [`ModelExt::stream_text`] for nicer API with an example.
    fn stream_text_with_callback(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: Sampler,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

#[doc = include_str!("../../docs/model.md")]
pub trait StructuredTextCompletionModel<
    Sampler,
    Constraints: ModelConstraints = NoConstraintsSupported,
>: TextCompletionModel<Sampler>
{
    /// Generate text with the given prompt.
    ///
    /// See [`ModelExt::stream_text`] for nicer API with an example.
    fn stream_text_with_callback_and_parser(
        &self,
        session: &mut Self::Session,
        text: &str,
        sampler: Sampler,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + Send;
}
