
/// An extension trait for models that can be converted into a trait object.
pub trait AnyModelExt: Model<TextStream = ChannelTextStream> + Send + Sync + 'static {
    /// Convert this model into a model trait object.
    fn into_any_model(self) -> DynModel
    where
        Self: Send + Sync + Sized,
        Self::Error: std::error::Error,
    {
        Box::new(AnyModel(self))
    }
}

impl<M: Model<TextStream = ChannelTextStream> + Send + Sync + 'static> AnyModelExt for M {}

/// A trait object for a model.
pub type DynModel = Box<
    dyn Model<
            TextStream = ChannelTextStream,
            SyncModel = BoxedSyncModel,
            Error = Box<dyn std::error::Error + Send + Sync>,
        > + Send,
>;

impl Model for DynModel {
    type TextStream = ChannelTextStream;
    type SyncModel = BoxedSyncModel;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn tokenizer(&self) -> Arc<Tokenizer> {
        let self_ref: &(dyn Model<
            TextStream = ChannelTextStream,
            SyncModel = BoxedSyncModel,
            Error = Box<dyn std::error::Error + Send + Sync>,
        > + Send) = self.as_ref();
        self_ref.tokenizer()
    }

    async fn stream_text_inner(
        &self,
        prompt: &str,
        parameters: GenerationParameters,
    ) -> Result<Self::TextStream, Self::Error> {
        let self_ref: &(dyn Model<
            TextStream = ChannelTextStream,
            SyncModel = BoxedSyncModel,
            Error = Box<dyn std::error::Error + Send + Sync>,
        > + Send) = self.as_ref();
        self_ref.stream_text_inner(prompt, parameters).await
    }

    async fn stream_text_with_sampler(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> Result<Self::TextStream, Self::Error> {
        let self_ref: &(dyn Model<
            TextStream = ChannelTextStream,
            SyncModel = BoxedSyncModel,
            Error = Box<dyn std::error::Error + Send + Sync>,
        > + Send) = self.as_ref();
        self_ref
            .stream_text_with_sampler(prompt, max_tokens, stop_on, sampler)
            .await
    }
}

trait AnySessionTrait {
    fn to_bytes(&self, into: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

impl<S: Any + Session> AnySessionTrait for S {
    fn to_bytes(&self, into: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Session::write_to(self, into).map_err(|e| e.into())
    }
}

/// An error that can occur when using a boxed session.
#[derive(Debug, Error)]
pub enum AnySessionError {
    /// An error from the underlying session.
    #[error("Underlying session error: {0}")]
    Session(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// An error that occurred when trying to load a boxed session. Boxed sessions do not support loading from a
    /// path because the type erased session does not have a known format.
    #[error("Loading boxed session from path is not supported")]
    Load,
}

/// A type-erased session.
///
/// > Note: boxed sessions do not support loading from a path.
#[derive(Clone)]
pub struct AnySession {
    session: Arc<dyn AnySessionTrait>,
}

impl Session for AnySession {
    type Error = AnySessionError;

    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error> {
        self.session
            .to_bytes(into)
            .map_err(AnySessionError::Session)
    }

    fn from_bytes(_bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Err(AnySessionError::Load)
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

struct AnyModel<M>(M);

impl<M, P> Model<GenerationParameters, P> for AnyModel<M>
where
    M: Model<TextStream = ChannelTextStream> + Send + Sync,
    M::Error: std::error::Error,
    P: Parse + Schema + 'static,
{
    type Session = AnySession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.0.tokenizer()
    }

    async fn stream_text_inner(
        &self,
        prompt: &str,
        params: GenerationParameters,
    ) -> Result<Self::TextStream, Self::Error> {
        self.0
            .stream_text_inner(prompt, params)
            .await
            .map_err(|e| e.into())
    }

    async fn stream_text_with_sampler(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> Result<Self::TextStream, Self::Error> {
        self.0
            .stream_text_with_sampler(prompt, max_tokens, stop_on, sampler)
            .await
            .map_err(|e| e.into())
    }
}