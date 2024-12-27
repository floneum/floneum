use kalosm_common::ModelLoadingProgress;

/// A builder that can create a model asynchronously.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
/// use kalosm_language_model::ModelBuilder;
///
/// #[tokio::main]
/// async fn main() {
///     let model = AdaEmbedderBuilder::default().start().await.unwrap();
/// }
/// ```
#[async_trait::async_trait]
pub trait ModelBuilder {
    /// The model that this trait creates.
    type Model;

    /// An error that can occur when creating the model.
    type Error: Send + Sync + 'static;

    /// Start the model.
    async fn start(self) -> Result<Self::Model, Self::Error>
    where
        Self: Sized,
    {
        self.start_with_loading_handler(|_| {}).await
    }

    /// Start the model with a loading handler.
    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error>
    where
        Self: Sized;

    /// Check if the model will need to be downloaded before use (default: false)
    fn requires_download(&self) -> bool {
        false
    }
}
