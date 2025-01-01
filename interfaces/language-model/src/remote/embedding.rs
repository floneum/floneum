use super::OpenAICompatibleClient;


/// An embedder that uses OpenAI's API for the a remote embedding model.
#[derive(Debug)]
pub struct OpenAICompatibleEmbeddingModel {
    model: String,
    client: OpenAICompatibleClient
}

/// A builder for an openai compatible embedding model.
#[derive(Debug, Default)]
pub struct OpenAICompatibleEmbeddingModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    client: OpenAICompatibleClient
}

impl OpenAICompatibleEmbeddingModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            config: Default::default(),
        }
    }
}

impl<const WITH_NAME: bool> OpenAICompatibleEmbeddingModelBuilder<WITH_NAME> {
    /// Set the name of the model to use.
    pub fn with_model(mut self, model: impl ToString) -> OpenAICompatibleEmbeddingModelBuilder<true> {
        OpenAICompatibleEmbeddingModelBuilder {
            model: Some(model.to_string()),
            client: self.client,
        }
    }

    /// Set the client used to make requests to the OpenAI API.
    pub fn with_client(mut self, client: OpenAICompatibleClient) -> Self {
        self.client = client;
        self
    }
}

impl OpenAICompatibleEmbeddingModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> OpenAICompatibleEmbeddingModel {
        OpenAICompatibleEmbeddingModel {
            model: self.config.model.unwrap(),
            client: self.client,
        }
    }
}

impl ModelBuilder for OpenAICompatibleEmbeddingModelBuilder<true> {
    type Model = AdaEmbedder;
    type Error = std::convert::Infallible;

    async fn start_with_loading_handler(
        self,
        _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<AdaEmbedder, Self::Error> {
        Ok(self.build())
    }

    fn requires_download(&self) -> bool {
        false
    }
}

/// The embedding space for the Ada embedding model.
pub struct AdaEmbedding;

impl VectorSpace for AdaEmbedding {}

impl AdaEmbedder {
    /// The model ID for the Ada embedding model.
    pub const MODEL_ID: &'static str = "text-embedding-ada-002";
}

impl Embedder for AdaEmbedder {
    type VectorSpace = AdaEmbedding;
    type Error = OpenAIError;

    fn embed_for(
        &self,
        input: crate::EmbeddingInput,
    ) -> impl Future<Output = Result<Embedding, Self::Error>> + Send {
        self.embed_string(input.text)
    }

    fn embed_vec_for(
        &self,
        inputs: Vec<crate::EmbeddingInput>,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::Error>> + Send {
        let inputs = inputs
            .into_iter()
            .map(|input| input.text)
            .collect::<Vec<_>>();
        self.embed_vec(inputs)
    }

    /// Embed a single string.
    fn embed_string(
        &self,
        input: String,
    ) -> impl Future<Output = Result<Embedding, Self::Error>> + Send {
        Box::pin(async move {
            let request = CreateEmbeddingRequestArgs::default()
                .model(Self::MODEL_ID)
                .input([input])
                .build()?;
            let response = self.client.embeddings().create(request).await?;

            let embedding = Embedding::from(response.data[0].embedding.iter().copied());

            Ok(embedding)
        })
    }

    /// Embed a single string.
    fn embed_vec(
        &self,
        input: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::Error>> + Send {
        Box::pin(async move {
            let request = CreateEmbeddingRequestArgs::default()
                .model(Self::MODEL_ID)
                .input(input)
                .build()?;
            let response = self.client.embeddings().create(request).await?;

            Ok(response
                .data
                .into_iter()
                .map(|data| Embedding::from(data.embedding.into_iter()))
                .collect())
        })
    }
}
