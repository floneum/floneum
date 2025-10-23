use kalosm_model_types::FileSource;

const SNOWFLAKE_EMBEDDING_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";

/// A the source of a [`crate::Bert`] model
pub struct BertSource {
    pub(crate) search_embedding_prefix: Option<String>,
    pub(crate) config: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) model: FileSource,
}

impl BertSource {
    /// Create a new [`BertSource`] for embedding plain text
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new [`BertSource`] for embedding text for search
    pub fn new_for_search() -> Self {
        Self::snowflake_arctic_embed_extra_small()
    }

    /// Set the model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    pub fn with_model(mut self, model: FileSource) -> Self {
        self.model = model;
        self
    }

    /// Set the tokenizer to use
    pub fn with_tokenizer(mut self, tokenizer: FileSource) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set the config to use
    pub fn with_config(mut self, config: FileSource) -> Self {
        self.config = config;
        self
    }

    /// Set the prefix to use when embedding search queries
    pub(crate) fn with_search_embedding_prefix(
        mut self,
        prefix: impl Into<Option<String>>,
    ) -> Self {
        self.search_embedding_prefix = prefix.into();
        self
    }

    /// Create a new [`BertSource`] with the BGE large english preset
    pub fn bge_large_en() -> Self {
        // https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/blob/main/bge-large-en-v1.5-q4_k_m.gguf
        Self::default()
            .with_model(FileSource::huggingface(
                "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
                "main".to_string(),
                "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the BGE base english preset
    pub fn bge_base_en() -> Self {
        // https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/blob/main/bge-base-en-v1.5-q4_k_m.gguf
        Self::default()
            .with_model(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the BGE small english preset
    pub fn bge_small_en() -> Self {
        // https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/blob/main/bge-small-en-v1.5-q4_k_m.gguf
        Self {
            config: FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            model: FileSource::huggingface(
                "CompendiumLabs/bge-small-en-v1.5-gguf".to_string(),
                "main".to_string(),
                "bge-small-en-v1.5-q4_k_m.gguf".to_string(),
            ),
            search_embedding_prefix: None,
        }
    }

    /// Create a new [`BertSource`] with the MiniLM-L6-v2 preset
    pub fn mini_lm_l6_v2() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) model
    pub fn snowflake_arctic_embed_extra_small() -> Self {
        // https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-xs-gguf/blob/main/snowflake-arctic-embed-xs--Q4_K_M.GGUF
        Self::default()
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-xs".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-xs".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_model(FileSource::huggingface(
                "ChristianAzinn/snowflake-arctic-embed-xs-gguf".to_string(),
                "main".to_string(),
                "snowflake-arctic-embed-xs--Q4_K_M.GGUF".to_string(),
            ))
            .with_search_embedding_prefix(SNOWFLAKE_EMBEDDING_PREFIX.to_string())
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) model
    pub fn snowflake_arctic_embed_small() -> Self {
        // https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-s-gguf/blob/main/snowflake-arctic-embed-s--Q4_K_M.GGUF
        Self::default()
            .with_model(FileSource::huggingface(
                "ChristianAzinn/snowflake-arctic-embed-s-gguf".to_string(),
                "main".to_string(),
                "snowflake-arctic-embed-s--Q4_K_M.GGUF".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-s".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-s".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
            .with_search_embedding_prefix(SNOWFLAKE_EMBEDDING_PREFIX.to_string())
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) model
    pub fn snowflake_arctic_embed_medium() -> Self {
        // https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-gguf/blob/main/snowflake-arctic-embed-m--Q4_K_M.GGUF
        Self::default()
            .with_model(FileSource::huggingface(
                "ChristianAzinn/snowflake-arctic-embed-m-gguf".to_string(),
                "main".to_string(),
                "snowflake-arctic-embed-m--Q4_K_M.GGUF".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_search_embedding_prefix(SNOWFLAKE_EMBEDDING_PREFIX.to_string())
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) model
    ///
    /// This model is slightly larger than [`Self::snowflake_arctic_embed_medium`] and supports longer contexts (up to 2048 tokens).
    pub fn snowflake_arctic_embed_medium_long() -> Self {
        // https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-long-GGUF/blob/main/snowflake-arctic-embed-m-long--Q4_K_M.GGUF
        Self::default()
            .with_model(FileSource::huggingface(
                "ChristianAzinn/snowflake-arctic-embed-m-long-GGUF".to_string(),
                "main".to_string(),
                "snowflake-arctic-embed-m-long--Q4_K_M.GGUF".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m-long".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m-long".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
            .with_search_embedding_prefix(SNOWFLAKE_EMBEDDING_PREFIX.to_string())
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) model
    pub fn snowflake_arctic_embed_large() -> Self {
        // https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-l-gguf/blob/main/snowflake-arctic-embed-l--Q4_K_M.GGUF
        Self::default()
            .with_model(FileSource::huggingface(
                "ChristianAzinn/snowflake-arctic-embed-l-gguf".to_string(),
                "main".to_string(),
                "snowflake-arctic-embed-l--Q4_K_M.GGUF".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-l".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-l".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
            .with_search_embedding_prefix(SNOWFLAKE_EMBEDDING_PREFIX.to_string())
    }
}

impl Default for BertSource {
    fn default() -> Self {
        Self::bge_small_en()
    }
}
