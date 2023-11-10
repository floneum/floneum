use tokenizers::Tokenizer;

/// A source for the Llama model.
pub struct LlamaSource {
    /// The model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) gguf_file: String,
    pub(crate) tokenizer_repo: String,
    pub(crate) tokenizer_file: String,
    pub(crate) group_query_attention: u8,
}

impl LlamaSource {
    /// Create a new source for the Llama model.
    pub fn new(model_id: String, gguf_file: String, tokenizer_file: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
            gguf_file,
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file,
            group_query_attention: 1,
        }
    }

    /// Set the revision of the model to use.
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }

    /// Set the tokenizer repository to use.
    pub fn with_tokenizer_repo(mut self, tokenizer_file: String) -> Self {
        self.tokenizer_file = tokenizer_file;
        self
    }

    /// Set the tokenizer file to use.
    pub fn with_tokenizer_file(mut self, tokenizer_file: String) -> Self {
        self.tokenizer_file = tokenizer_file;
        self
    }

    /// Set the model (gguf) file to use.
    pub fn with_model_file(mut self, gguf_file: String) -> Self {
        self.gguf_file = gguf_file;
        self
    }

    pub(crate) fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_path = {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = self.tokenizer_repo.to_string();
            let api = api.model(repo);
            api.get(&self.tokenizer_file)?
        };
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    pub(crate) fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model(self.model_id.to_string());
        let model_path = api.get(&self.gguf_file)?;
        Ok(model_path)
    }

    /// A preset for Mistral7b
    pub fn mistral_7b() -> Self {
        Self {
            model_id: "TheBloke/Mistral-7B-v0.1-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "mistral-7b-v0.1.Q4_K_S.gguf".into(),
            tokenizer_repo: "mistralai/Mistral-7B-v0.1".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
        }
    }

    /// A preset for Mistral7bInstruct
    pub fn mistral_7b_instruct() -> Self {
        Self {
            model_id: "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "mistral-7b-instruct-v0.1.Q4_K_S.gguf".into(),
            tokenizer_repo: "mistralai/Mistral-7B-v0.1".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
        }
    }

    /// A preset for Zephyr7bAlpha
    pub fn zephyr_7b_alpha() -> Self {
        Self {
            model_id: "TheBloke/zephyr-7B-alpha-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "zephyr-7b-alpha.Q4_K_M.gguf".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
        }
    }

    /// A preset for Zephyr7bBeta
    pub fn zephyr_7b_beta() -> Self {
        Self {
            model_id: "TheBloke/zephyr-7B-beta-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "zephyr-7b-beta.Q4_K_M.gguf".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
        }
    }

    /// A preset for Llama7b
    pub fn llama_7b() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-7B-GGML".to_string(),
            revision: "main".to_string(),
            gguf_file: "llama-2-7b.ggmlv3.q4_0.bin".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }

    /// A preset for Llama13b
    pub fn llama_13b() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-13B-GGML".to_string(),
            revision: "main".to_string(),
            gguf_file: "llama-2-13b.ggmlv3.q4_0.bin".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }

    /// A preset for Llama70b
    pub fn llama_70b() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-70B-GGML".to_string(),
            revision: "main".to_string(),
            gguf_file: "llama-2-70b.ggmlv3.q4_0.bin".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
        }
    }

    /// A preset for Llama7bChat
    pub fn llama_7b_chat() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-7B-Chat-GGML".to_string(),
            revision: "main".to_string(),
            gguf_file: "llama-2-7b-chat.ggmlv3.q4_0.bin".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }

    /// A preset for Llama13bChat
    pub fn llama_13b_chat() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-13B-Chat-GGML".to_string(),
            revision: "main".to_string(),
            gguf_file: "llama-2-13b-chat.ggmlv3.q4_0.bin".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }

    /// A preset for Llama70bChat
    pub fn llama_70b_chat() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-70B-Chat-GGML".to_string(),
            revision: "main".to_string(),
            gguf_file: "llama-2-70b-chat.ggmlv3.q4_0.bin".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
        }
    }

    /// A preset for Llama7bCode
    pub fn llama_7b_code() -> Self {
        Self {
            model_id: "TheBloke/CodeLlama-7B-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "codellama-7b.Q8_0.gguf".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }

    /// A preset for Llama13bCode
    pub fn llama_13b_code() -> Self {
        Self {
            model_id: "TheBloke/CodeLlama-13B-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "codellama-13b.Q8_0.gguf".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }

    /// A preset for Llama34bCode
    pub fn llama_34b_code() -> Self {
        Self {
            model_id: "TheBloke/CodeLlama-34B-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "codellama-34b.Q8_0.gguf".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 1,
        }
    }
}

impl Default for LlamaSource {
    fn default() -> Self {
        Self::llama_13b()
    }
}
