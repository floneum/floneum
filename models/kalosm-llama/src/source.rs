use tokenizers::Tokenizer;

#[derive(Default)]
pub(crate) struct ChatMarkers {
    pub(crate) user_marker: Option<&'static str>,
    pub(crate) end_user_marker: Option<&'static str>,
    pub(crate) assistant_marker: Option<&'static str>,
    pub(crate) end_assistant_marker: Option<&'static str>,
    pub(crate) system_prompt_marker: Option<&'static str>,
    pub(crate) end_system_marker: Option<&'static str>,
}

/// A source for the Llama model.
pub struct LlamaSource {
    /// The model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) gguf_file: String,
    pub(crate) tokenizer_repo: String,
    pub(crate) tokenizer_file: String,
    pub(crate) group_query_attention: u8,
    pub(crate) markers: ChatMarkers,
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
            markers: ChatMarkers::default(),
        }
    }

    /// Set the marker text for a user message
    pub fn with_user_marker(mut self, marker: &'static str) -> Self {
        self.markers.user_marker = Some(marker);

        self
    }

    /// Set the marker text for an assistant message
    pub fn with_assistant_marker(mut self, marker: &'static str) -> Self {
        self.markers.assistant_marker = Some(marker);

        self
    }

    /// Set the marker text for a the system prompt
    pub fn with_system_prompt_marker(mut self, marker: &'static str) -> Self {
        self.markers.user_marker = Some(marker);

        self
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
            ..Default::default()
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
            markers: ChatMarkers {
                system_prompt_marker: Some("<s>[INST] "),
                end_system_marker: Some(" [/INST]"),
                user_marker: Some("[INST] "),
                end_user_marker: Some(" [/INST]"),
                assistant_marker: Some(""),
                end_assistant_marker: Some("</s>"),
            },
        }
    }

    /// A preset for Mistral7bInstruct v0.2
    pub fn mistral_7b_instruct_2() -> Self {
        Self {
            model_id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf".into(),
            tokenizer_repo: "mistralai/Mistral-7B-v0.1".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
            markers: ChatMarkers {
                system_prompt_marker: Some("<s>[INST] "),
                end_system_marker: Some(" [/INST]"),
                user_marker: Some("[INST] "),
                end_user_marker: Some(" [/INST]"),
                assistant_marker: Some(""),
                end_assistant_marker: Some("</s>"),
            },
        }
    }

    /// A preset for Zephyr7bAlpha
    pub fn zephyr_7b_alpha() -> Self {
        Self {
            model_id: "TheBloke/zephyr-7B-alpha-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "zephyr-7b-alpha.Q4_K_M.gguf".into(),
            tokenizer_repo: "mistralai/Mistral-7B-v0.1".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
            markers: ChatMarkers {
                system_prompt_marker: Some("<|system|>"),
                user_marker: Some("<|user|>"),
                assistant_marker: Some("<|assistant|>"),
                end_system_marker: Some("</s>"),
                end_user_marker: Some("</s>"),
                end_assistant_marker: Some("</s>"),
            },
        }
    }

    /// A preset for Zephyr7bBeta
    pub fn zephyr_7b_beta() -> Self {
        Self {
            model_id: "TheBloke/zephyr-7B-beta-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "zephyr-7b-beta.Q4_K_M.gguf".into(),
            tokenizer_repo: "mistralai/Mistral-7B-v0.1".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
            markers: ChatMarkers {
                system_prompt_marker: Some("<|system|>"),
                user_marker: Some("<|user|>"),
                assistant_marker: Some("<|assistant|>"),
                end_system_marker: Some("</s>"),
                end_user_marker: Some("</s>"),
                end_assistant_marker: Some("</s>"),
            },
        }
    }

    /// A preset for Open chat 3.5
    pub fn open_chat_7b() -> Self {
        Self {
            model_id: "TheBloke/openchat_3.5-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "openchat_3.5.Q4_K_M.gguf".into(),
            tokenizer_repo: "openchat/openchat_3.5".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
            markers: ChatMarkers {
                system_prompt_marker: Some(""),
                end_system_marker: Some("<|end_of_turn|>"),
                user_marker: Some("GPT4 Correct User: "),
                end_user_marker: Some("<|end_of_turn|>"),
                assistant_marker: Some("GPT4 Correct Assistant: "),
                end_assistant_marker: Some("<|end_of_turn|>"),
            },
        }
    }

    /// A preset for Starling 7b Alpha
    pub fn starling_7b_alpha() -> Self {
        Self {
            model_id: "TheBloke/Starling-LM-7B-alpha-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "starling-lm-7b-alpha.Q4_K_M.gguf".into(),
            tokenizer_repo: "berkeley-nest/Starling-LM-7B-alpha".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 8,
            markers: ChatMarkers {
                system_prompt_marker: Some(""),
                end_system_marker: Some("<|end_of_turn|>"),
                user_marker: Some("GPT4 Correct User: "),
                end_user_marker: Some("<|end_of_turn|>"),
                assistant_marker: Some("GPT4 Correct Assistant: "),
                end_assistant_marker: Some("<|end_of_turn|>"),
            },
        }
    }

    /// A preset for tiny llama 1.1b 1.0 Chat
    pub fn tiny_llama_1_1b_chat() -> Self {
        Self {
            model_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into(),
            tokenizer_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 4,
            markers: ChatMarkers {
                system_prompt_marker: Some("<|system|>\n"),
                assistant_marker: Some("<|user|>\n"),
                user_marker: Some("<|assistant|>\n"),
                end_system_marker: Some("</s>"),
                end_user_marker: Some("</s>"),
                end_assistant_marker: Some("</s>"),
            },
        }
    }

    /// A preset for tiny llama 1.1b 1.0
    pub fn tiny_llama_1_1b() -> Self {
        Self {
            model_id: "TheBloke/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "tinyllama-1.1b-intermediate-step-1431k-3t.Q4_K_M.gguf".into(),
            tokenizer_repo: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            group_query_attention: 4,
            ..Default::default()
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
            ..Default::default()
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
            markers: Default::default(),
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
            ..Default::default()
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
            markers: ChatMarkers {
                system_prompt_marker: Some("<<SYS>>\n"),
                assistant_marker: Some(" [/INST] "),
                user_marker: Some("[INST]"),
                end_system_marker: Some("</s>"),
                end_user_marker: Some("</s>"),
                end_assistant_marker: Some("</s>"),
            },
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
            markers: ChatMarkers {
                system_prompt_marker: Some("<<SYS>>\n"),
                assistant_marker: Some(" [/INST] "),
                user_marker: Some("[INST]"),
                end_system_marker: Some("</s>"),
                end_user_marker: Some("</s>"),
                end_assistant_marker: Some("</s>"),
            },
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
            markers: ChatMarkers {
                system_prompt_marker: Some("<<SYS>>\n"),
                assistant_marker: Some(" [/INST] "),
                user_marker: Some("[INST]"),
                end_system_marker: Some("</s>"),
                end_user_marker: Some("</s>"),
                end_assistant_marker: Some("</s>"),
            },
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        }
    }

    /// A preset for the SOLAR 10.7B model
    pub fn solar_10_7b() -> Self {
        Self {
            model_id: "TheBloke/SOLAR-10.7B-v1.0-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "solar-10.7b-v1.0.Q4_K_M.gguf".into(),
            tokenizer_repo: "upstage/SOLAR-10.7B-v1.0".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            ..Default::default()
        }
    }

    /// A preset for the SOLAR 10.7B Instruct model
    pub fn solar_10_7b_instruct() -> Self {
        Self {
            model_id: "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF".to_string(),
            revision: "main".to_string(),
            gguf_file: "solar-10.7b-instruct-v1.0.Q4_K_M.gguf".into(),
            tokenizer_repo: "upstage/SOLAR-10.7B-Instruct-v1.0".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            markers: ChatMarkers {
                system_prompt_marker: Some("<s>### System:\n"),
                end_system_marker: Some(""),
                user_marker: Some("### User:\n"),
                end_user_marker: Some(""),
                assistant_marker: Some("### Assistant:\n"),
                end_assistant_marker: Some("</s>"),
            },
            ..Default::default()
        }
    }
}

impl Default for LlamaSource {
    fn default() -> Self {
        Self::llama_13b()
    }
}
