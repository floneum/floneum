use kalosm_common::FileSource;
use kalosm_language_model::ChatMarkers;
use tokenizers::Tokenizer;

fn llama_tokenizer() -> FileSource {
    FileSource::huggingface(
        "hf-internal-testing/llama-tokenizer".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

fn llama_v3_tokenizer() -> FileSource {
    FileSource::huggingface(
        "NousResearch/Meta-Llama-3-8B-Instruct".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

fn mistral_tokenizer() -> FileSource {
    FileSource::huggingface(
        "mistralai/Mistral-7B-v0.1".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

/// A source for the Llama model.
#[derive(Clone, Debug)]
pub struct LlamaSource {
    /// The model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    // pub(crate) model_id: String,
    // pub(crate) revision: String,
    // pub(crate) gguf_file: String,
    // pub(crate) tokenizer_repo: String,
    // pub(crate) tokenizer_file: String,
    pub(crate) model: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) group_query_attention: u8,
    pub(crate) markers: Option<ChatMarkers>,
}

impl LlamaSource {
    /// Create a new source for the Llama model.
    pub fn new(model: FileSource, tokenizer: FileSource) -> Self {
        Self {
            model,
            tokenizer,
            group_query_attention: 1,
            markers: Default::default(),
        }
    }

    /// Set the marker text for a user message
    pub fn with_chat_markers(mut self, markers: ChatMarkers) -> Self {
        self.markers = Some(markers);

        self
    }

    /// Set the group query attention for the model
    /// For the llama family of models, this is typically 1
    /// For the mistral family of models, this is typically 8
    pub fn with_group_query_attention(mut self, group_query_attention: u8) -> Self {
        self.group_query_attention = group_query_attention;

        self
    }

    pub(crate) async fn tokenizer(
        &self,
        progress: impl FnMut(f32) + Send + Sync,
    ) -> anyhow::Result<Tokenizer> {
        let tokenizer_path = self.tokenizer.download(progress).await?;
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    pub(crate) async fn model(
        &self,
        progress: impl FnMut(f32) + Send + Sync,
    ) -> anyhow::Result<std::path::PathBuf> {
        self.model.download(progress).await
    }

    /// A preset for Mistral7b
    pub fn mistral_7b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Mistral-7B-v0.1-GGUF".to_string(),
                "main".to_string(),
                "mistral-7b-v0.1.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            ..Default::default()
        }
    }

    /// A preset for Mistral7bInstruct
    pub fn mistral_7b_instruct() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
                "main".to_string(),
                "mistral-7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<s>[INST] ",
                end_system_prompt_marker: " [/INST]",
                user_marker: "[INST] ",
                end_user_marker: " [/INST]",
                assistant_marker: "",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for Mistral7bInstruct v0.2
    pub fn mistral_7b_instruct_2() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".to_string(),
                "main".to_string(),
                "mistral-7b-instruct-v0.2.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<s>[INST] ",
                end_system_prompt_marker: " [/INST]",
                user_marker: "[INST] ",
                end_user_marker: " [/INST]",
                assistant_marker: "",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for NeuralHermes-2.5-Mistral-7B-GGUF
    pub fn neural_hermes_2_5_mistral_7b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/NeuralHermes-2.5-Mistral-7B-GGUF".to_string(),
                "main".to_string(),
                "neuralhermes-2.5-mistral-7b.Q4_0.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<|im_start|>system\n",
                end_system_prompt_marker: "<|im_end|>",
                user_marker: "<|im_start|>user\n",
                end_user_marker: "<|im_end|>",
                assistant_marker: "<|im_start|>assistant\n",
                end_assistant_marker: "<|im_end|>",
            }),
        }
    }

    /// A preset for Neural Chat v3.3
    pub fn neural_chat_7b_v3_3() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/neural-chat-7B-v3-3-GGUF".to_string(),
                "main".to_string(),
                "neural-chat-7b-v3-3.Q4_0.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "Intel/neural-chat-7b-v3-3".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "### System:\n",
                end_system_prompt_marker: "\n",
                user_marker: "### User\n",
                end_user_marker: "\n",
                assistant_marker: "### Assistant:\n",
                end_assistant_marker: "\n",
            }),
        }
    }

    /// A preset for Zephyr7bAlpha
    pub fn zephyr_7b_alpha() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/zephyr-7B-alpha-GGUF".to_string(),
                "main".to_string(),
                "zephyr-7b-alpha.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<|system|>",
                user_marker: "<|user|>",
                assistant_marker: "<|assistant|>",
                end_system_prompt_marker: "</s>",
                end_user_marker: "</s>",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for Zephyr7bBeta
    pub fn zephyr_7b_beta() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/zephyr-7B-beta-GGUF".to_string(),
                "main".to_string(),
                "zephyr-7b-beta.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<|system|>",
                user_marker: "<|user|>",
                assistant_marker: "<|assistant|>",
                end_system_prompt_marker: "</s>",
                end_user_marker: "</s>",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for [Open chat 3.5 (0106)](https://huggingface.co/openchat/openchat-3.5-0106)
    pub fn open_chat_7b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/openchat-3.5-0106-GGUF".to_string(),
                "main".to_string(),
                "openchat-3.5-0106.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "openchat/openchat-3.5-0106".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "",
                end_system_prompt_marker: "<|end_of_turn|>",
                user_marker: "GPT4 Correct User: ",
                end_user_marker: "<|end_of_turn|>",
                assistant_marker: "GPT4 Correct Assistant: ",
                end_assistant_marker: "<|end_of_turn|>",
            }),
        }
    }

    /// A preset for Starling 7b Alpha
    pub fn starling_7b_alpha() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Starling-LM-7B-alpha-GGUF".to_string(),
                "main".to_string(),
                "starling-lm-7b-alpha.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "berkeley-nest/Starling-LM-7B-alpha".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "",
                end_system_prompt_marker: "<|end_of_turn|>",
                user_marker: "GPT4 Correct User: ",
                end_user_marker: "<|end_of_turn|>",
                assistant_marker: "GPT4 Correct Assistant: ",
                end_assistant_marker: "<|end_of_turn|>",
            }),
        }
    }

    /// A preset for Starling 7b Beta
    pub fn starling_7b_beta() -> Self {
        Self {
            model: FileSource::huggingface(
                "bartowski/Starling-LM-7B-beta-GGUF".to_string(),
                "main".to_string(),
                "Starling-LM-7B-beta-Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "Nexusflow/Starling-LM-7B-beta".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "",
                end_system_prompt_marker: "<|end_of_turn|>",
                user_marker: "GPT4 Correct User: ",
                end_user_marker: "<|end_of_turn|>",
                assistant_marker: "GPT4 Correct Assistant: ",
                end_assistant_marker: "<|end_of_turn|>",
            }),
        }
    }

    /// A preset for WizardLM 2 7B
    pub fn wizard_lm_7b_v2() -> Self {
        Self {
            model: FileSource::huggingface(
                "bartowski/WizardLM-2-7B-GGUF".to_string(),
                "main".to_string(),
                "WizardLM-2-7B-Q4_K_M.gguf".to_string(),
            ),
            tokenizer: mistral_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "",
                end_system_prompt_marker: "",
                user_marker: "USER: ",
                end_user_marker: "</s>",
                assistant_marker: "ASSISTANT: ",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for tiny llama 1.1b 1.0 Chat
    pub fn tiny_llama_1_1b_chat() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
                "main".to_string(),
                "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            group_query_attention: 4,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<|system|>\n",
                assistant_marker: "<|user|>\n",
                user_marker: "<|assistant|>\n",
                end_system_prompt_marker: "</s>",
                end_user_marker: "</s>",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for tiny llama 1.1b 1.0
    pub fn tiny_llama_1_1b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF".to_string(),
                "main".to_string(),
                "tinyllama-1.1b-intermediate-step-1431k-3t.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            group_query_attention: 4,
            ..Default::default()
        }
    }

    /// A preset for Llama7b v2
    pub fn llama_7b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Llama-2-7B-GGML".to_string(),
                "main".to_string(),
                "llama-2-7b.ggmlv3.q4_0.bin".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            ..Default::default()
        }
    }

    /// A preset for Llama8b v3
    pub fn llama_8b() -> Self {
        Self {
            model: FileSource::huggingface(
                "NousResearch/Meta-Llama-3-8B-GGUF".to_string(),
                "main".to_string(),
                "Meta-Llama-3-8B-Q4_K_M.gguf".to_string(),
            ),
            tokenizer: llama_v3_tokenizer(),
            group_query_attention: 1,
            ..Default::default()
        }
    }

    /// A preset for Llama8b v3
    pub fn llama_8b_chat() -> Self {
        Self {
            model: FileSource::huggingface(
                "bartowski/Meta-Llama-3-8B-Instruct-GGUF".to_string(),
                "main".to_string(),
                "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf".to_string(),
            ),
            tokenizer: llama_v3_tokenizer(),
            group_query_attention: 1,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
                end_system_prompt_marker: "<|eot_id|>",
                user_marker: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
                end_user_marker: "<|eot_id|>",
                assistant_marker: "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>",
                end_assistant_marker: "<|eot_id|>",
            }),
            ..Default::default()
        }
    }

    /// A preset for Llama13b
    pub fn llama_13b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Llama-2-13B-GGML".to_string(),
                "main".to_string(),
                "llama-2-13b.ggmlv3.q4_0.bin".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            markers: Default::default(),
        }
    }

    /// A preset for Llama70b
    pub fn llama_70b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Llama-2-70B-GGML".to_string(),
                "main".to_string(),
                "llama-2-70b.ggmlv3.q4_0.bin".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 8,
            ..Default::default()
        }
    }

    /// A preset for Llama7bChat
    pub fn llama_7b_chat() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Llama-2-7B-Chat-GGML".to_string(),
                "main".to_string(),
                "llama-2-7b-chat.ggmlv3.q4_0.bin".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<<SYS>>\n",
                assistant_marker: " [/INST] ",
                user_marker: "[INST]",
                end_system_prompt_marker: "</s>",
                end_user_marker: "</s>",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for Llama13bChat
    pub fn llama_13b_chat() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Llama-2-13B-Chat-GGML".to_string(),
                "main".to_string(),
                "llama-2-13b-chat.ggmlv3.q4_0.bin".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<<SYS>>\n",
                assistant_marker: " [/INST] ",
                user_marker: "[INST]",
                end_system_prompt_marker: "</s>",
                end_user_marker: "</s>",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for Llama70bChat
    pub fn llama_70b_chat() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/Llama-2-70B-Chat-GGML".to_string(),
                "main".to_string(),
                "llama-2-70b-chat.ggmlv3.q4_0.bin".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 8,
            markers: Some(ChatMarkers {
                system_prompt_marker: "<<SYS>>\n",
                assistant_marker: " [/INST] ",
                user_marker: "[INST]",
                end_system_prompt_marker: "</s>",
                end_user_marker: "</s>",
                end_assistant_marker: "</s>",
            }),
        }
    }

    /// A preset for Llama7bCode
    pub fn llama_7b_code() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/CodeLlama-7B-GGUF".to_string(),
                "main".to_string(),
                "codellama-7b.Q8_0.gguf".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            ..Default::default()
        }
    }

    /// A preset for Llama13bCode
    pub fn llama_13b_code() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/CodeLlama-13B-GGUF".to_string(),
                "main".to_string(),
                "codellama-13b.Q8_0.gguf".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            ..Default::default()
        }
    }

    /// A preset for Llama34bCode
    pub fn llama_34b_code() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/CodeLlama-34B-GGUF".to_string(),
                "main".to_string(),
                "codellama-34b.Q8_0.gguf".to_string(),
            ),
            tokenizer: llama_tokenizer(),
            group_query_attention: 1,
            ..Default::default()
        }
    }

    /// A preset for the SOLAR 10.7B model
    pub fn solar_10_7b() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/SOLAR-10.7B-v1.0-GGUF".to_string(),
                "main".to_string(),
                "solar-10.7b-v1.0.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "upstage/SOLAR-10.7B-v1.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            ..Default::default()
        }
    }

    /// A preset for the SOLAR 10.7B Instruct model
    pub fn solar_10_7b_instruct() -> Self {
        Self {
            model: FileSource::huggingface(
                "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF".to_string(),
                "main".to_string(),
                "solar-10.7b-instruct-v1.0.Q4_K_M.gguf".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "upstage/SOLAR-10.7B-Instruct-v1.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            markers: Some(ChatMarkers {
                system_prompt_marker: "<s>### System:\n",
                end_system_prompt_marker: "",
                user_marker: "### User:\n",
                end_user_marker: "",
                assistant_marker: "### Assistant:\n",
                end_assistant_marker: "</s>",
            }),
            ..Default::default()
        }
    }
}

impl Default for LlamaSource {
    fn default() -> Self {
        Self::llama_13b()
    }
}
