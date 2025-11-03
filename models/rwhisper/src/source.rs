use kalosm_model_types::FileSource;

/// Predefined Whisper model sources
#[derive(Debug, Clone)]
pub struct WhisperSource {
    pub(crate) model: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) config: FileSource,
    pub(crate) multilingual: bool,
    pub(crate) heads: Option<&'static [[usize; 2]]>,
}

impl Default for WhisperSource {
    fn default() -> Self {
        Self::tiny_en()
    }
}

impl WhisperSource {
    /// Create a new WhisperSource
    pub fn new(
        model: FileSource,
        tokenizer: FileSource,
        config: FileSource,
        multilingual: bool,
        heads: Option<&'static [[usize; 2]]>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            multilingual,
            heads,
        }
    }

    /// Tiny english model
    pub fn tiny_en() -> Self {
        let model = FileSource::huggingface(
            "lmz/candle-whisper".to_owned(),
            "main".to_owned(),
            "model-tiny-en-q80.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "lmz/candle-whisper".to_owned(),
            "main".to_owned(),
            "tokenizer-tiny-en.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "lmz/candle-whisper".to_owned(),
            "main".to_owned(),
            "config-tiny-en.json".to_owned(),
        );
        WhisperSource::new(
            model,
            tokenizer,
            config,
            false,
            Some(&[
                [1, 0],
                [2, 0],
                [2, 5],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]),
        )
    }

    /// Tiny model
    pub fn tiny() -> Self {
        let model = FileSource::huggingface(
            "lmz/candle-whisper".to_owned(),
            "main".to_owned(),
            "model-tiny-q80.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "lmz/candle-whisper".to_owned(),
            "main".to_owned(),
            "tokenizer-tiny.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "lmz/candle-whisper".to_owned(),
            "main".to_owned(),
            "config-tiny.json".to_owned(),
        );
        WhisperSource::new(
            model,
            tokenizer,
            config,
            true,
            Some(&[[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]),
        )
    }

    /// Base model
    pub fn base() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/fusor-whisper-base".to_owned(),
            "main".to_owned(),
            "whisper-base.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "openai/whisper-base".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "openai/whisper-base".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(model, tokenizer, config, true, None)
    }

    /// Base english model
    pub fn base_en() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/fusor-whisper-base-en".to_owned(),
            "main".to_owned(),
            "whisper-base-en.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "openai/whisper-base.en".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "openai/whisper-base.en".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(model, tokenizer, config, false, None)
    }

    /// Medium model
    pub fn medium() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/fusor-whisper-medium".to_owned(),
            "main".to_owned(),
            "whisper-medium.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "openai/whisper-medium".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "openai/whisper-medium".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(model, tokenizer, config, true, None)
    }

    /// Medium english model
    pub fn medium_en() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/fusor-whisper-medium-en".to_owned(),
            "main".to_owned(),
            "whisper-medium-en.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "openai/whisper-medium.en".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "openai/whisper-medium.en".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(model, tokenizer, config, false, None)
    }

    /// Large v3 model
    pub fn large_v3() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/fusor-whisper-large-v3".to_owned(),
            "main".to_owned(),
            "whisper-large-v3.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "openai/whisper-large-v3".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "openai/whisper-large-v3".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(model, tokenizer, config, true, None)
    }

    /// Distiled medium english model
    pub fn distil_medium_en() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-medium-distil".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-medium-distil".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-medium-distil".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(model, tokenizer, config, false, None)
    }

    /// Distiled large v3.5 model
    pub fn distil_large_v3_5() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/fusor-distil-whisper-large-v3.5".to_owned(),
            "main".to_owned(),
            "whisper-distil-large-3.5.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "distil-whisper/distil-large-v3.5".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "distil-whisper/distil-large-v3.5".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(
            model,
            tokenizer,
            config,
            true,
            Some(&[
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
                [1, 10],
                [1, 11],
                [1, 12],
                [1, 13],
                [1, 14],
                [1, 15],
                [1, 16],
                [1, 17],
                [1, 18],
                [1, 19],
            ]),
        )
    }

    /// Distiled large v3 model
    pub fn distil_large_v3() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-distil-v3".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-distil-v3".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-distil-v3".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(
            model,
            tokenizer,
            config,
            true,
            Some(&[
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
                [1, 10],
                [1, 11],
                [1, 12],
                [1, 13],
                [1, 14],
                [1, 15],
                [1, 16],
                [1, 17],
                [1, 18],
                [1, 19],
            ]),
        )
    }

    /// Large v3 turbo model
    pub fn large_v3_turbo() -> Self {
        let model = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-large-v3-turbo".to_owned(),
            "main".to_owned(),
            "model.gguf".to_owned(),
        );
        let tokenizer = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-large-v3-turbo".to_owned(),
            "main".to_owned(),
            "tokenizer.json".to_owned(),
        );
        let config = FileSource::huggingface(
            "Demonthos/candle-quantized-whisper-large-v3-turbo".to_owned(),
            "main".to_owned(),
            "config.json".to_owned(),
        );
        WhisperSource::new(
            model,
            tokenizer,
            config,
            true,
            Some(&[[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]]),
        )
    }
}
