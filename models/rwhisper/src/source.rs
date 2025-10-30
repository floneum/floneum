use std::{fmt::Display, str::FromStr};

use kalosm_model_types::FileSource;

/// The source whisper model to use.
#[derive(Clone, Copy, Debug, Default)]
pub enum WhisperSource {
    /// The tiny model quantized to run faster.
    QuantizedTiny,
    /// The tiny model with only English support quantized to run faster.
    QuantizedTinyEn,
    /// The medium model with only English support quantized to run faster.
    QuantizedDistilMediumEn,
    /// The quantized distil-large-v3 model.
    QuantizedDistilLargeV3,
    #[default]
    /// The quantized large-v3-turbo model.
    QuantizedLargeV3Turbo,
}

impl WhisperSource {
    /// Check if the model is multilingual.
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::QuantizedTiny | Self::QuantizedDistilLargeV3 | Self::QuantizedLargeV3Turbo => {
                true
            }
            Self::QuantizedTinyEn | Self::QuantizedDistilMediumEn => false,
        }
    }

    pub(crate) fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::QuantizedTiny => ("lmz/candle-whisper", "main"),
            Self::QuantizedTinyEn => ("lmz/candle-whisper", "main"),
            Self::QuantizedDistilMediumEn => {
                ("Demonthos/candle-quantized-whisper-medium-distil", "main")
            }
            Self::QuantizedDistilLargeV3 => {
                ("Demonthos/candle-quantized-whisper-distil-v3", "main")
            }
            Self::QuantizedLargeV3Turbo => {
                ("Demonthos/candle-quantized-whisper-large-v3-turbo", "main")
            }
        }
    }

    pub(crate) fn timestamp_attention_heads(&self) -> Option<&'static [[usize; 2]]> {
        match self {
            Self::QuantizedDistilMediumEn => None,
            Self::QuantizedTiny => Some(&[[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]),
            Self::QuantizedTinyEn => Some(&[
                [1, 0],
                [2, 0],
                [2, 5],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]),
            Self::QuantizedLargeV3Turbo => {
                Some(&[[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]])
            }
            Self::QuantizedDistilLargeV3 => Some(&[
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
        }
    }
}

/// Error that reports the unsupported value
#[derive(Debug, PartialEq, Eq)]
pub struct ParseWhisperSourceError(String);

impl Display for ParseWhisperSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source {} not supported ", self.0)
    }
}

impl FromStr for WhisperSource {
    type Err = ParseWhisperSourceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "quantized_tiny" => Ok(Self::QuantizedTiny),
            "quantized_tiny_en" => Ok(Self::QuantizedTinyEn),
            "quantized_distil_large_v3" => Ok(Self::QuantizedDistilLargeV3),
            "quantized_distil_medium_en" => Ok(Self::QuantizedDistilMediumEn),
            "quantized_large_v3_turbo" => Ok(Self::QuantizedLargeV3Turbo),
            _ => Err(ParseWhisperSourceError(s.to_owned())),
        }
    }
}

impl Display for WhisperSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QuantizedTiny => write!(f, "quantized_tiny"),
            Self::QuantizedTinyEn => write!(f, "quantized_tiny_en"),
            Self::QuantizedDistilMediumEn => write!(f, "quantized_distil_medium_en"),
            Self::QuantizedDistilLargeV3 => write!(f, "quantized_distil_large_v3"),
            Self::QuantizedLargeV3Turbo => write!(f, "quantized_large_v3_turbo"),
        }
    }
}

pub(crate) struct WhisperModelConfig {
    pub(crate) model: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) config: FileSource,
}
impl WhisperModelConfig {
    pub(crate) fn new(model: FileSource, tokenizer: FileSource, config: FileSource) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }
}
