use kalosm_llama::StructuredInferenceTimingInfo;
use serde::{Deserialize, Serialize};
use std::{iter::Sum, time::Duration};

#[derive(Clone, Serialize, Deserialize)]
pub struct Run {
    pub generation: u32,
    pub metadata: StructuredInferenceTimingInfo,
    pub pass: bool,
    pub entropy: f64,
    pub entropy_diff: f64,
    pub tokenization_error: bool,
    pub tokens_after_tokenization_error: u32,
    pub result: String,
    pub tokens: Vec<u32>,
}

impl Sum<Run> for Summary {
    fn sum<I: Iterator<Item = Run>>(iter: I) -> Self {
        let mut total_metadata = StructuredInferenceTimingInfo::default();
        let mut total_entropy = 0.0;
        let mut total_entropy_diff = 0.0;
        let mut total_tokenization_error = 0.0;
        let mut total_tokens_after_tokenization_error = 0.0;
        let mut total_token_count = 0;
        let mut count = 0;
        let mut total_duration = Duration::default();

        for run in iter {
            total_duration = total_duration.checked_add(run.metadata.total_time).unwrap();
            total_metadata = total_metadata + run.metadata;
            total_entropy += run.entropy;
            total_entropy_diff += run.entropy_diff;
            total_tokenization_error += if run.tokenization_error { 1.0 } else { 0.0 };
            total_tokens_after_tokenization_error += run.tokens_after_tokenization_error as f64;
            total_token_count += run.tokens.len();
            count += 1;
        }

        let average_metadata = if count > 0 {
            total_metadata / (count as f64)
        } else {
            StructuredInferenceTimingInfo::default()
        };
        let average_entropy = if count > 0 {
            total_entropy / (count as f64)
        } else {
            0.0
        };
        let average_entropy_diff = if count > 0 {
            total_entropy_diff / (count as f64)
        } else {
            0.0
        };
        let average_tokenization_error = if count > 0 {
            total_tokenization_error / (count as f64)
        } else {
            0.0
        };
        let average_tokens_after_first_token_error = if count > 0 {
            total_tokens_after_tokenization_error / (count as f64)
        } else {
            0.0
        };
        let average_token_count = if count > 0 {
            total_token_count as f64 / (count as f64)
        } else {
            0.0
        };

        Summary {
            average_metadata,
            average_entropy,
            average_entropy_diff,
            average_tokenization_error,
            average_tokens_after_first_token_error,
            average_token_count,
            total_duration,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Summary {
    pub average_metadata: StructuredInferenceTimingInfo,
    pub average_entropy: f64,
    pub average_entropy_diff: f64,
    pub average_tokenization_error: f64,
    pub average_tokens_after_first_token_error: f64,
    pub total_duration: Duration,
    pub average_token_count: f64,
}
