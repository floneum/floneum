// Forked from https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/bert.rs

mod embeddings;
use embeddings::*;
mod attention;
use attention::*;
mod encoder;
use encoder::*;
mod layer;
use layer::*;
mod output_layer;
use output_layer::*;
mod self_attention;
use self_attention::*;
mod self_output;
use self_output::*;
mod intermediate_layer;
use intermediate_layer::*;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

pub(crate) const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

/// The configuration of a [`BertModel`].
// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    hidden_dropout_prob: f32,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
}

/// A raw synchronous Bert model. You should generally use the [`super::Bert`] instead.
// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pub(crate) device: Device,
    span: tracing::Span,
}

impl BertModel {
    /// Load a new [`BertModel`] from [`VarBuilder`] with a [`Config`].
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        BertEmbeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        BertEncoder::load(vb.pp(&format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    /// Run the bert model with a batch of inputs.
    ///
    /// input_ids: The token ids of the input.
    /// token_type_ids: The token type ids of the input. (this should be a tensor of 0s for embedding tasks)
    /// attention_mask: The attention mask of the input. This can be None for embedding tasks with a single sentence. If you pad the input with 0s, you will need to create an attention mask.
    /// train: Whether to run the model in training mode or not.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids, train)?;
        let sequence_output = self
            .encoder
            .forward(&embedding_output, attention_mask, train)?;
        Ok(sequence_output)
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.embeddings.embedding_dim()
    }
}
