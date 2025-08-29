//! Calculates the embeddings for a given input.
//!
//! Bert embeddings contain word embeddings, embeddings about the token type and position information.

use fusor_core::{Device, VarBuilder};
use fusor_core::{Result, Tensor};

use crate::raw::embedding::{embedding, Embedding};
use crate::raw::layer_norm::{layer_norm, LayerNorm};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
pub(crate) struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let word_embeddings = embedding(device, &mut vb.pp("word_embeddings"))?;
        let position_embeddings = embedding(device, &mut vb.pp("position_embeddings"))?;
        let token_type_embeddings = embedding(device, &mut vb.pp("token_type_embeddings"))?;
        let layer_norm = layer_norm(device, &mut vb.pp("LayerNorm"), config.layer_norm_eps as _)?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub(crate) fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        token_type_ids: &Tensor<2, u32>,
    ) -> Tensor<2, f32> {
        let _enter = self.span.enter();
        let [_bsize, seq_len] = *input_ids.shape();
        let input_embeddings = self.word_embeddings.forward(input_ids);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);
        let mut embeddings = &input_embeddings + &token_type_embeddings;
        if let Some(position_embeddings) = &self.position_embeddings {
            let position_ids = Tensor::arange(input_ids.device(), 0, seq_len as u32);
            embeddings = embeddings.add_(&position_embeddings.forward(&position_ids))
        }
        self.layer_norm.forward(&embeddings)
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.word_embeddings.hidden_size()
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.position_embeddings
            .as_ref()
            .map(|p| p.embeddings().shape()[0])
            .unwrap_or(0)
    }
}
