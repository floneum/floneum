//! Calculates the embeddings for a given input.
//!
//! Bert embeddings contain word embeddings, embeddings about the token type and position information.

use fusor_core::{Device, VarBuilder};
use fusor_core::{Result, Tensor};
use pollster::FutureExt;

use crate::raw::embedding::{embedding, Embedding};
use crate::raw::layer_norm::{layer_norm, LayerNorm};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
pub(crate) struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm<1>,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let word_embeddings = embedding(device, &mut vb.pp("token_embd"))?;
        let position_embeddings = embedding(device, &mut vb.pp("position_embd"))?;
        let token_type_embeddings = embedding(device, &mut vb.pp("token_types"))?;
        let layer_norm = layer_norm(
            device,
            &mut vb.pp("token_embd_norm"),
            config.layer_norm_eps as _,
        )?;
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
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let [_bsize, seq_len] = *input_ids.shape();
        let input_embeddings = self.word_embeddings.forward(input_ids);
        println!(
            "input_embeddings: {:?}",
            input_embeddings.as_slice().block_on()
        );
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);
        println!(
            "token_type_embeddings: {:?}",
            token_type_embeddings.as_slice().block_on()
        );
        let mut embeddings = &input_embeddings + &token_type_embeddings;
        if let Some(position_embeddings) = &self.position_embeddings {
            let position_ids = Tensor::arange(input_ids.device(), 0, seq_len as u32);
            let pos_emb = position_embeddings.forward(&position_ids);
            println!("pos_emb: {:?}", pos_emb.as_slice().block_on());
            embeddings = embeddings.add_(&pos_emb)
        }
        println!("embeddings: {:?}", embeddings.as_slice().block_on());
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
