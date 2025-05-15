use candle_core::{Tensor, D};
use candle_transformers::quantized_var_builder::VarBuilder;
use kalosm_common::KvCache;

use crate::raw::rope::RopeCache;

use super::{
    qwen_patch_merger::Qwen2VLPatchMerger, qwen_rope::VisionRotaryEmbedding,
    qwen_vision::get_window_index, qwen_vision_block::VisionBlock,
    qwen_vision_embed::Qwen2_5VisionPatchEmbed,
};

struct QwenVisionTransformer {
    spacial_merge_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
    window_size: usize,
    spacial_merge_unit: usize,
    patch_embed: Qwen2_5VisionPatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: Qwen2VLPatchMerger,
    device: candle_core::Device,
}

impl QwenVisionTransformer {
    fn new(
        spacial_merge_size: usize,
        temporal_patch_size: usize,
        patch_size: usize,
        fullatt_block_indexes: Vec<usize>,
        window_size: usize,
        in_channels: usize,
        hidden_size: usize,
        num_heads: usize,
        depth: usize,
        vb: &VarBuilder,
    ) -> candle_core::Result<Self> {
        let spacial_merge_unit = spacial_merge_size * spacial_merge_size;
        let patch_embed = Qwen2_5VisionPatchEmbed::new(
            patch_size,
            temporal_patch_size,
            in_channels,
            hidden_size,
            vb,
        )
        .unwrap();
        let head_dim = hidden_size / num_heads;
        let rope_theta = 10000.0;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, rope_theta, vb.device())?;
        let blocks = (0..depth)
            .map(|i| {
                VisionBlock::new(
                    &vb.pp(&format!("visual.blocks.{i}")),
                    num_heads,
                    head_dim,
                    hidden_size,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let merger = Qwen2VLPatchMerger::new(
            hidden_size,
            hidden_size,
            spacial_merge_size,
            &vb.pp("visual.merger"),
        )?;

        Ok(Self {
            spacial_merge_size,
            patch_size,
            fullatt_block_indexes,
            window_size,
            spacial_merge_unit,
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            device: vb.device().clone(),
        })
    }

    fn rot_pos_emb(&self, grid_thw: &Vec<(u32, u32, u32)>) -> candle_core::Result<Tensor> {
        let device = &self.device;
        let mut pos_ids = vec![];
        for (t, h, w) in grid_thw {
            let hpos_ids = Tensor::arange(0, *h, device)?
                .unsqueeze(1)?
                .expand(&[*h as usize, *w as usize])?
                .reshape(&[
                    *h as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                    *w as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                ])?
                .permute([0, 2, 1, 3])?
                .flatten_all()?;

            let wpos_ids = Tensor::arange(0, *w, device)?
                .unsqueeze(0)?
                .expand(&[*h as usize, *w as usize])?
                .reshape(&[
                    *h as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                    *w as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                ])?
                .permute([0, 2, 1, 3])?
                .flatten_all()?;
            let pos_id =
                Tensor::stack(&[hpos_ids, wpos_ids], D::Minus1)?.repeat(&[*t as usize, 1])?;
            pos_ids.push(pos_id);
        }
        let pos_ids = Tensor::cat(&pos_ids, 0)?;
        let max_grid_size = grid_thw.iter().map(|(_, h, w)| (*h).max(*w)).max().unwrap();
        let rotary_pos_emb_full = self.rotary_pos_emb.make_embeds(max_grid_size)?;
        let rotary_pos_emb = rotary_pos_emb_full.index_select(&pos_ids, 0)?.squeeze(1)?;
        Ok(rotary_pos_emb)
    }

    fn forward(
        &self,
        hidden_states: Tensor,
        grid_thw: &Vec<(u32, u32, u32)>,
        mut cache: Option<&mut KvCache>,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.patch_embed.forward(&hidden_states)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let (window_index, mut cu_window_seqlens) = get_window_index(
            grid_thw
                .iter()
                .map(|(t, h, w)| (*t as usize, *h as usize, *w as usize)),
            self.window_size,
            self.spacial_merge_size,
            self.spacial_merge_unit,
            self.patch_size,
            &self.device,
        )?;
        let mut last_item = None;
        cu_window_seqlens.retain(|&x| {
            if let Some(last) = last_item {
                if last == x {
                    return false;
                }
            }
            last_item = Some(x);
            true
        });

        let seq_len = hidden_states.dim(0)?;
        let hidden_states = hidden_states.reshape((
            seq_len / self.spacial_merge_unit,
            self.spacial_merge_unit,
            (),
        ))?;
        let hidden_states = hidden_states.index_select(&window_index, 0)?;
        let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((
            seq_len / self.spacial_merge_unit,
            self.spacial_merge_unit,
            (),
        ))?;
        let rotary_pos_emb = rotary_pos_emb.index_select(&window_index, 0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[rotary_pos_emb.clone(), rotary_pos_emb], D::Minus1)?;
        let rope_cache = RopeCache::from_parts(emb.cos()?, emb.sin()?)?;

        let cu_seqlens = grid_thw
            .iter()
            .flat_map(|(t, h, w)| std::iter::repeat_n((*h * *w) as u32, *t as usize))
            .map({
                let mut sum = 0;
                move |x| {
                    sum += x;
                    sum
                }
            });

        let cu_seqlens = std::iter::once(0).chain(cu_seqlens).collect::<Vec<_>>();

        for (layer_num, blk) in self.blocks.iter().enumerate() {
            let cu_seqlens_now = if self.fullatt_block_indexes.contains(&layer_num) {
                &cu_seqlens
            } else {
                &cu_window_seqlens
            };
            hidden_states = blk.forward(
                &hidden_states,
                cu_seqlens_now.as_slice(),
                &rope_cache,
                0,
                cache.as_deref_mut(),
            )?;
        }

        let hidden_states = self.merger.forward(&hidden_states)?;
        let reverse_indices = window_index.arg_sort_last_dim(true)?;
        let hidden_states = hidden_states.index_select(&reverse_indices, 0)?;

        Ok(hidden_states)
    }
}
