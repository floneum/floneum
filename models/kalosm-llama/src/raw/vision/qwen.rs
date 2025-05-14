use candle_core::{Tensor, D};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::{
    qwen_patch_merger::Qwen2VLPatchMerger, qwen_rope::VisionRotaryEmbedding,
    qwen_vision_block::VisionBlock, qwen_vision_embed::Qwen2_5VisionPatchEmbed,
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

}
