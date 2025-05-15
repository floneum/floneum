use candle_core::{IndexOp, Tensor, D};
use candle_transformers::quantized_var_builder::VarBuilder;
use kalosm_common::{accelerated_device_if_available, KvCache};
use tracing::span;

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
        out_hidden_size: usize,
        embed_dim: usize,
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
            &vb.pp("visual.patch_embed"),
        )
        .unwrap();
        let head_dim = hidden_size / num_heads;
        let rope_theta = 10000.0;
        let rotary_pos_emb =
            VisionRotaryEmbedding::new(head_dim / 2, rope_theta, vb.device()).unwrap();
        let blocks = (0..depth)
            .map(|i| {
                VisionBlock::new(
                    &vb.pp(&format!("visual.blocks.{i}")),
                    num_heads,
                    head_dim,
                    hidden_size,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()
            .unwrap();
        let merger = Qwen2VLPatchMerger::new(
            out_hidden_size,
            hidden_size,
            spacial_merge_size,
            &vb.pp("visual.merger"),
        )
        .unwrap();

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
            let hpos_ids = Tensor::arange(0, *h, device)
                .unwrap()
                .unsqueeze(1)
                .unwrap()
                .expand(&[*h as usize, *w as usize])
                .unwrap()
                .reshape(&[
                    *h as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                    *w as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                ])
                .unwrap()
                .permute([0, 2, 1, 3])
                .unwrap()
                .flatten_all()
                .unwrap();

            let wpos_ids = Tensor::arange(0, *w, device)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .expand(&[*h as usize, *w as usize])
                .unwrap()
                .reshape(&[
                    *h as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                    *w as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                ])
                .unwrap()
                .permute([0, 2, 1, 3])
                .unwrap()
                .flatten_all()
                .unwrap();
            let pos_id = Tensor::stack(&[hpos_ids, wpos_ids], D::Minus1)
                .unwrap()
                .repeat(&[*t as usize, 1])
                .unwrap();
            pos_ids.push(pos_id);
        }
        let pos_ids = Tensor::cat(&pos_ids, 0).unwrap();
        let max_grid_size = grid_thw.iter().map(|(_, h, w)| (*h).max(*w)).max().unwrap();
        let rotary_pos_emb_full = self.rotary_pos_emb.make_embeds(max_grid_size).unwrap();

        let rotary_pos_emb_0 = rotary_pos_emb_full
            .index_select(&pos_ids.i((.., 0)).unwrap(), 0)
            .unwrap();

        let rotary_pos_emb_1 = rotary_pos_emb_full
            .index_select(&pos_ids.i((.., 1)).unwrap(), 0)
            .unwrap();

        let rotary_pos_emb = Tensor::cat(&[rotary_pos_emb_0, rotary_pos_emb_1], D::Minus1).unwrap();

        Ok(rotary_pos_emb)
    }

    fn forward(
        &self,
        hidden_states: Tensor,
        grid_thw: &Vec<(u32, u32, u32)>,
        mut cache: Option<&mut KvCache>,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.patch_embed.forward(&hidden_states).unwrap();
        let rotary_pos_emb = self.rot_pos_emb(grid_thw).unwrap();
        let (window_index, mut cu_window_seqlens) = get_window_index(
            grid_thw
                .iter()
                .map(|(t, h, w)| (*t as usize, *h as usize, *w as usize)),
            self.window_size,
            self.spacial_merge_size,
            self.spacial_merge_unit,
            self.patch_size,
            &self.device,
        )
        .unwrap();
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

        let seq_len = hidden_states.dim(0).unwrap();
        let hidden_states = hidden_states
            .reshape((
                seq_len / self.spacial_merge_unit,
                self.spacial_merge_unit,
                (),
            ))
            .unwrap();
        let hidden_states = hidden_states.index_select(&window_index, 0).unwrap();
        let mut hidden_states = hidden_states.reshape((seq_len, ())).unwrap();
        let rotary_pos_emb = rotary_pos_emb
            .reshape((
                seq_len / self.spacial_merge_unit,
                self.spacial_merge_unit,
                (),
            ))
            .unwrap();
        let rotary_pos_emb = rotary_pos_emb.index_select(&window_index, 0).unwrap();
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ())).unwrap();
        let emb = Tensor::cat(&[rotary_pos_emb.clone(), rotary_pos_emb], D::Minus1).unwrap();
        let rope_cache = RopeCache::from_parts(emb.cos().unwrap(), emb.sin().unwrap()).unwrap();

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
            hidden_states = blk
                .forward(
                    &hidden_states,
                    cu_seqlens_now.as_slice(),
                    &rope_cache,
                    0,
                    cache.as_deref_mut(),
                )
                .unwrap();
        }

        let hidden_states = self.merger.forward(&hidden_states).unwrap();
        let reverse_indices = window_index.arg_sort_last_dim(true).unwrap();
        let hidden_states = hidden_states.index_select(&reverse_indices, 0).unwrap();

        Ok(hidden_states)
    }
}

#[tokio::test]
async fn test_loading_qwen_vision() {
    use super::qwen_vision_embed::assert_2d_vec_eq;

    let device = accelerated_device_if_available().unwrap();
    let vb = VarBuilder::from_gguf(
        "/Users/evanalmloff/Desktop/Github/candle/qwen_2_5_3b_f16.gguf",
        &device,
    )
    .unwrap();
    // "depth": 32,
    //     "hidden_act": "silu",
    //     "hidden_size": 1280,
    //     "intermediate_size": 3420,
    //     "num_heads": 16,
    //     "in_chans": 3,
    //     "out_hidden_size": 2048,
    //     "patch_size": 14,
    //     "spatial_merge_size": 2,
    //     "spatial_patch_size": 14,
    //     "window_size": 112,
    //     "fullatt_block_indexes": [
    //       7,
    //       15,
    //       23,
    //       31
    //     ],
    //     "tokens_per_second": 2,
    //     "temporal_patch_size": 2

    let spacial_merge_size = 2;
    let temporal_patch_size = 2;
    let patch_size = 14;
    let fullatt_block_indexes = vec![7, 15, 23, 31];
    let window_size = 112;
    let in_channels = 3;
    let hidden_size = 1280;
    let out_hidden_size = 2048;
    let embed_dim = 1152;
    let num_heads = 16;
    let depth = 32;
    let qwen_vision = QwenVisionTransformer::new(
        spacial_merge_size,
        temporal_patch_size,
        patch_size,
        fullatt_block_indexes,
        window_size,
        in_channels,
        hidden_size,
        out_hidden_size,
        embed_dim,
        num_heads,
        depth,
        &vb,
    )
    .unwrap();

    let out = qwen_vision.rot_pos_emb(&vec![(2, 4, 4)]).unwrap();
    println!("Rotary Pos Emb: {:?}", out);
    let out_first_5_by_5 = out.i((0..5, 0..5)).unwrap();
    let out_first_5_by_5 = out_first_5_by_5.to_vec2::<f32>().unwrap();

    let expected = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            1.0,
            0.630957305431366,
            0.3981071710586548,
            0.25118863582611084,
            0.15848931670188904,
        ],
        [
            1.0,
            0.630957305431366,
            0.3981071710586548,
            0.25118863582611084,
            0.15848931670188904,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    assert_2d_vec_eq(out_first_5_by_5, expected, 1e-2);

    let hidden_states = Tensor::randn(0.0, 1.0, (1, 3, 224, 224), &device).unwrap();
    let grid_thw = vec![(1, 112, 112)];
    qwen_vision.forward(hidden_states, &grid_thw, None).unwrap();
}
