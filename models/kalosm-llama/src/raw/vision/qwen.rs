use std::path::Path;

use candle_core::{quantized::gguf_file, IndexOp, Tensor, D};
use candle_transformers::quantized_var_builder::VarBuilder;
use kalosm_common::KvCache;

use crate::raw::rope::RopeCache;

use super::{
    qwen_image_processing::process_image,
    qwen_patch_merger::Qwen2VLPatchMerger,
    qwen_rope::{get_rope_index, VisionRotaryEmbedding},
    qwen_vision::get_window_index,
    qwen_vision_block::VisionBlock,
    qwen_vision_embed::Qwen2_5VisionPatchEmbed,
    QWEN_EPS,
};

pub(crate) struct QwenVisionTransformer {
    pub(crate) spacial_merge_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
    window_size: usize,
    spacial_merge_unit: usize,
    patch_embed: Qwen2_5VisionPatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: Qwen2VLPatchMerger,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    device: candle_core::Device,
}

impl QwenVisionTransformer {
    pub(crate) fn from_gguf(
        vision_ct: gguf_file::Content,
        vision_file: &Path,
        device: &candle_core::Device,
    ) -> candle_core::Result<Self> {
        let block_count = vision_ct
            .metadata
            .get("clip.vision.block_count")
            .and_then(|x| x.to_u64().ok())
            .unwrap_or(32) as usize;
        let head_count = vision_ct
            .metadata
            .get("clip.vision.attention.head_count")
            .and_then(|x| x.to_u64().ok())
            .unwrap_or(16) as usize;
        let spacial_merge_size = 2;
        let temporal_patch_size = 2;
        let patch_size = vision_ct
            .metadata
            .get("clip.vision.patch_size")
            .and_then(|x| x.to_u64().ok())
            .map(|x| x as usize)
            .unwrap_or(14);
        let fullatt_block = vision_ct
            .metadata
            .get("clip.vision.n_wa_pattern")
            .and_then(|x| x.to_u64().ok())
            .map(|x| {
                let mut v = vec![];
                for i in (0..block_count).step_by(x as usize) {
                    v.push(i as usize);
                }
                v
            })
            .unwrap_or(vec![7, 15, 23, 31]);
        let layer_norm_eps = vision_ct
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(|x| x.to_f64().ok())
            .unwrap_or(QWEN_EPS);
        let hidden_size = vision_ct
            .metadata
            .get("clip.vision.embedding_length")
            .and_then(|x| x.to_u64().ok())
            .unwrap_or(1280) as usize;
        let _feed_forward_length = vision_ct
            .metadata
            .get("clip.vision.feed_forward_length")
            .and_then(|x| x.to_u64().ok())
            .unwrap_or(3420) as usize;
        let out_hidden_size = vision_ct
            .metadata
            .get("clip.vision.projection_dim")
            .and_then(|x| x.to_u64().ok())
            .unwrap_or(2048) as usize;
        let image_mean = vision_ct
            .metadata
            .get("clip.vision.image_mean")
            .and_then(|x| {
                x.to_vec()
                    .ok()
                    .map(|x| x.iter().map(|x| x.to_f32()).collect())
            })
            .transpose()?
            .unwrap_or(vec![
                0.48145467042922974,
                0.45782750844955444,
                0.40821072459220886,
            ]);
        let image_std = vision_ct
            .metadata
            .get("clip.vision.image_std")
            .and_then(|x| {
                x.to_vec()
                    .ok()
                    .map(|x| x.iter().map(|x| x.to_f32()).collect())
            })
            .transpose()?
            .unwrap_or(vec![
                0.2686295509338379,
                0.2613025903701782,
                0.27577710151672363,
            ]);
        let in_channels = 3;

        let vb = VarBuilder::from_gguf(vision_file, device).unwrap();
        Self::new(
            spacial_merge_size,
            temporal_patch_size,
            patch_size,
            fullatt_block,
            112,
            in_channels,
            hidden_size,
            out_hidden_size,
            head_count,
            block_count,
            layer_norm_eps,
            image_mean,
            image_std,
            &vb,
        )
    }

    fn new(
        spacial_merge_size: usize,
        temporal_patch_size: usize,
        patch_size: usize,
        fullatt_block_indexes: Vec<usize>,
        window_size: usize,
        in_channels: usize,
        hidden_size: usize,
        out_hidden_size: usize,
        num_heads: usize,
        depth: usize,
        layer_norm_eps: f64,
        image_mean: Vec<f32>,
        image_std: Vec<f32>,
        vb: &VarBuilder,
    ) -> candle_core::Result<Self> {
        let spacial_merge_unit = spacial_merge_size * spacial_merge_size;
        let patch_embed = Qwen2_5VisionPatchEmbed::new(
            patch_size,
            temporal_patch_size,
            in_channels,
            hidden_size,
            &vb.pp("v.patch_embd"),
        )
        .unwrap();
        let head_dim = hidden_size / num_heads;
        let rope_theta = 10000.0;
        let rotary_pos_emb =
            VisionRotaryEmbedding::new(head_dim / 2, rope_theta, vb.device()).unwrap();
        let blocks = (0..depth)
            .map(|i| {
                VisionBlock::new(
                    &vb.pp(&format!("v.blk.{i}")),
                    num_heads,
                    head_dim,
                    hidden_size,
                    layer_norm_eps,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()
            .unwrap();
        let merger = Qwen2VLPatchMerger::new(
            out_hidden_size,
            hidden_size,
            spacial_merge_size,
            layer_norm_eps,
            &vb,
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
            image_mean,
            image_std,
            device: vb.device().clone(),
        })
    }

    fn rot_pos_emb(&self, grid_thw: &Vec<[u32; 3]>) -> candle_core::Result<Tensor> {
        let device = &self.device;
        let mut pos_ids = vec![];
        for [t, h, w] in grid_thw {
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
        let max_grid_size = grid_thw
            .iter()
            .map(|[_, h, w]| (*h).max(*w))
            .max()
            .unwrap_or_default();
        let rotary_pos_emb_full = self
            .rotary_pos_emb
            .make_embeds(max_grid_size)
            .unwrap()
            .contiguous()
            .unwrap();

        let rotary_pos_emb_0 = rotary_pos_emb_full
            .index_select(&pos_ids.i((.., 0)).unwrap().contiguous().unwrap(), 0)
            .unwrap();

        let rotary_pos_emb_1 = rotary_pos_emb_full
            .index_select(&pos_ids.i((.., 1)).unwrap().contiguous().unwrap(), 0)
            .unwrap();

        let rotary_pos_emb = Tensor::cat(&[rotary_pos_emb_0, rotary_pos_emb_1], D::Minus1).unwrap();

        Ok(rotary_pos_emb)
    }

    pub(crate) fn preprocess_image(
        &self,
        image: &image::DynamicImage,
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
    ) -> candle_core::Result<(Tensor, [u32; 3])> {
        process_image(
            image,
            self.patch_size,
            self.spacial_merge_size,
            min_pixels,
            max_pixels,
            &self.image_mean,
            &self.image_std,
            &self.device,
        )
    }

    pub(crate) fn get_rope_index(
        &self,
        input_ids: &[u32],
        grid_thw: &[[u32; 3]],
        config: &crate::raw::LlamaConfig,
        start_time: u32,
    ) -> candle_core::Result<(Tensor, u32)> {
        let (rope_index, max_time_index) = get_rope_index(
            self.spacial_merge_size,
            config.image_pad_token.unwrap(),
            config.video_pad_token.unwrap(),
            config.vision_start_token.unwrap(),
            input_ids,
            grid_thw,
            &[],
            start_time,
        );

        let tensor = Tensor::from_iter(
            rope_index.iter().flat_map(|x| [x.x, x.y, x.time]),
            &self.device,
        )?
        .reshape(((), 3))?
        .t()?;

        Ok((tensor, max_time_index))
    }

    pub(crate) fn forward_image(
        &self,
        pixels: &Tensor,
        grid: [u32; 3],
    ) -> candle_core::Result<Tensor> {
        self.forward(pixels, &vec![grid], None)
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        grid_thw: &Vec<[u32; 3]>,
        mut cache: Option<&mut KvCache>,
    ) -> candle_core::Result<Tensor> {
        // println!("input");
        // let vec2 = hidden_states
        //     .i((..25, ..25))
        //     .unwrap()
        //     .to_dtype(candle_core::DType::F32)
        //     .unwrap()
        //     .to_vec2::<f32>()
        //     .unwrap();
        // for list in vec2.iter() {
        //     println!("{:?}", list);
        // }
        let hidden_states = self.patch_embed.forward(&hidden_states).unwrap();
        // println!("input patch");
        // let vec2 = hidden_states
        //     .i((..25, ..25))
        //     .unwrap()
        //     .to_dtype(candle_core::DType::F32)
        //     .unwrap()
        //     .to_vec2::<f32>()
        //     .unwrap();
        // for list in vec2.iter() {
        //     println!("{:?}", list);
        // }
        let rotary_pos_emb = self.rot_pos_emb(grid_thw).unwrap();
        let (window_index, mut cu_window_seqlens) = get_window_index(
            grid_thw
                .iter()
                .map(|[t, h, w]| (*t as usize, *h as usize, *w as usize)),
            self.window_size,
            self.spacial_merge_size,
            self.spacial_merge_unit,
            self.patch_size,
            &self.device,
        )
        .unwrap();
        let mut last_item = None;
        cu_window_seqlens.retain(|&x| {
            if last_item.is_some_and(|y| y == x) {
                return false;
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
        let rope_cache =
            RopeCache::from_parts(rotary_pos_emb.cos().unwrap(), rotary_pos_emb.sin().unwrap())
                .unwrap();

        let cu_seqlens = grid_thw
            .iter()
            .flat_map(|[t, h, w]| std::iter::repeat_n((*h * *w) as u32, *t as usize))
            .map({
                let mut sum = 0;
                move |x| {
                    sum += x;
                    sum
                }
            });

        let cu_seqlens = std::iter::once(0).chain(cu_seqlens).collect::<Vec<_>>();

        // println!("start");
        // let vec2 = hidden_states
        //     .i((..25, ..25))
        //     .unwrap()
        //     .to_dtype(candle_core::DType::F32)
        //     .unwrap()
        //     .to_vec2::<f32>()
        //     .unwrap();
        // for list in vec2.iter() {
        //     println!("{:?}", list);
        // }

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
            // println!("{layer_num} {}/{}", layer_num, self.blocks.len());
            // let vec2 = hidden_states
            //     .i((..25, ..25))
            //     .unwrap()
            //     .to_vec2::<f32>()
            //     .unwrap();
            // for list in vec2.iter() {
            //     println!("{:?}", list);
            // }
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
    use kalosm_common::{accelerated_device_if_available, Cache};

    let device = accelerated_device_if_available().unwrap();
    let path = Cache::default()
        .get(
            &kalosm_model_types::FileSource::HuggingFace {
                model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
                revision: "main".into(),
                file: "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf".into(),
            },
            |_| {},
        )
        .await
        .unwrap();
    let vb = VarBuilder::from_gguf(&path, &device).unwrap();

    let spacial_merge_size = 2;
    let temporal_patch_size = 2;
    let patch_size = 14;
    let fullatt_block_indexes = vec![7, 15, 23, 31];
    let window_size = 112;
    let in_channels = 3;
    let hidden_size = 1280;
    let out_hidden_size = 2048;
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
        num_heads,
        depth,
        QWEN_EPS,
        [0.2686295509338379, 0.2613025903701782, 0.27577710151672363].to_vec(),
        [0.48145467042922974, 0.45782750844955444, 0.40821072459220886].to_vec(),
        &vb,
    )
    .unwrap();

    let out = qwen_vision.rot_pos_emb(&vec![[2, 4, 4]]).unwrap();
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

    let hidden_states = Tensor::randn(0.0f32, 1.0, (1944, 1176), &candle_core::Device::Cpu)
        .unwrap()
        .to_device(&device)
        .unwrap();
    let grid_thw = vec![[1, 36, 54]];
    let out = qwen_vision
        .forward(&hidden_states, &grid_thw, None)
        .unwrap();
    println!(
        "Qwen Vision: {:?}",
        out.i((0..5, 0..5)).unwrap().to_vec2::<f32>().unwrap()
    );

    // download image from https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
    let image_bytes =
        reqwest::get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    let image = image::load_from_memory(&image_bytes).unwrap();

    let (pixels, grid) = qwen_vision
        .preprocess_image(&image, Some(256 * 28 * 28), Some(512 * 28 * 28))
        .unwrap();
    let out = qwen_vision.forward_image(&pixels, grid).unwrap();
    println!(
        "Qwen Vision: {:?}",
        out.i((0..5, 0..5)).unwrap().to_vec2::<f32>().unwrap()
    );
}
