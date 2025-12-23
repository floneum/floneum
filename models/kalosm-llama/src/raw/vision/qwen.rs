use std::path::Path;

use fusor_core::{
    cache::KvCache, CastTensor, Device, FloatDataType, Result, Tensor, VarBuilder, D,
};
use fusor_gguf::GgufMetadata;

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

pub(crate) struct QwenVisionTransformer<F: FloatDataType = f32> {
    pub(crate) spacial_merge_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
    window_size: usize,
    spacial_merge_unit: usize,
    patch_embed: Qwen2_5VisionPatchEmbed<F>,
    rotary_pos_emb: VisionRotaryEmbedding<F>,
    blocks: Vec<VisionBlock<F>>,
    merger: Qwen2VLPatchMerger<F>,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    device: Device,
}

impl<F: FloatDataType> QwenVisionTransformer<F>
where
    f32: CastTensor<F>,
    F: CastTensor<f32>,
{
    pub(crate) fn from_gguf(
        vision_ct: GgufMetadata,
        vision_file: &Path,
        device: &Device,
    ) -> Result<Self> {
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
            .map(|x| generate_full_attention_blocks(block_count, x))
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
                x.to_array()
                    .ok()
                    .map(|x| x.iter().map(|x| x.to_f32()).collect())
            })
            .transpose()?
            .unwrap_or(vec![0.481_454_67, 0.457_827_5, 0.408_210_72]);
        let image_std = vision_ct
            .metadata
            .get("clip.vision.image_std")
            .and_then(|x| {
                x.to_array()
                    .ok()
                    .map(|x| x.iter().map(|x| x.to_f32()).collect())
            })
            .transpose()?
            .unwrap_or(vec![0.268_629_55, 0.261_302_6, 0.275_777_1]);
        let in_channels = 3;

        let reader = std::fs::File::open(vision_file)?;
        let mut buffered_reader = std::io::BufReader::new(reader);
        let mut vb = VarBuilder::from_gguf(&mut buffered_reader)?;
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
            &mut vb,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        vb: &mut VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let spacial_merge_unit = spacial_merge_size * spacial_merge_size;
        let patch_embed = Qwen2_5VisionPatchEmbed::new(
            patch_size,
            temporal_patch_size,
            in_channels,
            hidden_size,
            &mut vb.pp("v.patch_embd"),
            device,
        )?;
        let temporal_patch_size = patch_embed.temporal_patch_size();
        let head_dim = hidden_size / num_heads;
        let rope_theta = 10000.0;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, rope_theta, device)?;
        let blocks = (0..depth)
            .map(|i| {
                VisionBlock::new(
                    &mut vb.pp(format!("v.blk.{i}")),
                    device,
                    num_heads,
                    head_dim,
                    hidden_size,
                    layer_norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let merger = Qwen2VLPatchMerger::new(
            hidden_size,
            out_hidden_size,
            spacial_merge_size,
            layer_norm_eps,
            vb,
            device,
        )?;

        Ok(Self {
            spacial_merge_size,
            patch_size,
            temporal_patch_size,
            fullatt_block_indexes,
            window_size,
            spacial_merge_unit,
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            image_mean,
            image_std,
            device: device.clone(),
        })
    }

    fn rot_pos_emb(&self, grid_thw: &Vec<[u32; 3]>) -> Result<Tensor<2, F>> {
        let device = &self.device;
        let mut pos_ids = vec![];
        for [t, h, w] in grid_thw {
            let hpos_ids = Tensor::arange(device, 0, *h)
                .unsqueeze(1)
                .expand([*h as usize, *w as usize])
                .reshape([
                    *h as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                    *w as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                ])
                .permute([0, 2, 1, 3])
                .flatten_all();

            let wpos_ids = Tensor::arange(device, 0, *w)
                .unsqueeze(0)
                .expand([*h as usize, *w as usize])
                .reshape([
                    *h as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                    *w as usize / self.spacial_merge_size,
                    self.spacial_merge_size,
                ])
                .permute([0, 2, 1, 3])
                .flatten_all();
            let pos_id = Tensor::stack([hpos_ids, wpos_ids], D::Minus1).repeat([*t as usize, 1]);
            pos_ids.push(pos_id);
        }
        let pos_ids = Tensor::cat(pos_ids, 0);
        let max_grid_size = grid_thw
            .iter()
            .map(|[_, h, w]| (*h).max(*w))
            .max()
            .unwrap_or_default();
        let rotary_pos_emb_full = self.rotary_pos_emb.make_embeds(max_grid_size)?;

        let rotary_pos_emb_0 = rotary_pos_emb_full.index_select(0, &pos_ids.i((.., 0)));

        let rotary_pos_emb_1 = rotary_pos_emb_full.index_select(0, &pos_ids.i((.., 1)));

        let rotary_pos_emb = Tensor::cat([rotary_pos_emb_0, rotary_pos_emb_1], D::Minus1);

        Ok(rotary_pos_emb)
    }

    pub(crate) fn preprocess_image(
        &self,
        image: &image::DynamicImage,
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
    ) -> Result<(Tensor<2, f32>, [u32; 3])> {
        process_image(
            image,
            self.patch_size,
            self.spacial_merge_size,
            self.temporal_patch_size,
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
        config: &crate::raw::LlamaConfig<F>,
        start_time: u32,
    ) -> Result<(Tensor<2, u32>, u32)> {
        let (rope_index, max_time_index) = get_rope_index(
            self.spacial_merge_size,
            config.image_pad_token.expect("token"),
            config.video_pad_token.expect("token"),
            config.vision_start_token.expect("token"),
            input_ids,
            grid_thw,
            &[],
            start_time,
        );

        let tensor = Tensor::new(
            &self.device,
            &rope_index
                .iter()
                .flat_map(|x| [x.x, x.y, x.time])
                .collect::<Vec<_>>(),
        )
        .reshape(((), 3))
        .t();

        Ok((tensor, max_time_index))
    }

    pub(crate) fn forward_image(
        &self,
        pixels: &Tensor<2, F>,
        grid: [u32; 3],
    ) -> Result<Tensor<2, F>> {
        self.forward(pixels, &vec![grid], None)
    }

    fn forward(
        &self,
        hidden_states: &Tensor<2, F>,
        grid_thw: &Vec<[u32; 3]>,
        mut cache: Option<&mut KvCache<F>>,
    ) -> Result<Tensor<2, F>> {
        let hidden_states = self.patch_embed.forward(hidden_states)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let (window_index, mut cu_window_seqlens) = get_window_index(
            grid_thw
                .iter()
                .map(|[t, h, w]| (*t as usize, *h as usize, *w as usize)),
            self.window_size,
            self.spacial_merge_size,
            self.spacial_merge_unit,
            self.patch_size,
            &self.device,
        )?;
        let mut last_item = None;
        cu_window_seqlens.retain(|&x| {
            if last_item.is_some_and(|y| y == x) {
                return false;
            }
            last_item = Some(x);
            true
        });

        let seq_len = hidden_states.shape()[0];
        let hidden_states = hidden_states.reshape((
            seq_len / self.spacial_merge_unit,
            self.spacial_merge_unit,
            (),
        ));
        let hidden_states = hidden_states.index_select(0, &window_index);
        let mut hidden_states = hidden_states.reshape((seq_len, ()));
        let rotary_pos_emb = rotary_pos_emb.reshape((
            seq_len / self.spacial_merge_unit,
            self.spacial_merge_unit,
            (),
        ));
        let rotary_pos_emb = rotary_pos_emb.index_select(0, &window_index);
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()));
        let rope_cache = RopeCache::from_parts(rotary_pos_emb.cos(), rotary_pos_emb.sin());

        let cu_seqlens = grid_thw
            .iter()
            .flat_map(|[t, h, w]| std::iter::repeat_n(*h * *w, *t as usize))
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
                cache.as_deref_mut(),
            )?;
        }

        let hidden_states = self.merger.forward(&hidden_states);
        let reverse_indices = {
            let indices_flat = window_index.reshape([window_index.shape().iter().product(), 1]);
            let indices_vec2 = pollster::block_on(indices_flat.to_vec2())?;
            let indices: Vec<u32> = indices_vec2.into_iter().map(|v| v[0]).collect();
            let mut indices_with_pos: Vec<_> = indices.iter().enumerate().collect();
            indices_with_pos.sort_by_key(|&(_, &val)| val);
            let sorted_indices: Vec<u32> = indices_with_pos
                .into_iter()
                .map(|(i, _)| i as u32)
                .collect();
            Tensor::new(&self.device, &sorted_indices)
        };
        let hidden_states = hidden_states.index_select(0, &reverse_indices);

        Ok(hidden_states)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_loading_qwen_vision() {
    use kalosm_common::Cache;

    // Skip in macOS CI
    #[cfg(target_os = "macos")]
    if std::env::var("CI").is_ok() {
        return;
    }

    let device = Device::new().await.unwrap();
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
    let reader = std::fs::File::open(&path).unwrap();
    let mut buffered_reader = std::io::BufReader::new(reader);
    let mut vb = VarBuilder::from_gguf(&mut buffered_reader).unwrap();

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
        [0.268_629_55, 0.261_302_6, 0.275_777_1].to_vec(),
        [0.481_454_67, 0.457_827_5, 0.408_210_72].to_vec(),
        &mut vb,
        &device,
    )
    .unwrap();

    let out = qwen_vision.rot_pos_emb(&vec![[2, 4, 4]]).unwrap();
    println!("Rotary Pos Emb: {out:?}");
    let out_first_5_by_5 = out.i((0..5, 0..5));
    let out_first_5_by_5: Vec<Vec<f32>> = out_first_5_by_5.to_vec2().await.unwrap();

    let expected: [[f32; 5]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.630_957_3, 0.398_107_17, 0.251_188_64, 0.158_489_32],
        [1.0, 0.630_957_3, 0.398_107_17, 0.251_188_64, 0.158_489_32],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    for i in 0..5 {
        for j in 0..5 {
            assert!((out_first_5_by_5[i][j] - expected[i][j]).abs() < 1e-2);
        }
    }

    // Create random tensor using fusor-core
    let hidden_states_data: Vec<f32> = (0..1944 * 1176).map(|_| rand::random()).collect();
    let hidden_states = Tensor::new(&device, &hidden_states_data).reshape([1944, 1176]);

    let grid_thw = vec![[1, 36, 54]];
    let out = qwen_vision
        .forward(&hidden_states, &grid_thw, None)
        .unwrap();
    println!(
        "Qwen Vision: {:?}",
        out.i((0..5, 0..5)).to_vec2().await.unwrap()
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
        out.i((0..5, 0..5)).to_vec2().await.unwrap()
    );
}

fn generate_full_attention_blocks(block_count: usize, n_wa_pattern: u64) -> Vec<usize> {
    let n_wa_pattern = n_wa_pattern as usize;
    let mut fullatt_block_indexes = vec![];
    for i in (n_wa_pattern - 1..block_count).step_by(n_wa_pattern) {
        fullatt_block_indexes.push(i);
    }
    fullatt_block_indexes
}

#[test]
fn test_generate_full_attention_blocks() {
    let block_count = 32;
    let n_wa_pattern = 8;
    let fullatt_block_indexes = generate_full_attention_blocks(block_count, n_wa_pattern);
    assert_eq!(fullatt_block_indexes, [7, 15, 23, 31])
}
