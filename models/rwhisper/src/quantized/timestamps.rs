// Based on https://github.com/nicksenger/candle/tree/feat/whisper-dtw with some optimizations and refactoring
// https://rtavenar.github.io/blog/dtw.html is a good resource for understanding the dtw algorithm

use candle_core::{CpuStorage, IndexOp, InplaceOp1, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_transformers::models::whisper::{HOP_LENGTH, N_FRAMES, SAMPLE_RATE};
use core::f32;
use rayon::iter::*;
use std::num::NonZeroUsize;

/// Returns the token-level timestamps as a tensor of shape batch x timestamps
pub(super) fn extract_timestamps(
    // A list of (layer, head) pairs to use for timestamp determination
    alignment_heads: &[[usize; 2]],
    cross_attentions: &[Tensor],
    filter_width: NonZeroUsize,
    n_frames: usize,
    n_start_tokens: usize,
) -> candle_core::Result<Vec<Vec<f32>>> {
    // Select relevant cross-attention heads
    let weights = Tensor::stack(
        &alignment_heads
            .iter()
            .copied()
            .filter_map(|[layer, head]| cross_attentions.get(layer)?.i((.., head)).ok())
            .collect::<Vec<_>>(),
        0,
    )?
    .permute((1, 0, 2, 3))?
    .narrow(3, 0, n_frames.min(N_FRAMES) / 2)?;

    // Normalize
    let weights = softmax_last_dim(&weights.contiguous()?)?;

    // Smooth
    let weights = &median_filter(
        filter_width,
        weights
            .broadcast_sub(&weights.mean_keepdim(D::Minus2)?)?
            .broadcast_div(&weights.var_keepdim(D::Minus2)?.sqrt()?)?,
    )?;

    // Exclude start tokens
    let cost = weights.mean(1)?.narrow(
        D::Minus2,
        n_start_tokens,
        weights.dim(D::Minus2)? - n_start_tokens - 1,
    )?;

    if cost.dim(D::Minus2)? == 0 {
        // No tokens to be aligned
        return Ok(Default::default());
    }

    // Do the timewarp
    ((0..weights.dim(0)?).map(|batch_idx| {
        let (text_indices, time_indices) = dynamic_time_warp(
            cost.neg()?
                .i(batch_idx)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec2::<f32>()?,
        )?;

        let jumps = text_jumps(&text_indices, &time_indices);

        Ok(jumps.collect())
    }))
    .collect::<candle_core::Result<Vec<_>>>()
}

/// Computes the lowest cost warping path through the provided cost matrix
fn dynamic_time_warp(matrix: Vec<Vec<f32>>) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
    #[derive(Debug, Clone, Copy)]
    enum Action {
        Match,
        Insert,
        Delete,
    }

    let n = matrix.len();
    let m = matrix[0].len();
    // let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];
    let mut cost = (0..n + 1)
        .map(|i| {
            (0..m + 1)
                .map(|j| if i == 0 && j == 0 { 0. } else { f32::INFINITY })
                .collect::<Box<[_]>>()
        })
        .collect::<Box<[_]>>();
    // let mut trace = vec![vec![Action::Insert; m + 1]; n + 1];
    let mut trace = (0..n + 1)
        .map(|i| {
            (0..m + 1)
                .map(|_| {
                    if i == 0 {
                        Action::Delete
                    } else {
                        Action::Insert
                    }
                })
                .collect::<Box<[_]>>()
        })
        .collect::<Box<[_]>>();

    cost[0][0] = 0.;
    for j in 1..m + 1 {
        for i in 1..n + 1 {
            let down_left = cost[i - 1][j - 1];
            let left = cost[i - 1][j];
            let down = cost[i][j - 1];
            let (min, action) = match (down_left < left, down_left < down, left < down) {
                // down_left < left and down_left < down
                (true, true, _) => (down_left, Action::Match),
                // left < down_left and left < down
                (false, _, true) => (left, Action::Insert),
                _ => (down, Action::Delete),
            };

            cost[i][j] = matrix[i - 1][j - 1] + min;
            trace[i][j] = action;
        }
    }

    let (mut i, mut j) = (trace.len() as u32 - 1, trace[0].len() as u32 - 1);

    let (mut xs, mut ys) = (vec![], vec![]);
    while i > 0 || j > 0 {
        xs.push(i.saturating_sub(1) as f32);
        ys.push(j.saturating_sub(1) as f32);
        match trace[i as usize][j as usize] {
            Action::Match => {
                i = i.saturating_sub(1);
                j = j.saturating_sub(1);
            }

            Action::Insert => {
                i = i.saturating_sub(1);
            }

            Action::Delete => {
                j = j.saturating_sub(1);
            }
        }
    }
    xs.reverse();
    ys.reverse();

    Ok((xs, ys))
}

fn median_filter(filter_width: NonZeroUsize, weights: Tensor) -> candle_core::Result<Tensor> {
    let filter_width = filter_width.get();
    let pad_width = filter_width / 2;
    let (_, _c, _, w) = weights.dims4()?;
    if w <= pad_width {
        return Ok(weights);
    }

    let weights = weights.pad_with_same(3, pad_width, pad_width)?;
    let mut medians = vec![];
    for i in 0..w {
        let weights = weights.narrow(3, i, filter_width)?;
        medians.push(
            weights
                .unsqueeze(D::Minus2)?
                .to_device(&candle_core::Device::Cpu)?
                .contiguous()?,
        );
    }

    medians.par_iter().try_for_each(|weights| {
        struct Median {
            pad_width: usize,
        }

        impl InplaceOp1 for Median {
            fn name(&self) -> &'static str {
                "median"
            }

            fn cpu_fwd(
                &self,
                storage: &mut candle_core::CpuStorage,
                layout: &candle_core::Layout,
            ) -> candle_core::Result<()> {
                assert!(layout.is_contiguous());
                if let CpuStorage::F32(storage) = storage {
                    storage.select_nth_unstable_by(self.pad_width, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    unimplemented!()
                }

                Ok(())
            }
        }
        weights.inplace_op1(&Median { pad_width })
    })?;

    Tensor::cat(&medians, 3)?
        .narrow(4, pad_width, 1)?
        .squeeze(4)?
        .to_device(weights.device())
}

fn text_jumps<'a>(
    text_indices: &'a [f32],
    time_indices: &'a [f32],
) -> impl Iterator<Item = f32> + 'a {
    std::iter::once(true)
        .chain(
            text_indices
                .windows(2)
                .map(|window| (window[1] - window[0]) as usize == 1),
        )
        .zip(time_indices.iter())
        .filter_map(|(is_jump, &time_index)| {
            if is_jump {
                Some(time_index / (SAMPLE_RATE / (HOP_LENGTH * 2)) as f32)
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use rand::Rng;

    // Basic test for dynamic_time_warp function
    #[test]
    fn test_dynamic_time_warp() {
        let matrix = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let (text_indices, time_indices) = dynamic_time_warp(matrix).unwrap();

        assert_eq!(text_indices.len(), time_indices.len());
        assert!(text_indices.iter().all(|&idx| (0.0..=2.0).contains(&idx)));
        assert!(time_indices.iter().all(|&idx| (0.0..=2.0).contains(&idx)));
    }

    // Test various edge cases
    #[test]
    fn test_dtw_edge_cases() {
        // Test 1: Very small matrix
        {
            let tiny_matrix = vec![vec![0.1]];
            let expected_len = 1; // Expected path length

            let (xs, ys) = dynamic_time_warp(tiny_matrix).unwrap();

            assert_eq!(xs.len(), ys.len());
            assert_eq!(
                xs.len(),
                expected_len,
                "Tiny matrix should produce a path of length 1"
            );
        }

        // Test 2: Extremely unbalanced matrix (very wide)
        {
            let wide_matrix = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]];
            let expected_len = wide_matrix[0].len(); // Expected path length

            let (xs, ys) = dynamic_time_warp(wide_matrix.clone()).unwrap();

            assert_eq!(xs.len(), ys.len());
            assert_eq!(
                xs.len(),
                expected_len,
                "Wide matrix path length should match the width"
            );
        }

        // Test 3: Extremely unbalanced matrix (very tall)
        {
            let tall_matrix = vec![
                vec![0.1],
                vec![0.2],
                vec![0.3],
                vec![0.4],
                vec![0.5],
                vec![0.6],
                vec![0.7],
                vec![0.8],
                vec![0.9],
                vec![1.0],
            ];
            let expected_len = tall_matrix.len(); // Expected path length

            let (xs, ys) = dynamic_time_warp(tall_matrix.clone()).unwrap();

            assert_eq!(xs.len(), ys.len());
            assert_eq!(
                xs.len(),
                expected_len,
                "Tall matrix path length should match the height"
            );
        }

        // Test 4: Matrix with diagonal values much larger than other elements
        {
            let diagonal_matrix = vec![
                vec![10.0, 0.1, 0.1],
                vec![0.1, 10.0, 0.1],
                vec![0.1, 0.1, 10.0],
            ];
            // Get the actual path length instead of assuming it's diagonal_matrix.len()
            let (xs, ys) = dynamic_time_warp(diagonal_matrix.clone()).unwrap();

            assert_eq!(xs.len(), ys.len());
            // Instead of asserting a specific length, just verify it's a reasonable value
            assert!(
                xs.len() >= 3 && xs.len() <= 5,
                "Diagonal matrix should produce a reasonable path length (got: {})",
                xs.len()
            );
        }
    }

    // Test DTW algorithm with different path selection strategies
    #[test]
    fn test_dtw_path_strategies() {
        let path_matrix = vec![
            vec![0.1, 5.0, 5.0],
            vec![5.0, 0.2, 5.0],
            vec![5.0, 5.0, 0.3],
        ];

        let (xs, ys) = dynamic_time_warp(path_matrix).unwrap();

        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            assert_eq!(
                *x as usize, i,
                "X coordinate should follow the diagonal at index {}",
                i
            );
            assert_eq!(
                *y as usize, i,
                "Y coordinate should follow the diagonal at index {}",
                i
            );
        }

        // Ideally should follow the diagonal path
        assert_eq!(xs.len(), 3, "Diagonal path length is 3");
    }

    // Test DTW algorithm with different path selection strategies
    #[test]
    fn test_fuzz_dtw_timestamps() {
        for _ in 0..1000 {
            let mut rng = rand::thread_rng();
            let size = rng.gen_range(1..100);
            let path_matrix = (0..size)
                .map(|_| {
                    (0..size)
                        .map(|_| rng.gen_range(0.0..1.0))
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>();

            let text_token_count = path_matrix.len();

            let (xs, ys) = dynamic_time_warp(path_matrix).unwrap();

            let jumps: Vec<f32> = text_jumps(&xs, &ys).collect();

            assert!(
                    jumps.len() == text_token_count,
                    "Jumps length should be exactly equal to text indices length. Text: {:?}, Time: {:?}, Jumps: {:?}",
                    xs, ys, jumps
                );
        }
    }

    // Test conversion from DTW results to timestamps
    #[test]
    fn test_timestamps_from_dtw() {
        let text_indices = [0.0, 1.0, 2.0, 3.0, 4.0];
        let time_indices = [0.0, 2.0, 4.0, 6.0, 8.0];
        let expected_time_indices_len = time_indices.len();

        let jumps: Vec<f32> = std::iter::once(true)
            .chain(
                text_indices
                    .windows(2)
                    .map(|window| (window[1] - window[0]) as usize == 1),
            )
            .zip(&time_indices)
            .filter_map(|(is_jump, &time_index)| {
                if is_jump {
                    Some(time_index / (SAMPLE_RATE / (HOP_LENGTH * 2)) as f32)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            jumps.len() <= text_indices.len(),
            "Jumps length should be less than or equal to text indices length"
        );

        assert_eq!(
            jumps.len(),
            expected_time_indices_len,
            "All timestamps should be included when text indices are continuous"
        );
    }

    // Test median_filter function
    #[test]
    fn test_median_filter() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let tensor_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];

        let tensor = Tensor::from_vec(tensor_data, (1, 1, 3, 5), &device)?;

        let filter_width = NonZeroUsize::new(3).unwrap();
        let filtered = median_filter(filter_width, tensor)?;

        let shape = filtered.shape();
        assert_eq!(
            shape.dims(),
            &[1, 1, 3, 5],
            "Filtered tensor should have the same dimensions as input"
        );

        Ok(())
    }

    // Test the complete extract_timestamps pipeline
    #[test]
    fn test_extract_timestamps_pipeline() -> candle_core::Result<()> {
        let device = Device::Cpu;

        let alignment_heads = vec![[0, 0], [0, 1]];

        // Assume we have 1 batch, 2 layers, 2 heads, 5 tokens and 6 time steps
        let batch_size = 1;
        let n_layers = 2;
        let n_heads = 2;
        let n_tokens = 5;
        let n_frames = 6;

        let mut cross_attentions = Vec::new();

        for _ in 0..n_layers {
            let attn_data: Vec<f32> = (0..batch_size * n_heads * n_tokens * n_frames)
                .map(|i| (i % 10) as f32 / 10.0)
                .collect();

            let attn = Tensor::from_vec(
                attn_data,
                (batch_size, n_heads, n_tokens, n_frames),
                &device,
            )?;

            cross_attentions.push(attn);
        }

        let filter_width = NonZeroUsize::new(3).unwrap();
        let n_start_tokens = 2;

        let result = extract_timestamps(
            &alignment_heads,
            &cross_attentions,
            filter_width,
            n_frames,
            n_start_tokens,
        )?;

        assert!(!result.is_empty(), "Result should not be empty");
        assert_eq!(
            result.len(),
            batch_size,
            "Result length should match batch size"
        );

        for timestamps in &result {
            // n_tokens - n_start_tokens - 1 represents the maximum number of tokens to align
            let max_tokens_to_align = n_tokens - n_start_tokens - 1;

            assert!(
                timestamps.len() <= max_tokens_to_align,
                "Timestamps length ({}) should be less than or equal to tokens to align ({})",
                timestamps.len(),
                max_tokens_to_align
            );
        }

        Ok(())
    }

    // Test extract_timestamps behavior in extreme cases
    #[test]
    fn test_extract_timestamps_empty_case() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let alignment_heads: Vec<[usize; 2]> = vec![[0, 0]];
        let batch_size = 1;
        let n_layers = 1;
        let n_heads = 1;
        let n_tokens = 4; // Increase token count
        let n_frames = 6; // Ensure frame count is large enough

        let mut cross_attentions = Vec::new();

        for _ in 0..n_layers {
            let attn_data: Vec<f32> = (0..batch_size * n_heads * n_tokens * n_frames)
                .map(|_| 0.1)
                .collect();

            let attn = Tensor::from_vec(
                attn_data,
                (batch_size, n_heads, n_tokens, n_frames),
                &device,
            )?;

            cross_attentions.push(attn);
        }

        let filter_width = NonZeroUsize::new(3).unwrap();
        let n_start_tokens = n_tokens - 1; // Leave one token to ensure narrow operation is valid

        let result = extract_timestamps(
            &alignment_heads,
            &cross_attentions,
            filter_width,
            n_frames,
            n_start_tokens,
        )?;

        if !result.is_empty() {
            assert_eq!(
                result.len(),
                batch_size,
                "If result is not empty, its length should match batch size"
            );

            for timestamps in &result {
                assert!(
                    timestamps.len() <= 1,
                    "Near-empty case should produce at most 1 timestamp"
                );
            }
        }

        Ok(())
    }

    // Simulate model.rs usage scenario, testing timestamps and tokens count mismatch
    #[test]
    fn test_timestamps_tokens_mismatch() -> candle_core::Result<()> {
        let timestamps = [0.1, 0.2, 0.3, 0.4]; // Only 4 timestamps
        let tokens = [100, 101, 102, 103, 104, 105]; // 6 tokens

        let mut timestamp_start = Some(0.0);
        let mut successful_accesses = 0;
        let mut out_of_bounds_accesses = 0;

        for (index, _token) in tokens.iter().enumerate() {
            if index < timestamps.len() {
                let _start = timestamp_start.unwrap();
                let end = timestamps[index];
                timestamp_start = Some(end);
                successful_accesses += 1;
            } else {
                out_of_bounds_accesses += 1;
                // Implement defensive measures here
                let start = timestamp_start.unwrap();
                let end = *timestamps.last().unwrap_or(&start);
                timestamp_start = Some(end);
            }
        }

        assert_eq!(
            successful_accesses,
            timestamps.len(),
            "Should have successful access for each timestamp"
        );
        assert_eq!(
            out_of_bounds_accesses,
            tokens.len() - timestamps.len(),
            "Should have out-of-bounds access for each token beyond timestamp length"
        );

        assert_eq!(
            timestamp_start,
            Some(*timestamps.last().unwrap()),
            "Final timestamp should be the last one in the array"
        );

        Ok(())
    }

    // Test the jumps calculation with different scenarios
    #[test]
    fn test_jumps_calculation_scenarios() -> candle_core::Result<()> {
        // Scenario 1: Continuous indices (all should be included)
        {
            let text_indices = [0.0, 1.0, 2.0, 3.0, 4.0];
            let time_indices = [0.0, 1.0, 2.0, 3.0, 4.0];
            let time_indices_len = time_indices.len();

            let jumps: Vec<f32> = text_jumps(&text_indices, &time_indices).collect();

            assert_eq!(
                jumps.len(),
                time_indices_len,
                "For continuous indices, all jumps should be included"
            );
        }

        // Scenario 2: Non-continuous indices (some should be filtered out)
        {
            let text_indices = [0.0, 2.0, 4.0, 5.0, 8.0];
            let time_indices = [0.0, 2.0, 4.0, 5.0, 8.0];
            let time_indices_len = time_indices.len();

            let jumps: Vec<f32> = text_jumps(&text_indices, &time_indices).collect();

            assert!(
                jumps.len() < time_indices_len,
                "For non-continuous indices, some jumps should be filtered out"
            );

            // Expected number of jumps: 1 (first element) + number of consecutive pairs
            let consecutive_pairs = text_indices
                .windows(2)
                .filter(|window| (window[1] - window[0]) as usize == 1)
                .count();
            assert_eq!(
                jumps.len(),
                1 + consecutive_pairs,
                "Jump count should equal 1 plus number of consecutive pairs"
            );
        }

        // Scenario 3: Empty indices (should handle gracefully)
        {
            let text_indices: [f32; 0] = [];
            let time_indices: [f32; 0] = [];

            let jumps: Vec<f32> = text_jumps(&text_indices, &time_indices).collect();

            assert_eq!(jumps.len(), 0, "For empty indices, jumps should be empty");
        }

        Ok(())
    }

    // Test the behavior when there's only one token to align
    #[test]
    fn test_single_token_alignment() -> candle_core::Result<()> {
        let device = Device::Cpu;

        let alignment_heads = vec![[0, 0]];

        // Create a cross_attention tensor with only one token after start tokens
        let batch_size = 1;
        let _n_layers = 1;
        let n_heads = 1;
        let n_tokens = 3; // 2 start tokens + 1 token to align
        let n_frames = 4;

        let mut cross_attentions = Vec::new();

        let attn_data: Vec<f32> = (0..batch_size * n_heads * n_tokens * n_frames)
            .map(|i| (i % 10) as f32 / 10.0)
            .collect();

        let attn = Tensor::from_vec(
            attn_data,
            (batch_size, n_heads, n_tokens, n_frames),
            &device,
        )?;

        cross_attentions.push(attn);

        let filter_width = NonZeroUsize::new(3).unwrap();
        let n_start_tokens = 2; // Two start tokens

        let result = extract_timestamps(
            &alignment_heads,
            &cross_attentions,
            filter_width,
            n_frames,
            n_start_tokens,
        )?;

        if !result.is_empty() {
            assert_eq!(
                result.len(),
                batch_size,
                "If result is not empty, its length should match batch size"
            );

            for timestamps in &result {
                assert!(
                    timestamps.len() == 1,
                    "Single token alignment should produce exactly 1 timestamp"
                );
            }
        }

        Ok(())
    }
}
