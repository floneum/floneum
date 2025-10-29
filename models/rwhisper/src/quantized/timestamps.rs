// Based on https://github.com/nicksenger/candle/tree/feat/whisper-dtw with some optimizations and refactoring
// https://rtavenar.github.io/blog/dtw.html is a good resource for understanding the dtw algorithm

use core::f32;
use fusor_core::Tensor;
use rayon::iter::*;
use std::num::NonZeroUsize;

use crate::config::{HOP_LENGTH, N_FRAMES, SAMPLE_RATE};

/// Returns the token-level timestamps as a tensor of shape batch x timestamps
pub(super) fn extract_timestamps(
    // A list of (layer, head) pairs to use for timestamp determination
    alignment_heads: &[[usize; 2]],
    cross_attentions: &[Tensor<4, f32>],
    filter_width: NonZeroUsize,
    n_frames: usize,
    mask: Vec<Vec<bool>>,
) -> fusor_core::Result<Vec<Vec<f32>>> {
    // Select relevant cross-attention heads
    let weights = Tensor::stack(
        alignment_heads
            .iter()
            .copied()
            .filter_map(|[layer, head]| cross_attentions.get(layer)?.i((.., head)).ok()),
        0,
    )
    .permute([1, 0, 2, 3])
    .narrow(3, 0, n_frames.min(N_FRAMES) / 2);

    if weights.shape().contains(&0) {
        // No tokens to be aligned
        return Ok(Vec::new());
    }

    // Normalize
    let weights = weights.softmax_last_dim();

    // Smooth
    let weights = &median_filter(
        filter_width,
        weights
            .sub_(&weights.mean_keepdim(weights.rank() - 2))
            .div_(&weights.var_keepdim(weights.rank() - 2).sqrt()),
    )?;

    let cost = weights.mean(1);

    // Do the timewarp
    ((0..weights.shape()[0]).map(|batch_idx| {
        // Exclude any tokens in the mask
        let batch_index_cost = cost.neg().i(batch_idx).cast::<f32>();
        let batch_index_cost = batch_index_cost.to_vec2::<f32>()?;
        let batch_index_cost = batch_index_cost
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| if mask[batch_idx][i] { Some(v) } else { None })
            .collect::<Vec<_>>();
        if batch_index_cost.is_empty() || batch_index_cost[0].is_empty() {
            return Ok(Vec::new());
        }
        let (text_indices, time_indices) = dynamic_time_warp(batch_index_cost)?;

        let jumps = std::iter::once(true)
            .chain(
                text_indices
                    .iter()
                    .zip(text_indices.iter().skip(1))
                    .map(|(a, b)| (b - a) as usize == 1),
            )
            .zip(time_indices)
            .filter_map(|(is_jump, time_index)| {
                if is_jump {
                    Some(time_index / (SAMPLE_RATE / (HOP_LENGTH * 2)) as f32)
                } else {
                    None
                }
            });

        Ok(jumps.collect())
    }))
    .collect::<fusor_core::Result<Vec<_>>>()
}

/// Computes the lowest cost warping path through the provided cost matrix
fn dynamic_time_warp(matrix: Vec<Vec<f32>>) -> fusor_core::Result<(Vec<f32>, Vec<f32>)> {
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

fn median_filter(
    filter_width: NonZeroUsize,
    weights: Tensor<4, f32>,
) -> fusor_core::Result<Tensor<4, f32>> {
    // let filter_width = filter_width.get();
    // let pad_width = filter_width / 2;
    // let [_, _c, _, w] = *weights.shape();
    // if w <= pad_width {
    //     return Ok(weights);
    // }

    // let weights = weights.pad_with_same(3, pad_width, pad_width);
    // let mut medians = vec![];
    // for i in 0..w {
    //     let weights = weights.narrow(3, i, filter_width);
    //     medians.push(weights.unsqueeze(weights.rank() - 2));
    // }

    // medians.par_iter().try_for_each(|weights| {
    //     struct Median {
    //         pad_width: usize,
    //     }

    //     impl InplaceOp1 for Median {
    //         fn name(&self) -> &'static str {
    //             "median"
    //         }

    //         fn cpu_fwd(
    //             &self,
    //             storage: &mut fusor_core::CpuStorage,
    //             layout: &fusor_core::Layout,
    //         ) -> fusor_core::Result<()> {
    //             assert!(layout.is_contiguous());
    //             if let CpuStorage::F32(storage) = storage {
    //                 storage.select_nth_unstable_by(self.pad_width, |a, b| {
    //                     a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    //                 });
    //             } else {
    //                 unimplemented!()
    //             }

    //             Ok(())
    //         }
    //     }
    //     weights.inplace_op1(&Median { pad_width })
    // })?;

    // Tensor::cat(&medians, 3)
    //     .narrow(4, pad_width, 1)
    //     .squeeze(4)?
    //     .to_device(weights.device())
    todo!()
}
