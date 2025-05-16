use candle_core::{IndexOp, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};
use candle_transformers::quantized_var_builder::VarBuilder;

pub(crate) struct Qwen2_5VisionPatchEmbed {
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    first_frame_conv: candle_nn::Conv2d,
    second_frame_conv: candle_nn::Conv2d,
}

impl Qwen2_5VisionPatchEmbed {
    pub fn new(
        patch_size: usize,
        temporal_patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        vb: &VarBuilder,
    ) -> candle_core::Result<Self> {
        let device = vb.device();
        let ws = vb
            .get(
                (
                    embed_dim,
                    in_channels,
                    temporal_patch_size,
                    patch_size,
                    patch_size,
                ),
                "proj.weight",
            )?
            .dequantize_f16(device)?;

        Self::from_weight(patch_size, temporal_patch_size, in_channels, embed_dim, &ws)
    }

    pub fn from_weight(
        patch_size: usize,
        temporal_patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        weight: &Tensor,
    ) -> candle_core::Result<Self> {
        assert_eq!(
            temporal_patch_size, 2,
            "Only 2 temporal patch size is supported"
        );

        let (first_frame, second_frame) = split_frames(weight)?;

        let cfg = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };

        Ok(Self {
            patch_size,
            temporal_patch_size,
            in_channels,
            embed_dim,
            first_frame_conv: Conv2d::new(first_frame.contiguous()?, None, cfg),
            second_frame_conv: Conv2d::new(second_frame.contiguous()?, None, cfg),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let target_dtype = self.first_frame_conv.weight().dtype();
        let hidden_states = hidden_states
            .reshape((
                (),
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ))?
            .to_dtype(target_dtype)?;
        // Index on temporal dimension
        let (first_frame, second_frame) = split_frames(&hidden_states)?;

        // Reduce with both convs
        let first_frame = self
            .first_frame_conv
            .forward(&first_frame.contiguous()?)?
            .reshape(((), self.embed_dim))?;
        let second_frame = self
            .second_frame_conv
            .forward(&second_frame.contiguous()?)?
            .reshape(((), self.embed_dim))?;
        // Concatenate on temporal dimension
        let combined = (first_frame + second_frame)?.unsqueeze(2)?;

        // Reshape to (batch_size, embed_dim)
        combined.reshape(((), self.embed_dim))
    }
}

fn split_frames(x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
    let first_frame = x.i((.., .., 0, .., ..))?;
    let second_frame = x.i((.., .., 1, .., ..))?;
    Ok((first_frame, second_frame))
}

#[test]
fn test_vision_patch_embed() {
    use candle_core::{DType, Device};

    let embed_dim = 4;
    let in_channels = 3;
    let temporal_patch_size = 2;
    let patch_size = 2;

    let dims = [
        embed_dim,
        in_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    ];
    let weights = [
        [
            [
                [[-2.5095e-02, 4.8800e-03], [7.8459e-03, 2.8647e-04]],
                [[6.4076e-03, 5.8325e-03], [1.0669e-02, -4.5015e-03]],
            ],
            [
                [[-1.8527e-03, 7.5276e-03], [4.0476e-03, 1.7847e-03]],
                [[2.6491e-03, 1.2732e-02], [-1.3109e-05, -3.0360e-03]],
            ],
            [
                [[-1.4570e-02, -1.0234e-03], [-5.9915e-03, 4.7706e-03]],
                [[7.2618e-03, 9.1152e-04], [-3.8907e-03, 5.2792e-03]],
            ],
        ],
        [
            [
                [[-1.2685e-04, 2.4084e-03], [1.3254e-03, 7.6424e-03]],
                [[1.0950e-02, 3.3989e-03], [7.1997e-03, 4.1141e-03]],
            ],
            [
                [[1.9312e-02, 1.0119e-02], [-1.4364e-02, -1.1299e-02]],
                [[-1.3603e-03, 1.6354e-02], [6.5474e-03, 5.7600e-03]],
            ],
            [
                [[1.1415e-02, 1.8565e-04], [-1.8058e-02, 9.2543e-03]],
                [[-3.7534e-03, 1.0331e-02], [-6.8665e-03, 6.3681e-03]],
            ],
        ],
        [
            [
                [[-9.7267e-03, 9.5846e-03], [1.6192e-02, 1.4506e-02]],
                [[2.6948e-03, -2.1038e-03], [-7.3280e-03, 1.0430e-03]],
            ],
            [
                [[3.4875e-03, 9.6759e-03], [-4.6569e-03, 1.6048e-02]],
                [[-2.4801e-02, -4.1754e-03], [-1.1955e-02, 8.1234e-03]],
            ],
            [
                [[-1.9006e-02, 2.2858e-03], [2.4859e-04, -3.4595e-03]],
                [[2.8683e-03, -7.3084e-03], [1.7482e-03, -1.0939e-02]],
            ],
        ],
        [
            [
                [[-1.6022e-02, 1.3529e-02], [1.2888e-02, 5.2295e-04]],
                [[-1.5469e-02, 7.5671e-03], [7.7552e-03, 2.0265e-02]],
            ],
            [
                [[3.5818e-04, 1.2059e-03], [-8.0566e-03, -2.0758e-03]],
                [[-9.3195e-03, -1.5910e-02], [-1.1360e-02, -5.2260e-03]],
            ],
            [
                [[-5.1877e-03, -1.5013e-02], [-1.9267e-02, 1.2785e-03]],
                [[1.0229e-02, -5.5580e-03], [7.0427e-03, 7.0988e-03]],
            ],
        ],
    ];

    let weight = Tensor::from_iter(
        weights.into_iter().flatten().flatten().flatten().flatten(),
        &Device::Cpu,
    )
    .unwrap()
    .reshape(&dims)
    .unwrap()
    .to_dtype(DType::F16)
    .unwrap();

    let patch_embed = Qwen2_5VisionPatchEmbed::from_weight(
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        &weight,
    )
    .unwrap();

    let input = [[
        [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        [[[8., 9.], [10., 11.]], [[12., 13.], [14., 15.]]],
        [[[16., 17.], [18., 19.]], [[20., 21.], [22., 23.]]],
    ]];
    let input = Tensor::from_iter(
        input.into_iter().flatten().flatten().flatten().flatten(),
        &Device::Cpu,
    )
    .unwrap()
    .reshape(&[1, in_channels, temporal_patch_size, patch_size, patch_size])
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();

    let output = patch_embed
        .forward(&input)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let output_vec = output.to_vec2::<f32>().unwrap();
    println!("Output: {:?}", output_vec);
    let expected_output = [[0.3058, 0.6866, -0.7391, -0.6952]];
    assert_2d_vec_eq(output_vec, expected_output, 1e-2);
}

#[cfg(test)]
pub(crate) fn assert_2d_vec_eq(
    a: impl IntoIterator<Item = impl IntoIterator<Item = f32>>,
    b: impl IntoIterator<Item = impl IntoIterator<Item = f32>>,
    tolerance: f32,
) {
    let a = a
        .into_iter()
        .map(|x| x.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let b = b
        .into_iter()
        .map(|x| x.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(a.len(), b.len());
    for (a_row, b_row) in a.iter().zip(b.iter()) {
        assert_eq!(a_row.len(), b_row.len());
        for (a_elem, b_elem) in a_row.iter().zip(b_row.iter()) {
            assert!(
                (a_elem - b_elem).abs() < tolerance,
                "Elements are not equal: {} vs {}",
                a_elem,
                b_elem
            );
        }
    }
}
