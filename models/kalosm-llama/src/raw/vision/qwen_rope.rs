use candle_core::{Device, Result, Tensor, D};

use crate::raw::rope::create_inverse_frequency;

pub struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    pub(crate) fn new(dim: usize, rope_theta: f32, device: &Device) -> Result<Self> {
        Ok(Self {
            inv_freq: create_inverse_frequency(
                None,
                None,
                candle_core::DType::F32,
                dim,
                rope_theta,
                device,
            )?,
        })
    }

    pub(crate) fn make_embeds(&self, sequence_length: u32) -> Result<Tensor> {
        let seq = Tensor::arange(0f32, sequence_length as f32, self.inv_freq.device())?
            .unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ImageSize {
    width: u32,
    height: u32,
}

impl ImageSize {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    fn patch_size(&self, merge_size: usize) -> Self {
        let width = (self.width + merge_size as u32 - 1) / merge_size as u32;
        let height = (self.height + merge_size as u32 - 1) / merge_size as u32;
        Self { width, height }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct VideoSize {
    image_size: ImageSize,
    frames: u32,
}

impl VideoSize {
    fn new(width: u32, height: u32, frames: u32) -> Self {
        Self {
            image_size: ImageSize { width, height },
            frames,
        }
    }

    fn patch_size(&self, merge_size: usize) -> Self {
        let image_size = self.image_size.patch_size(merge_size);
        Self {
            image_size,
            frames: self.frames,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RopeIndex {
    x: u32,
    y: u32,
    time: u32,
}

// Calculate the rope index for each token in the input_ids.
// Images are encoded as vision_start_token_id then a sequence of image_token_id.
// Videos are encoded as vision_start_token_id then a sequence of video_token_id.
//
// Rope indexes are calculated with an monotonic increasing time index.
// The x and y indexes are calculated based on the patch index of the image or video.
//
// Eg. The series [vision_start_token_id, image_token_id, image_token_id, image_token_id, image_token_id]
//     will be encoded as:
//     [RopeIndex { x: 0, y: 0, time: 0 }, RopeIndex { x: 1, y: 1, time: 1 }, RopeIndex { x: 1, y: 2, time: 1 },
//      RopeIndex { x: 2, y: 1, time: 1 }, RopeIndex { x: 2, y: 2, time: 1 }]
fn get_rope_index(
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    input_ids: &[u32],
    image_sizes: &[ImageSize],
    video_sizes: &[VideoSize],
) -> Vec<RopeIndex> {
    let mut images = image_sizes.iter();
    let mut videos = video_sizes.iter();
    let mut indexes = Vec::new();
    let mut max_time_index = 0;
    let mut index = 0;

    while let Some(token) = input_ids.get(index).copied() {
        push_text_token(&mut indexes, &mut max_time_index, &mut index);
        if token == vision_start_token_id {
            let Some(token) = input_ids.get(index).copied() else {
                break;
            };
            if token == image_token_id {
                let image = images.next().unwrap();
                let image_size = image.patch_size(spatial_merge_size);
                let image_width = image_size.width;
                let image_height = image_size.height;
                for j in 0..image_width {
                    for k in 0..image_height {
                        indexes.push(RopeIndex {
                            x: max_time_index + j,
                            y: max_time_index + k,
                            time: max_time_index,
                        });
                    }
                }
                max_time_index += 2;
                index = index + image_width as usize * image_height as usize;
            } else if token == video_token_id {
                let video = videos.next().unwrap();
                let video_size = video.patch_size(spatial_merge_size);
                let image_width = video_size.image_size.width;
                let image_height = video_size.image_size.height;
                for frame in 0..video_size.frames {
                    for j in 0..image_width {
                        for k in 0..image_height {
                            indexes.push(RopeIndex {
                                x: max_time_index + j,
                                y: max_time_index + k,
                                time: max_time_index + frame,
                            });
                        }
                    }
                }
                index =
                    index + image_width as usize * image_height as usize * video.frames as usize;
                max_time_index = max_time_index.max(max_time_index + video.frames + 1);
            }
        }
    }
    debug_assert_eq!(
        indexes.len(),
        input_ids.len(),
        "The length of indexes should match the length of input_ids"
    );

    indexes
}

fn push_text_token(indexes: &mut Vec<RopeIndex>, max_time_index: &mut u32, index: &mut usize) {
    // Push the token sequence with increasing time, x, and y positions
    indexes.push(RopeIndex {
        x: *max_time_index,
        y: *max_time_index,
        time: *max_time_index,
    });
    *max_time_index += 1;
    *index += 1;
}

#[test]
fn test_get_rope_index_text() {
    let spatial_merge_size = 16;
    let image_token_id = 100;
    let video_token_id = 200;
    let vision_start_token_id = 10;
    let input_ids = vec![0, 1, 2, 3, 4, 5];
    let image_sizes = vec![ImageSize::new(32, 32), ImageSize::new(64, 64)];
    let video_sizes = vec![VideoSize::new(32, 32, 10), VideoSize::new(64, 64, 20)];

    let indexes = get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        &input_ids,
        &image_sizes,
        &video_sizes,
    );

    assert_eq!(
        indexes,
        vec![
            RopeIndex {
                x: 0,
                y: 0,
                time: 0
            },
            RopeIndex {
                x: 1,
                y: 1,
                time: 1
            },
            RopeIndex {
                x: 2,
                y: 2,
                time: 2
            },
            RopeIndex {
                x: 3,
                y: 3,
                time: 3
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 4
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 5
            },
        ]
    );
}

#[test]
fn test_get_rope_index() {
    let spatial_merge_size = 16;
    let image_token_id = 1;
    let video_token_id = 2;
    let vision_start_token_id = 0;
    let input_ids = vec![
        0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5,
    ];
    let image_sizes = vec![ImageSize::new(32, 32)];
    let video_sizes = vec![VideoSize::new(32, 32, 5)];

    let indexes = get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        &input_ids,
        &image_sizes,
        &video_sizes,
    );

    assert_eq!(
        indexes,
        [
            // Start image token
            RopeIndex {
                x: 0,
                y: 0,
                time: 0
            },
            // 4 image tokens for a 2x2 patch
            RopeIndex {
                x: 1,
                y: 1,
                time: 1
            },
            RopeIndex {
                x: 1,
                y: 2,
                time: 1
            },
            RopeIndex {
                x: 2,
                y: 1,
                time: 1
            },
            RopeIndex {
                x: 2,
                y: 2,
                time: 1
            },
            // Start video token
            RopeIndex {
                x: 3,
                y: 3,
                time: 3
            },
            // 20 video tokens for a 2x2x5 patch
            RopeIndex {
                x: 4,
                y: 4,
                time: 4
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 4
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 4
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 4
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 5
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 5
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 5
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 5
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 6
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 6
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 6
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 6
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 7
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 7
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 7
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 7
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 8
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 8
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 8
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 8
            },
            // Remaining text tokens
            RopeIndex {
                x: 10,
                y: 10,
                time: 10
            },
            RopeIndex {
                x: 11,
                y: 11,
                time: 11
            },
            RopeIndex {
                x: 12,
                y: 12,
                time: 12
            }
        ]
    );
}
