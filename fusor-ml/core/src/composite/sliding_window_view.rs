use crate::{DataType, LargerRank, LargerRankInner, Tensor, map_layout::MapLayoutOperation};

/// Configuration for sliding window
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindow {
    axis: usize,
    window_size: usize,
    step: usize,
}

impl SlidingWindow {
    /// Create a new SlidingWindow configuration
    pub fn new(axis: usize, window_size: usize, step: usize) -> Self {
        Self {
            axis,
            window_size,
            step,
        }
    }
}

impl From<(usize, usize)> for SlidingWindow {
    fn from(val: (usize, usize)) -> Self {
        SlidingWindow {
            axis: val.0,
            window_size: val.1,
            step: 1,
        }
    }
}

impl From<[usize; 2]> for SlidingWindow {
    fn from(val: [usize; 2]) -> Self {
        SlidingWindow {
            axis: val[0],
            window_size: val[1],
            step: 1,
        }
    }
}

impl From<(usize, usize, usize)> for SlidingWindow {
    fn from(val: (usize, usize, usize)) -> Self {
        SlidingWindow {
            axis: val.0,
            window_size: val.1,
            step: val.2,
        }
    }
}

impl From<[usize; 3]> for SlidingWindow {
    fn from(val: [usize; 3]) -> Self {
        SlidingWindow {
            axis: val[0],
            window_size: val[1],
            step: val[2],
        }
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Create a sliding window view of a tensor using strided views (zero-copy)
    ///
    /// # Arguments
    /// - `windows`: Array of SlidingWindow configurations specifying the axis, window size, and step
    /// # Returns
    /// - A new tensor with increased rank, with new dimensions appended for each sliding window
    /// # Panics
    /// - If any of the specified axes are out of bounds or not unique
    pub fn sliding_window_view<const DIFF: usize, const R2: usize>(
        &self,
        windows: [impl Into<SlidingWindow>; DIFF],
    ) -> <Self as LargerRankInner<DIFF>>::LargerRank
    where
        Self: LargerRank<DIFF, R2, D>,
    {
        let shape = *self.shape();
        let mut windows: [SlidingWindow; DIFF] = windows.map(|w| w.into());
        windows.sort_by_key(|w| w.axis);
        #[cfg(debug_assertions)]
        {
            windows.iter().for_each(|w| {
                assert!(w.axis < R, "Sliding window axis out of bounds");
            });
            windows.windows(2).for_each(|w_pair| {
                assert!(
                    w_pair[0].axis != w_pair[1].axis,
                    "Sliding window axes must be unique"
                );
            });
        }

        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            // Transform shape: insert num_windows and window_size dimensions
            move |shape| {
                Box::new(std::array::from_fn::<_, R2, _>(|i| {
                    if i < shape.len() {
                        // Original dimension
                        if let Ok(idx) = windows.binary_search_by_key(&i, |w| w.axis) {
                            // This dimension is being windowed
                            let dim_size = shape[i];
                            let window = &windows[idx];
                            (dim_size - window.window_size) / window.step + 1
                        } else {
                            // Not windowed
                            shape[i]
                        }
                    } else {
                        // New dimensions for windows
                        let index = i - shape.len();
                        let window = &windows[index];
                        window.window_size
                    }
                }))
            },
            // Transform strides: insert strides for the new dimensions
            move |offset, strides| {
                (
                    offset,
                    Box::new(std::array::from_fn::<_, R2, _>(|i| {
                        if i < strides.len() {
                            // Original dimension
                            if let Ok(idx) = windows.binary_search_by_key(&i, |w| w.axis) {
                                // This dimension is being windowed
                                let window = &windows[idx];
                                strides[i] * window.step
                            } else {
                                // Not windowed - keep original stride
                                strides[i]
                            }
                        } else {
                            // New dimensions for windows
                            let index = i - shape.len();
                            let window = &windows[index];
                            strides[window.axis]
                        }
                    })),
                )
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_sliding_window_view_1d() {
        let device = Device::test_instance();

        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let input = Tensor::new(&device, &input_data);

        let windows = input.sliding_window_view([SlidingWindow::new(0, 3, 2)]);

        // Expected num_windows = (7 - 3) / 2 + 1 = 3
        // Expected shape: (3, 3)
        assert_eq!(windows.shape(), &[3, 3]);

        // Verify the content
        let result = windows.as_slice().await.unwrap();

        // First window: [1, 2, 3]
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[0, 2]], 3.0);

        // Second window: [3, 4, 5]
        assert_eq!(result[[1, 0]], 3.0);
        assert_eq!(result[[1, 1]], 4.0);
        assert_eq!(result[[1, 2]], 5.0);

        // Third window: [5, 6, 7]
        assert_eq!(result[[2, 0]], 5.0);
        assert_eq!(result[[2, 1]], 6.0);
        assert_eq!(result[[2, 2]], 7.0);
    }

    #[tokio::test]
    async fn test_sliding_window_view_2d() {
        let device = Device::test_instance();

        let input_data = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ];
        let input = Tensor::new(&device, &input_data);

        let windows =
            input.sliding_window_view([SlidingWindow::new(0, 3, 3), SlidingWindow::new(1, 3, 3)]);

        let result = windows.as_slice().await.unwrap();
        println!("{result:?}");

        assert_eq!(windows.shape(), &[2, 2, 3, 3]);
        let expected = [
            [
                [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [13.0, 14.0, 15.0]],
                [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0], [16.0, 17.0, 18.0]],
            ],
            [
                [[19.0, 20.0, 21.0], [25.0, 26.0, 27.0], [31.0, 32.0, 33.0]],
                [[22.0, 23.0, 24.0], [28.0, 29.0, 30.0], [34.0, 35.0, 36.0]],
            ],
        ];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_eq!(result[[i, j, k, l]], expected[i][j][k][l]);
                    }
                }
            }
        }
    }
}
