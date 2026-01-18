use std::ops::Range;

pub const TILE_SIZE: u32 = 1;

pub fn slice_strides(
    slices: &[Range<usize>],
    offset: usize,
    strides: &[usize],
) -> (usize, Box<[usize]>) {
    let start_offset = slices
        .iter()
        .zip(strides.iter())
        .map(|(range, stride)| *stride * range.start)
        .sum::<usize>();

    (offset + start_offset, strides.into())
}

pub fn slice_shape(slices: &[Range<usize>], _shape: &[usize]) -> Box<[usize]> {
    slices.iter().map(|range| range.len()).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Layout {
    pub fn contiguous(shape: &[usize]) -> Self {
        let strides = Self::continuous_strides(shape);
        Self {
            offset: 0,
            shape: shape.into(),
            strides,
        }
    }

    pub fn from_parts(offset: usize, shape: Box<[usize]>, strides: Box<[usize]>) -> Self {
        Self {
            offset,
            shape,
            strides,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == Self::continuous_strides(&self.shape)
    }

    pub fn slice(&self, slices: &[Range<usize>]) -> Self {
        let (offset, strides) = slice_strides(slices, self.offset, &self.strides);
        let shape = slice_shape(slices, &self.strides);

        Self {
            offset,
            shape,
            strides,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Check if any items in this tensor point to the same allocation. This will be true if
    /// the tensor has a stride of 0.
    pub fn allocation_overlaps(&self) -> bool {
        self.strides.contains(&0)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Calculate the linear index for a given set of logical indices
    pub fn linear_index(&self, indices: &[usize]) -> usize {
        self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>()
    }

    /// Get the total number of elements in the tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn continuous_strides(shape: &[usize]) -> Box<[usize]> {
        let mut acc = 1;
        let mut strides = vec![0; shape.len()].into_boxed_slice();
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    /// Permute the layout dimensions according to the given axes order.
    ///
    /// # Arguments
    /// * `axes` - A permutation specifying the new order of dimensions
    ///
    /// # Panics
    /// Panics if axes don't form a valid permutation
    pub fn permute(&self, axes: &[usize]) -> Self {
        let rank = self.shape.len();
        assert_eq!(axes.len(), rank, "Axes length must match rank");

        // Validate axes are a valid permutation
        let mut seen = vec![false; rank];
        for &axis in axes {
            assert!(axis < rank, "Axis {} out of range for rank {}", axis, rank);
            assert!(!seen[axis], "Duplicate axis {} in permutation", axis);
            seen[axis] = true;
        }

        let mut new_shape = vec![0usize; rank];
        let mut new_strides = vec![0usize; rank];
        for (new_idx, &old_idx) in axes.iter().enumerate() {
            new_shape[new_idx] = self.shape[old_idx];
            new_strides[new_idx] = self.strides[old_idx];
        }

        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
        }
    }

    /// Transpose two dimensions of the layout.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// # Panics
    /// Panics if either dimension is out of range
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let rank = self.shape.len();
        assert!(dim0 < rank, "dim0 {} out of range for rank {}", dim0, rank);
        assert!(dim1 < rank, "dim1 {} out of range for rank {}", dim1, rank);

        let mut new_shape: Vec<usize> = self.shape.to_vec();
        let mut new_strides: Vec<usize> = self.strides.to_vec();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
        }
    }

    /// Create a broadcast layout to match the target shape.
    ///
    /// Broadcasting rules (flexible, iterating from right to left):
    /// - Source dimensions are consumed only when they match the target or are size 1
    /// - Matching dimensions keep their stride
    /// - Size-1 dimensions get stride 0 (broadcast)
    /// - Unmatched target dimensions get stride 0 (new dimensions)
    /// - New dimensions can be added anywhere, not just on the left
    ///
    /// # Panics
    /// Panics if not all source dimensions can be matched to target dimensions
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Self {
        let new_rank = target_shape.len();
        assert!(
            new_rank >= self.shape.len(),
            "Cannot broadcast to smaller rank"
        );

        let mut new_strides = vec![0usize; new_rank];
        let mut shape_iter = self.shape.iter().rev().peekable();
        let mut strides_iter = self.strides.iter().rev();

        for (new_idx, &target_dim) in target_shape.iter().enumerate().rev() {
            let stride = if let Some(&src_dim) = shape_iter.next_if(|&&src_dim| {
                src_dim == target_dim || (src_dim == 1 && target_dim > 1)
            }) {
                let stride = *strides_iter.next().unwrap();
                // Matching dim, use the same stride
                if src_dim == target_dim {
                    stride
                }
                // Broadcasted dim, set stride to 0
                else {
                    0
                }
            } else {
                // New dimension or non-matching - stride 0
                0
            };
            new_strides[new_idx] = stride;
        }

        assert_eq!(
            shape_iter.len(),
            0,
            "Failed to broadcast: not all source dimensions were matched. \
            Source shape {:?} is not compatible with target shape {:?}",
            self.shape,
            target_shape
        );

        Self {
            offset: self.offset,
            shape: target_shape.into(),
            strides: new_strides.into_boxed_slice(),
        }
    }

    /// Squeeze (remove) a dimension of size 1.
    ///
    /// # Arguments
    /// * `dim` - The dimension to squeeze (must have size 1)
    ///
    /// # Panics
    /// Panics if the dimension doesn't have size 1 or is out of range
    pub fn squeeze(&self, dim: usize) -> Self {
        let rank = self.shape.len();
        assert!(rank > 0, "Cannot squeeze a scalar layout");
        assert!(dim < rank, "Dimension {} out of range for rank {}", dim, rank);
        assert!(
            self.shape[dim] == 1,
            "Cannot squeeze dimension {} of size {} (must be 1)",
            dim,
            self.shape[dim]
        );

        let mut new_shape = Vec::with_capacity(rank - 1);
        let mut new_strides = Vec::with_capacity(rank - 1);

        for i in 0..rank {
            if i != dim {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }
        }

        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
        }
    }

    /// Unsqueeze (add) a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    /// * `dim` - Where to insert the new dimension
    ///
    /// # Panics
    /// Panics if dim is out of range
    pub fn unsqueeze(&self, dim: usize) -> Self {
        let rank = self.shape.len();
        assert!(dim <= rank, "Dimension {} out of range for inserting into rank {}", dim, rank);

        let new_rank = rank + 1;
        let mut new_shape = Vec::with_capacity(new_rank);
        let mut new_strides = Vec::with_capacity(new_rank);

        for i in 0..new_rank {
            if i == dim {
                new_shape.push(1);
                // Stride for size-1 dim doesn't matter, but using 1 is conventional
                new_strides.push(1);
            } else {
                let old_idx = if i < dim { i } else { i - 1 };
                new_shape.push(self.shape[old_idx]);
                new_strides.push(self.strides[old_idx]);
            }
        }

        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
        }
    }

    /// Reshape the layout to a new shape.
    ///
    /// This only works on contiguous layouts. For non-contiguous layouts,
    /// the caller must first make the data contiguous.
    ///
    /// # Arguments
    /// * `new_shape` - The target shape (must have same total element count)
    ///
    /// # Panics
    /// Panics if the layout is not contiguous or element count doesn't match
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let old_elements: usize = self.shape.iter().product();
        let new_elements: usize = new_shape.iter().product();
        assert_eq!(
            old_elements, new_elements,
            "Cannot reshape: element count mismatch ({} vs {})",
            old_elements, new_elements
        );
        assert!(
            self.is_contiguous(),
            "Cannot reshape non-contiguous layout; make it contiguous first"
        );

        Self::contiguous(new_shape)
    }

    /// Narrow the layout along a given dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow
    /// * `start` - The starting index
    /// * `length` - The length of the slice
    ///
    /// # Panics
    /// Panics if the slice is out of bounds
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Self {
        let rank = self.shape.len();
        assert!(dim < rank, "Dimension {} out of range for rank {}", dim, rank);
        assert!(
            start + length <= self.shape[dim],
            "Narrow out of bounds: {}..{} for dimension of size {}",
            start,
            start + length,
            self.shape[dim]
        );

        // Build ranges for all dimensions
        let slices: Vec<Range<usize>> = (0..rank)
            .map(|i| {
                if i == dim {
                    start..start + length
                } else {
                    0..self.shape[i]
                }
            })
            .collect();

        self.slice(&slices)
    }
}

#[test]
fn test_contiguous() {
    let layout = Layout::contiguous(&[2, 3]);
    assert!(layout.is_contiguous());
    assert!(!layout.slice(&[0..1, 0..1]).is_contiguous());
    assert!(!layout.slice(&[1..2, 0..3]).is_contiguous());
}
