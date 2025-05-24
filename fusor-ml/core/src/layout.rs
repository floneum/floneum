use std::ops::Range;

pub(crate) const TILE_SIZE: u32 = 8;

fn continuous_strides(shape: &[usize]) -> Box<[usize]> {
    let mut acc = 1;
    let mut strides = vec![0; shape.len()].into_boxed_slice();
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

pub(crate) fn slice_strides(
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

pub(crate) fn slice_shape(slices: &[Range<usize>], _shape: &[usize]) -> Box<[usize]> {
    slices.iter().map(|range| range.len()).collect()
}

#[derive(Clone, Debug)]
pub struct Layout {
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Layout {
    pub fn contiguous(shape: &[usize]) -> Self {
        let strides = continuous_strides(shape);
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
        self.offset == 0 && self.strides == continuous_strides(&self.shape)
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
    pub(crate) fn allocation_overlaps(&self) -> bool {
        self.strides.contains(&0)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[test]
fn test_contiguous() {
    let layout = Layout::contiguous(&[2, 3]);
    assert!(layout.is_contiguous());
    assert!(!layout.slice(&[0..1, 0..1]).is_contiguous());
    assert!(!layout.slice(&[1..2, 0..3]).is_contiguous());
}
