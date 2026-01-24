//! ConcreteTensor - the actual tensor storage implementation

use std::mem::MaybeUninit;

use aligned_vec::{ABox, AVec};
use fusor_types::Layout;
use pulp::Simd;

use crate::expr::{Expr, linear_to_indices};
use crate::{MAX_SIMD_LANES, ResolveTensor, ResolvedTensor, SimdElement, TensorBacking};

/// Helper to iterate over indices of a tensor with given shape
pub struct IndexIterator {
    shape: Box<[usize]>,
    indices: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    pub fn new(shape: &[usize]) -> Self {
        let done = shape.iter().any(|&s| s == 0);
        Self {
            shape: shape.into(),
            indices: vec![0; shape.len()],
            done,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.indices.clone();

        // Special case: 0D tensor (scalar) - yield once then done
        if self.shape.is_empty() {
            self.done = true;
            return Some(result);
        }

        // Increment indices (row-major order)
        for i in (0..self.shape.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.shape[i] {
                break;
            }
            self.indices[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(result)
    }
}

#[derive(Clone)]
pub struct ConcreteTensor<T: SimdElement, const R: usize> {
    layout: Layout,
    backing: ABox<[T]>,
}

impl<T, const R: usize> TensorBacking<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    type Elem = T;
}

impl<T, const R: usize> ResolveTensor<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    fn to_concrete(&self) -> ConcreteTensor<T, R> {
        self.clone()
    }
}

impl<T, const R: usize> ResolvedTensor<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn data(&self) -> &ABox<[Self::Elem]> {
        &self.backing
    }
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        &mut self.backing
    }
}

impl<T: SimdElement, const R: usize> Expr for ConcreteTensor<T, R> {
    type Elem = T;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> T {
        if self.layout.is_contiguous() {
            self.backing[idx]
        } else {
            // Convert linear index to logical indices for strided access
            let indices = linear_to_indices::<R>(idx, self.layout.shape());
            let phys_idx = self.layout.linear_index(&indices);
            self.backing[phys_idx]
        }
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> T::Simd<S> {
        if self.layout.is_contiguous() {
            // Fast path: direct SIMD load from contiguous, aligned data
            // - base_idx is always aligned to SIMD width (caller guarantees this)
            // - backing is allocated with 64-byte alignment via aligned-vec
            // - base_idx + lane_count <= len (caller guarantees this)
            let (simd_slice, _) = T::as_simd::<S>(&self.backing[base_idx..]);
            simd_slice[0]
        } else {
            // Slow path: gather elements one by one
            // For strided tensors, we fall back to scalar evaluation
            // and construct a SIMD vector manually
            let lane_count = std::mem::size_of::<T::Simd<S>>() / std::mem::size_of::<T>();
            let mut temp = [T::default(); MAX_SIMD_LANES];
            for i in 0..lane_count {
                temp[i] = self.eval_scalar(base_idx + i);
            }
            let (simd_vec, _) = T::as_simd::<S>(&temp[..lane_count]);
            simd_vec[0]
        }
    }

    fn len(&self) -> usize {
        self.layout.num_elements()
    }

    fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

// Implement ResolveTensor for references (TensorBacking is covered by blanket impl in lib.rs)
impl<T: SimdElement, const R: usize> ResolveTensor<R> for &ConcreteTensor<T, R> {
    fn to_concrete(&self) -> ConcreteTensor<T, R> {
        (*self).clone()
    }
}

impl<T: SimdElement, const R: usize> ConcreteTensor<T, R> {
    /// Create a new tensor with contiguous layout from shape, filled with zeros
    pub fn zeros(shape: [usize; R]) -> Self
    where
        T: Default,
    {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        let mut vec: AVec<T> = AVec::with_capacity(64, num_elements);
        vec.resize(num_elements, T::default());
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Create a new tensor with uninitialized memory (for internal use only)
    #[inline]
    pub(crate) fn uninit_unchecked(shape: [usize; R]) -> Self {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        // Transmute the MaybeUninit vec to T vec - this is safe because we will
        // write to all elements before reading, and MaybeUninit<T> has same layout as T
        // SAFETY: MaybeUninit<T> has same layout as T, and T is Pod so any memory layout is valid
        let backing: ABox<[T]> = unsafe {
            // Allocate the aligned pointer
            let mut vec: AVec<MaybeUninit<T>> = AVec::with_capacity(64, num_elements);
            vec.set_len(num_elements);
            std::mem::transmute::<ABox<[MaybeUninit<T>]>, ABox<[T]>>(vec.into_boxed_slice())
        };
        Self { layout, backing }
    }

    /// Create a new tensor from existing data with contiguous layout
    pub fn from_slice(shape: [usize; R], data: &[T]) -> Self {
        let layout = Layout::contiguous(&shape);
        assert_eq!(layout.num_elements(), data.len());
        let mut vec: AVec<T> = AVec::with_capacity(64, data.len());
        vec.extend_from_slice(data);
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Get a reference to the layout
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Calculate the linear index for given logical indices
    fn linear_index(&self, indices: &[usize; R]) -> usize {
        self.layout.linear_index(indices)
    }

    /// Get element at logical indices
    pub fn get(&self, indices: [usize; R]) -> T {
        let idx = self.linear_index(&indices);
        self.backing[idx]
    }

    /// Set element at logical indices
    pub fn set(&mut self, indices: [usize; R], value: T) {
        let idx = self.linear_index(&indices);
        self.backing[idx] = value;
    }

    /// Create a new tensor from existing layout and backing data
    pub(crate) fn from_parts(layout: Layout, backing: ABox<[T]>) -> Self {
        Self { layout, backing }
    }

    /// Get the backing data
    pub(crate) fn backing(&self) -> &ABox<[T]> {
        &self.backing
    }

    /// Get the backing data mutably
    pub(crate) fn backing_mut(&mut self) -> &mut ABox<[T]> {
        &mut self.backing
    }

    /// Consume the tensor and return the backing data.
    ///
    /// This is used by layout operations to move ownership of the backing data
    /// into a `MapLayout` without cloning.
    pub(crate) fn into_backing(self) -> ABox<[T]> {
        self.backing
    }
}
