//! MapLayout - A lazy tensor wrapper that holds backing data with a transformed layout
//!
//! This module provides `MapLayout`, which enables O(1) layout operations (slice, permute,
//! transpose, broadcast, etc.) by storing the backing data with a new layout instead of
//! cloning the data.

use aligned_vec::ABox;
use fusor_types::Layout;
use pulp::Simd;

use crate::expr::{linear_to_indices, materialize_expr, Expr};
use crate::{ConcreteTensor, ResolveTensor, ResolvedTensor, SimdElement, TensorBacking, MAX_SIMD_LANES};

/// A tensor that holds backing data with a transformed layout.
///
/// `MapLayout` is created by layout-changing operations like `slice`, `permute`,
/// `transpose`, and `broadcast_as`. It owns the backing data (moved from the
/// original tensor) and stores a new layout that describes how to interpret that data.
///
/// This enables O(1) layout operations instead of O(n) cloning.
pub struct MapLayout<E: SimdElement, const R: usize> {
    backing: ABox<[E]>,
    layout: Layout,
}

impl<E: SimdElement, const R: usize> Clone for MapLayout<E, R> {
    fn clone(&self) -> Self {
        Self {
            backing: self.backing.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<E: SimdElement, const R: usize> MapLayout<E, R> {
    /// Create a new MapLayout from backing data and a layout.
    pub(crate) fn new(backing: ABox<[E]>, layout: Layout) -> Self {
        Self { backing, layout }
    }

    /// Get a reference to the layout.
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Get a reference to the backing data.
    pub(crate) fn backing(&self) -> &ABox<[E]> {
        &self.backing
    }

    /// Get a mutable reference to the backing data.
    pub(crate) fn backing_mut(&mut self) -> &mut ABox<[E]> {
        &mut self.backing
    }

    /// Get element at logical indices.
    pub fn get(&self, indices: [usize; R]) -> E {
        let idx = self.layout.linear_index(&indices);
        self.backing[idx]
    }
}

impl<E: SimdElement, const R: usize> TensorBacking<R> for MapLayout<E, R> {
    type Elem = E;
}

impl<E: SimdElement, const R: usize> Expr for MapLayout<E, R> {
    type Elem = E;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> E {
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
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> E::Simd<S> {
        if self.layout.is_contiguous() {
            // Fast path: direct SIMD load from contiguous, aligned data
            let (simd_slice, _) = E::as_simd::<S>(&self.backing[base_idx..]);
            simd_slice[0]
        } else {
            // Slow path: gather elements one by one
            let lane_count = std::mem::size_of::<E::Simd<S>>() / std::mem::size_of::<E>();
            let mut temp = [E::default(); MAX_SIMD_LANES];
            for i in 0..lane_count {
                temp[i] = self.eval_scalar(base_idx + i);
            }
            let (simd_vec, _) = E::as_simd::<S>(&temp[..lane_count]);
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

impl<E: SimdElement, const R: usize> ResolveTensor<R> for MapLayout<E, R> {
    fn to_concrete(&self) -> ConcreteTensor<E, R> {
        let shape: [usize; R] = self
            .layout
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        materialize_expr(self, shape)
    }
}

// Implement ResolveTensor for references
impl<E: SimdElement, const R: usize> ResolveTensor<R> for &MapLayout<E, R> {
    fn to_concrete(&self) -> ConcreteTensor<E, R> {
        (*self).to_concrete()
    }
}

impl<E: SimdElement, const R: usize> ResolvedTensor<R> for MapLayout<E, R> {
    fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    fn offset(&self) -> usize {
        self.layout.offset()
    }

    fn data(&self) -> &ABox<[Self::Elem]> {
        &self.backing
    }

    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        &mut self.backing
    }
}
