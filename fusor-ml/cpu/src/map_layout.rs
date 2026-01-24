//! MapLayout - A lazy tensor wrapper that holds backing data with a transformed layout
//!
//! This module provides `MapLayout`, which enables O(1) layout operations (slice, permute,
//! transpose, broadcast, etc.) by storing the backing data with a new layout instead of
//! cloning the data.

use aligned_vec::ABox;
use fusor_types::Layout;
use pulp::Simd;

use crate::{ConcreteTensor, ResolvedTensor, SimdElement, TensorBacking};

/// A tensor that holds backing data with a transformed layout.
///
/// `MapLayout` is created by layout-changing operations like `slice`, `permute`,
/// `transpose`, and `broadcast_as`. It owns the backing data (moved from the
/// original tensor) and stores a new layout that describes how to interpret that data.
///
/// This enables O(1) layout operations instead of O(n) cloning.
pub struct MapLayout<E: SimdElement, const R: usize> {
    tensor: ConcreteTensor<E, R>,
}

impl<E: SimdElement, const R: usize> Clone for MapLayout<E, R> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
        }
    }
}

impl<E: SimdElement, const R: usize> MapLayout<E, R> {
    /// Create a new MapLayout from backing data and a layout.
    pub(crate) fn new(backing: ABox<[E]>, layout: Layout) -> Self {
        Self { tensor: ConcreteTensor::from_parts(layout, backing) }
    }

    /// Get element at logical indices.
    pub fn get(&self, indices: [usize; R]) -> E {
        let idx = self.tensor.layout().linear_index(&indices);
        self.tensor.backing()[idx]
    }
}

impl<E: SimdElement, const R: usize> TensorBacking<R> for MapLayout<E, R> {
    type Elem = E;

    fn layout(&self) -> Layout {
        self.tensor.layout().clone()
    }

    fn to_concrete(&self) -> ConcreteTensor<E, R> {
        self.tensor.clone()
    }

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> E {
        self.tensor.eval_scalar(idx)
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> E::Simd<S> {
        self.tensor.eval_simd(_simd, base_idx)
    }
}

impl<E: SimdElement, const R: usize> ResolvedTensor<R> for MapLayout<E, R> {
    fn data(&self) -> &ABox<[Self::Elem]> {
        self.tensor.data()
    }

    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        self.tensor.data_mut()
    }
}
