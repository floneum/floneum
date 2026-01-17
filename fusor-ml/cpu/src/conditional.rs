//! Conditional tensor operations: where_cond
//! Selects elements based on condition tensor != 0

use pulp::{Arch, Simd, WithSimd};

use crate::{
    ConcreteTensor, IndexIterator, ResolvedTensor, SimdElement,
};

/// Helper trait for types that can be compared to zero
pub trait IsNonZero: SimdElement {
    fn is_nonzero(&self) -> bool;
}

impl IsNonZero for f32 {
    fn is_nonzero(&self) -> bool { *self != 0.0 }
}

impl IsNonZero for f64 {
    fn is_nonzero(&self) -> bool { *self != 0.0 }
}

impl IsNonZero for i8 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for i16 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for i32 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for i64 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for u8 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for u16 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for u32 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

impl IsNonZero for u64 {
    fn is_nonzero(&self) -> bool { *self != 0 }
}

/// Helper struct for dispatching where_cond operations via Arch::dispatch
struct WhereCondDispatch<'a, E: SimdElement + IsNonZero> {
    cond: &'a [E],
    on_true: &'a [E],
    on_false: &'a [E],
    out: &'a mut [E],
}

impl<E: SimdElement + IsNonZero> WithSimd for WhereCondDispatch<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
        // Process SIMD vectors by extracting lanes (no native SIMD select for arbitrary masks)
        let lane_count = std::mem::size_of::<E::Simd<S>>() / std::mem::size_of::<E>();

        let (cond_simd, cond_tail) = E::as_simd::<S>(self.cond);
        let (on_true_simd, on_true_tail) = E::as_simd::<S>(self.on_true);
        let (on_false_simd, on_false_tail) = E::as_simd::<S>(self.on_false);
        let (out_simd, out_tail) = E::as_mut_simd::<S>(self.out);

        // Process SIMD chunks
        for (((cond, on_true), on_false), out) in cond_simd
            .iter()
            .zip(on_true_simd.iter())
            .zip(on_false_simd.iter())
            .zip(out_simd.iter_mut())
        {
            let mut temp_cond = [E::default(); crate::MAX_SIMD_LANES];
            let mut temp_true = [E::default(); crate::MAX_SIMD_LANES];
            let mut temp_false = [E::default(); crate::MAX_SIMD_LANES];

            unsafe {
                std::ptr::copy_nonoverlapping(
                    cond as *const _ as *const E,
                    temp_cond.as_mut_ptr(),
                    lane_count,
                );
                std::ptr::copy_nonoverlapping(
                    on_true as *const _ as *const E,
                    temp_true.as_mut_ptr(),
                    lane_count,
                );
                std::ptr::copy_nonoverlapping(
                    on_false as *const _ as *const E,
                    temp_false.as_mut_ptr(),
                    lane_count,
                );
            }

            for i in 0..lane_count {
                temp_cond[i] = if temp_cond[i].is_nonzero() {
                    temp_true[i]
                } else {
                    temp_false[i]
                };
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    temp_cond.as_ptr(),
                    out as *mut _ as *mut E,
                    lane_count,
                );
            }
        }

        // Process tail elements
        for (((c, t), f), o) in cond_tail
            .iter()
            .zip(on_true_tail.iter())
            .zip(on_false_tail.iter())
            .zip(out_tail.iter_mut())
        {
            *o = if c.is_nonzero() { *t } else { *f };
        }
    }
}

/// Perform where_cond operation on contiguous slices
#[inline(always)]
fn where_cond_contiguous<E: SimdElement + IsNonZero>(
    cond: &[E],
    on_true: &[E],
    on_false: &[E],
    out: &mut [E],
) {
    Arch::new().dispatch(WhereCondDispatch::<E> {
        cond,
        on_true,
        on_false,
        out,
    });
}

/// Conditional selection: where condition != 0, select on_true, else on_false
#[inline(always)]
pub(crate) fn where_cond_ref<E, const RANK: usize>(
    cond: &ConcreteTensor<E, RANK>,
    on_true: &ConcreteTensor<E, RANK>,
    on_false: &ConcreteTensor<E, RANK>,
) -> ConcreteTensor<E, RANK>
where
    E: SimdElement + IsNonZero,
{
    let shape: [usize; RANK] = ResolvedTensor::shape(cond)
        .try_into()
        .expect("Shape length mismatch");
    let mut output = ConcreteTensor::<E, RANK>::uninit_unchecked(shape);

    let all_contiguous = cond.layout().is_contiguous()
        && on_true.layout().is_contiguous()
        && on_false.layout().is_contiguous();

    if all_contiguous {
        where_cond_contiguous(cond.data(), on_true.data(), on_false.data(), output.data_mut());
    } else {
        let tensor_shape = ResolvedTensor::shape(cond);
        for indices in IndexIterator::new(tensor_shape) {
            let cond_idx = cond.layout().linear_index(&indices);
            let true_idx = on_true.layout().linear_index(&indices);
            let false_idx = on_false.layout().linear_index(&indices);
            let out_idx = output.layout().linear_index(&indices);

            let cond_val = cond.data()[cond_idx];
            output.data_mut()[out_idx] = if cond_val.is_nonzero() {
                on_true.data()[true_idx]
            } else {
                on_false.data()[false_idx]
            };
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_where_cond_f32() {
        let cond = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 0.0, 1.0, 0.0]);
        let on_true = ConcreteTensor::<f32, 1>::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let on_false = ConcreteTensor::<f32, 1>::from_slice([4], &[100.0, 200.0, 300.0, 400.0]);

        let result = where_cond_ref(&cond, &on_true, &on_false);

        assert_eq!(result.get([0]), 10.0);   // cond=1, select on_true
        assert_eq!(result.get([1]), 200.0);  // cond=0, select on_false
        assert_eq!(result.get([2]), 30.0);   // cond=1, select on_true
        assert_eq!(result.get([3]), 400.0);  // cond=0, select on_false
    }

    #[test]
    fn test_where_cond_i32() {
        let cond = ConcreteTensor::<i32, 1>::from_slice([4], &[1, 0, -1, 0]);
        let on_true = ConcreteTensor::<i32, 1>::from_slice([4], &[10, 20, 30, 40]);
        let on_false = ConcreteTensor::<i32, 1>::from_slice([4], &[100, 200, 300, 400]);

        let result = where_cond_ref(&cond, &on_true, &on_false);

        assert_eq!(result.get([0]), 10);   // cond=1 (nonzero), select on_true
        assert_eq!(result.get([1]), 200);  // cond=0, select on_false
        assert_eq!(result.get([2]), 30);   // cond=-1 (nonzero), select on_true
        assert_eq!(result.get([3]), 400);  // cond=0, select on_false
    }
}
