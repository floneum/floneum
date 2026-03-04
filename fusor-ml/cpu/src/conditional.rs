//! Conditional tensor operations: where_cond
//! Selects elements based on condition tensor != 0

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Helper trait for types that can be compared to zero
pub trait IsNonZero: SimdElement {
    fn is_nonzero(&self) -> bool;
}

impl IsNonZero for f32 {
    fn is_nonzero(&self) -> bool {
        *self != 0.0
    }
}

impl IsNonZero for f64 {
    fn is_nonzero(&self) -> bool {
        *self != 0.0
    }
}

impl IsNonZero for i8 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for i16 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for i32 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for i64 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for u8 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for u16 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for u32 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

impl IsNonZero for u64 {
    fn is_nonzero(&self) -> bool {
        *self != 0
    }
}

/// Conditional selection: where condition != 0, select on_true, else on_false
#[inline(always)]
pub(crate) fn where_cond_ref<E, const R: usize>(
    cond: &ConcreteTensor<E, R>,
    on_true: &ConcreteTensor<E, R>,
    on_false: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement + IsNonZero,
{
    let shape: [usize; R] = cond
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");

    let all_contiguous = cond.layout().is_contiguous()
        && on_true.layout().is_contiguous()
        && on_false.layout().is_contiguous();

    if all_contiguous {
        let cond_data = cond.data();
        let true_data = on_true.data();
        let false_data = on_false.data();
        ConcreteTensor::from_fn(shape, |i| {
            if cond_data[i].is_nonzero() {
                true_data[i]
            } else {
                false_data[i]
            }
        })
    } else {
        let output_layout = fusor_types::Layout::contiguous(&shape);
        let output_strides: Box<[usize]> = output_layout.strides().into();

        ConcreteTensor::from_fn(shape, |out_idx| {
            // Convert linear index to multi-dimensional indices
            let mut indices = vec![0usize; R];
            let mut remaining = out_idx;
            for i in 0..R {
                indices[i] = remaining / output_strides[i];
                remaining %= output_strides[i];
            }

            let cond_idx = cond.layout().linear_index(&indices);
            let true_idx = on_true.layout().linear_index(&indices);
            let false_idx = on_false.layout().linear_index(&indices);

            let cond_val = cond.data()[cond_idx];
            if cond_val.is_nonzero() {
                on_true.data()[true_idx]
            } else {
                on_false.data()[false_idx]
            }
        })
    }
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

        assert_eq!(result.get([0]), 10.0); // cond=1, select on_true
        assert_eq!(result.get([1]), 200.0); // cond=0, select on_false
        assert_eq!(result.get([2]), 30.0); // cond=1, select on_true
        assert_eq!(result.get([3]), 400.0); // cond=0, select on_false
    }

    #[test]
    fn test_where_cond_i32() {
        let cond = ConcreteTensor::<i32, 1>::from_slice([4], &[1, 0, -1, 0]);
        let on_true = ConcreteTensor::<i32, 1>::from_slice([4], &[10, 20, 30, 40]);
        let on_false = ConcreteTensor::<i32, 1>::from_slice([4], &[100, 200, 300, 400]);

        let result = where_cond_ref(&cond, &on_true, &on_false);

        assert_eq!(result.get([0]), 10); // cond=1 (nonzero), select on_true
        assert_eq!(result.get([1]), 200); // cond=0, select on_false
        assert_eq!(result.get([2]), 30); // cond=-1 (nonzero), select on_true
        assert_eq!(result.get([3]), 400); // cond=0, select on_false
    }
}
