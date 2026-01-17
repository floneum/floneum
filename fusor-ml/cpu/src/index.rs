//! Index operations: index_select (gather)

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Compute the output shape for index_select
fn index_select_output_shape<const R: usize>(
    input_shape: &[usize],
    dimension: usize,
    num_indices: usize,
) -> [usize; R] {
    let mut output_shape = [0usize; R];
    for (i, &dim) in input_shape.iter().enumerate() {
        output_shape[i] = if i == dimension { num_indices } else { dim };
    }
    output_shape
}

/// Index select: gather elements along a dimension using indices
///
/// For a 2D tensor with shape [M, N] and indices [I]:
/// - index_select(dim=0, indices) -> shape [I, N], selecting rows
/// - index_select(dim=1, indices) -> shape [M, I], selecting columns
pub(crate) fn index_select_ref<E, const R: usize>(
    input: &ConcreteTensor<E, R>,
    dimension: usize,
    indices: &ConcreteTensor<u32, 1>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement,
{
    assert!(dimension < R, "dimension out of bounds");

    let input_shape = ResolvedTensor::shape(input);
    let num_indices = indices.data().len();
    let output_shape = index_select_output_shape::<R>(input_shape, dimension, num_indices);

    let mut output = ConcreteTensor::<E, R>::uninit_unchecked(output_shape);

    // Compute strides for iteration
    let input_strides = ResolvedTensor::strides(input);
    let output_strides = output.layout().strides.clone();

    // For each position in output, compute corresponding input position
    let total_elements: usize = output_shape.iter().product();

    for out_linear in 0..total_elements {
        // Convert linear index to multi-dimensional indices for output
        let mut out_indices = [0usize; R];
        let mut remaining = out_linear;
        for i in 0..R {
            out_indices[i] = remaining / output_strides[i];
            remaining %= output_strides[i];
        }

        // Build input indices: same as output, except at `dimension` we use indices[out_indices[dimension]]
        let mut in_indices = out_indices;
        let index_pos = out_indices[dimension];
        let actual_index = indices.data()[index_pos] as usize;
        in_indices[dimension] = actual_index;

        // Compute linear index for input
        let in_linear: usize = in_indices
            .iter()
            .zip(input_strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        output.data_mut()[out_linear] = input.data()[in_linear];
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_select_dim0() {
        // [[1, 2, 3], [4, 5, 6]] with indices [1, 0] -> [[4, 5, 6], [1, 2, 3]]
        let input = ConcreteTensor::<f32, 2>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = ConcreteTensor::<u32, 1>::from_slice([2], &[1, 0]);

        let result = index_select_ref(&input, 0, &indices);

        assert_eq!(result.get([0, 0]), 4.0);
        assert_eq!(result.get([0, 1]), 5.0);
        assert_eq!(result.get([0, 2]), 6.0);
        assert_eq!(result.get([1, 0]), 1.0);
        assert_eq!(result.get([1, 1]), 2.0);
        assert_eq!(result.get([1, 2]), 3.0);
    }

    #[test]
    fn test_index_select_dim1() {
        // [[1, 2, 3], [4, 5, 6]] with indices [1, 2, 0] -> [[2, 3, 1], [5, 6, 4]]
        let input = ConcreteTensor::<f32, 2>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = ConcreteTensor::<u32, 1>::from_slice([3], &[1, 2, 0]);

        let result = index_select_ref(&input, 1, &indices);

        assert_eq!(result.get([0, 0]), 2.0);
        assert_eq!(result.get([0, 1]), 3.0);
        assert_eq!(result.get([0, 2]), 1.0);
        assert_eq!(result.get([1, 0]), 5.0);
        assert_eq!(result.get([1, 1]), 6.0);
        assert_eq!(result.get([1, 2]), 4.0);
    }

    #[test]
    fn test_index_select_1d() {
        let input = ConcreteTensor::<f32, 1>::from_slice([5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let indices = ConcreteTensor::<u32, 1>::from_slice([3], &[4, 2, 0]);

        let result = index_select_ref(&input, 0, &indices);

        assert_eq!(result.get([0]), 50.0);
        assert_eq!(result.get([1]), 30.0);
        assert_eq!(result.get([2]), 10.0);
    }
}
