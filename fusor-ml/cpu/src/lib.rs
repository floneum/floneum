use std::ops::Add as StdAdd;

use aligned_vec::{ABox, AVec};
use generativity::Id;
use pulp::{Arch, Simd, WithSimd, m8, m16, m32, m64};

struct Dim<'a> {
    id: Id<'a>,
}

fn with_simd_dim<F, R>(f: F) -> R
where
    F: for<'a> FnOnce(Dim<'a>) -> R,
{
    generativity::make_guard!(id);
    f(Dim { id: id.into() })
}

trait Tensor {
    type Elem: SimdElement;
    const RANK: usize;
    type Concrete: ResolvedTensor<Elem = Self::Elem>;
    const ASSERT: () = {
        assert!(
            Self::Concrete::RANK == Self::RANK,
            "Tensor rank mismatch in ConcreteTensor"
        );
    };
    fn to_concrete(&self) -> Self::Concrete;
}

trait ResolvedTensor: Tensor {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn data(&self) -> &ABox<[Self::Elem]>;
}

trait ResolvedTensorMut: ResolvedTensor {
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]>;
}

/// Helper to iterate over indices of a tensor with given shape
struct IndexIterator {
    shape: Box<[usize]>,
    indices: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    fn new(shape: &[usize]) -> Self {
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

#[derive(Clone, Debug, PartialEq)]
struct Layout {
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Layout {
    fn contiguous(shape: &[usize]) -> Self {
        let strides = Self::contiguous_strides(shape);
        Self {
            offset: 0,
            shape: shape.into(),
            strides,
        }
    }

    fn from_parts(offset: usize, shape: Box<[usize]>, strides: Box<[usize]>) -> Self {
        Self {
            offset,
            shape,
            strides,
        }
    }

    fn contiguous_strides(shape: &[usize]) -> Box<[usize]> {
        let mut acc = 1;
        let mut strides = vec![0; shape.len()].into_boxed_slice();
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == Self::contiguous_strides(&self.shape)
    }

    fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate the linear index for a given set of logical indices
    fn linear_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>()
    }
}

#[derive(Clone)]
struct ConcreteTensor<T: SimdElement, const RANK: usize> {
    layout: Layout,
    backing: ABox<[T]>,
}

impl<T, const RANK: usize> Tensor for ConcreteTensor<T, RANK>
where
    T: SimdElement,
{
    type Elem = T;
    const RANK: usize = RANK;
    type Concrete = Self;
    fn to_concrete(&self) -> Self::Concrete {
        self.clone()
    }
}

impl<T, const RANK: usize> ResolvedTensor for ConcreteTensor<T, RANK>
where
    T: SimdElement,
{
    fn shape(&self) -> &[usize] {
        self.layout.shape.as_ref()
    }
    fn strides(&self) -> &[usize] {
        self.layout.strides.as_ref()
    }
    fn offset(&self) -> usize {
        self.layout.offset
    }
    fn data(&self) -> &ABox<[Self::Elem]> {
        &self.backing
    }
}

impl<T, const RANK: usize> ResolvedTensorMut for ConcreteTensor<T, RANK>
where
    T: SimdElement,
{
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        &mut self.backing
    }
}

impl<T: SimdElement, const RANK: usize> ConcreteTensor<T, RANK> {
    /// Create a new tensor with contiguous layout from shape, filled with zeros
    fn zeros(shape: [usize; RANK]) -> Self
    where
        T: Default,
    {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        let mut vec: AVec<T> = AVec::with_capacity(64, num_elements);
        for _ in 0..num_elements {
            vec.push(T::default());
        }
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Create a new tensor from existing data with contiguous layout
    fn from_slice(shape: [usize; RANK], data: &[T]) -> Self {
        let layout = Layout::contiguous(&shape);
        assert_eq!(layout.num_elements(), data.len());
        let mut vec: AVec<T> = AVec::with_capacity(64, data.len());
        for &item in data {
            vec.push(item);
        }
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Get a reference to the layout
    fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Calculate the linear index for given logical indices
    fn linear_index(&self, indices: &[usize; RANK]) -> usize {
        self.layout.linear_index(indices)
    }

    /// Get element at logical indices
    fn get(&self, indices: [usize; RANK]) -> T {
        let idx = self.linear_index(&indices);
        self.backing[idx]
    }

    /// Set element at logical indices
    fn set(&mut self, indices: [usize; RANK], value: T) {
        let idx = self.linear_index(&indices);
        self.backing[idx] = value;
    }
}

struct Add<E: SimdElement, const RANK: usize, T1: Tensor<Elem = E>, T2: Tensor<Elem = E>> {
    lhs: T1,
    rhs: T2,
    _marker: std::marker::PhantomData<E>,
}

impl<E, const RANK: usize, T1, T2> Add<E, RANK, T1, T2>
where
    E: SimdElement,
    T1: Tensor<Elem = E>,
    T2: Tensor<Elem = E>,
{
    fn new(lhs: T1, rhs: T2) -> Self {
        Self {
            lhs,
            rhs,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E, const RANK: usize, T1, T2> Tensor for Add<E, RANK, T1, T2>
where
    E: SimdElement + StdAdd<Output = E> + Default + SimdAdd,
    T1: Tensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
    T2: Tensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
{
    type Elem = E;
    const RANK: usize = {
        assert!(T2::RANK == T1::RANK, "Tensor rank mismatch in Add");
        T1::RANK
    };
    type Concrete = ConcreteTensor<Self::Elem, RANK>;

    fn to_concrete(&self) -> Self::Concrete {
        let lhs_concrete = self.lhs.to_concrete();
        let rhs_concrete = self.rhs.to_concrete();

        // Create output tensor with same shape, contiguous layout
        let shape: [usize; RANK] = lhs_concrete
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<Self::Elem, RANK>::zeros(shape);

        // Clone layouts to avoid borrow checker issues
        let lhs_layout = lhs_concrete.layout().clone();
        let rhs_layout = rhs_concrete.layout().clone();
        let out_layout = output.layout().clone();
        let tensor_shape: Box<[usize]> = lhs_concrete.shape().into();

        // Check if all tensors are contiguous - use fast SIMD path
        let all_contiguous =
            lhs_layout.is_contiguous() && rhs_layout.is_contiguous() && out_layout.is_contiguous();

        if all_contiguous {
            // Use arch dispatch for SIMD operations
            struct AddOp<'a, E> {
                lhs: &'a [E],
                rhs: &'a [E],
                out: &'a mut [E],
            }

            impl<E: SimdAdd> WithSimd for AddOp<'_, E> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    E::simd_add(simd, self.lhs, self.rhs, self.out);
                }
            }

            let lhs_slice: &[E] = &lhs_concrete.backing;
            let rhs_slice: &[E] = &rhs_concrete.backing;
            let out_slice: &mut [E] = &mut output.backing;

            Arch::new().dispatch(AddOp {
                lhs: lhs_slice,
                rhs: rhs_slice,
                out: out_slice,
            });
        } else {
            // General strided case - scalar only
            let lhs_data = lhs_concrete.data();
            let rhs_data = rhs_concrete.data();
            let out_data = output.data_mut();

            for indices in IndexIterator::new(&tensor_shape) {
                let lhs_idx = lhs_layout.linear_index(&indices);
                let rhs_idx = rhs_layout.linear_index(&indices);
                let out_idx = out_layout.linear_index(&indices);
                out_data[out_idx] = lhs_data[lhs_idx] + rhs_data[rhs_idx];
            }
        }

        output
    }
}

/// Trait for element types that support SIMD addition via pulp.
/// Each element type implements the SIMD operations using the specific
/// Simd instance passed to with_simd.
trait SimdAdd: SimdElement + StdAdd<Output = Self> {
    /// Perform SIMD-accelerated addition on contiguous slices.
    /// Uses the provided Simd instance for vectorized operations.
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]);
}

impl SimdAdd for f32 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::f32s>() / std::mem::size_of::<f32>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        // Process SIMD chunks
        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::f32s);
                let b = *(rhs_ptr.add(i) as *const S::f32s);
                let c = simd.add_f32s(a, b);
                *(out_ptr.add(i) as *mut S::f32s) = c;
            }
            i += lanes;
        }

        // Process remainder
        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for f64 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::f64s>() / std::mem::size_of::<f64>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::f64s);
                let b = *(rhs_ptr.add(i) as *const S::f64s);
                let c = simd.add_f64s(a, b);
                *(out_ptr.add(i) as *mut S::f64s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for i8 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::i8s>() / std::mem::size_of::<i8>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::i8s);
                let b = *(rhs_ptr.add(i) as *const S::i8s);
                let c = simd.add_i8s(a, b);
                *(out_ptr.add(i) as *mut S::i8s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for i16 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::i16s>() / std::mem::size_of::<i16>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::i16s);
                let b = *(rhs_ptr.add(i) as *const S::i16s);
                let c = simd.add_i16s(a, b);
                *(out_ptr.add(i) as *mut S::i16s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for i32 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::i32s>() / std::mem::size_of::<i32>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::i32s);
                let b = *(rhs_ptr.add(i) as *const S::i32s);
                let c = simd.add_i32s(a, b);
                *(out_ptr.add(i) as *mut S::i32s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for i64 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::i64s>() / std::mem::size_of::<i64>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::i64s);
                let b = *(rhs_ptr.add(i) as *const S::i64s);
                let c = simd.add_i64s(a, b);
                *(out_ptr.add(i) as *mut S::i64s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for u8 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::u8s>() / std::mem::size_of::<u8>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::u8s);
                let b = *(rhs_ptr.add(i) as *const S::u8s);
                let c = simd.add_u8s(a, b);
                *(out_ptr.add(i) as *mut S::u8s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for u16 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::u16s>() / std::mem::size_of::<u16>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::u16s);
                let b = *(rhs_ptr.add(i) as *const S::u16s);
                let c = simd.add_u16s(a, b);
                *(out_ptr.add(i) as *mut S::u16s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for u32 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::u32s>() / std::mem::size_of::<u32>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::u32s);
                let b = *(rhs_ptr.add(i) as *const S::u32s);
                let c = simd.add_u32s(a, b);
                *(out_ptr.add(i) as *mut S::u32s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

impl SimdAdd for u64 {
    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        let len = out.len();
        let lanes = std::mem::size_of::<S::u64s>() / std::mem::size_of::<u64>();

        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let simd_len = len / lanes * lanes;
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let a = *(lhs_ptr.add(i) as *const S::u64s);
                let b = *(rhs_ptr.add(i) as *const S::u64s);
                let c = simd.add_u64s(a, b);
                *(out_ptr.add(i) as *mut S::u64s) = c;
            }
            i += lanes;
        }

        for j in simd_len..len {
            out[j] = lhs[j] + rhs[j];
        }
    }
}

/// Marker trait for SIMD element types
trait SimdElement: Sized + Copy + Default {}

impl SimdElement for f32 {}
impl SimdElement for f64 {}
impl SimdElement for i8 {}
impl SimdElement for i16 {}
impl SimdElement for i32 {}
impl SimdElement for i64 {}
impl SimdElement for u8 {}
impl SimdElement for u16 {}
impl SimdElement for u32 {}
impl SimdElement for u64 {}
impl SimdElement for m8 {}
impl SimdElement for m16 {}
impl SimdElement for m32 {}
impl SimdElement for m64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_contiguous() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        assert_eq!(layout.shape.as_ref(), &[2, 3, 4]);
        assert_eq!(layout.strides.as_ref(), &[12, 4, 1]);
        assert_eq!(layout.offset, 0);
        assert!(layout.is_contiguous());
        assert_eq!(layout.num_elements(), 24);
    }

    #[test]
    fn test_layout_linear_index() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        assert_eq!(layout.linear_index(&[0, 0, 0]), 0);
        assert_eq!(layout.linear_index(&[0, 0, 1]), 1);
        assert_eq!(layout.linear_index(&[0, 1, 0]), 4);
        assert_eq!(layout.linear_index(&[1, 0, 0]), 12);
        assert_eq!(layout.linear_index(&[1, 2, 3]), 12 + 8 + 3);
    }

    #[test]
    fn test_layout_with_offset() {
        let layout =
            Layout::from_parts(10, vec![2, 3].into_boxed_slice(), vec![3, 1].into_boxed_slice());
        assert_eq!(layout.offset, 10);
        assert_eq!(layout.linear_index(&[0, 0]), 10);
        assert_eq!(layout.linear_index(&[1, 2]), 10 + 3 + 2);
    }

    #[test]
    fn test_concrete_tensor_creation() {
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data().len(), 6);
        for i in 0..6 {
            assert_eq!(tensor.data()[i], 0.0);
        }
    }

    #[test]
    fn test_concrete_tensor_from_slice() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get([0, 0]), 0.0);
        assert_eq!(tensor.get([0, 1]), 1.0);
        assert_eq!(tensor.get([0, 2]), 2.0);
        assert_eq!(tensor.get([1, 0]), 3.0);
        assert_eq!(tensor.get([1, 1]), 4.0);
        assert_eq!(tensor.get([1, 2]), 5.0);
    }

    #[test]
    fn test_concrete_tensor_get_set() {
        let mut tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
        tensor.set([0, 1], 42.0);
        tensor.set([1, 2], 100.0);
        assert_eq!(tensor.get([0, 1]), 42.0);
        assert_eq!(tensor.get([1, 2]), 100.0);
        assert_eq!(tensor.get([0, 0]), 0.0);
    }

    #[test]
    fn test_add_tensor_contiguous() {
        let lhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get([0, 0]), 11.0);
        assert_eq!(result.get([0, 1]), 22.0);
        assert_eq!(result.get([0, 2]), 33.0);
        assert_eq!(result.get([1, 0]), 44.0);
        assert_eq!(result.get([1, 1]), 55.0);
        assert_eq!(result.get([1, 2]), 66.0);
    }

    #[test]
    fn test_add_tensor_1d() {
        let lhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[1, 2, 3, 4]);
        let rhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[10, 20, 30, 40]);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.get([0]), 11);
        assert_eq!(result.get([1]), 22);
        assert_eq!(result.get([2]), 33);
        assert_eq!(result.get([3]), 44);
    }

    #[test]
    fn test_add_tensor_3d() {
        let lhs_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs_data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let lhs: ConcreteTensor<f64, 3> = ConcreteTensor::from_slice([2, 2, 2], &lhs_data);
        let rhs: ConcreteTensor<f64, 3> = ConcreteTensor::from_slice([2, 2, 2], &rhs_data);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        assert_eq!(result.shape(), &[2, 2, 2]);
        assert!((result.get([0, 0, 0]) - 1.1).abs() < 1e-10);
        assert!((result.get([0, 0, 1]) - 2.2).abs() < 1e-10);
        assert!((result.get([1, 1, 1]) - 8.8).abs() < 1e-10);
    }

    #[test]
    fn test_add_large_tensor() {
        // Test with a larger tensor to exercise SIMD paths
        let size = 1024;
        let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
        let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        for i in 0..size {
            assert_eq!(result.get([i]), (i + i * 2) as f32);
        }
    }

    #[test]
    fn test_index_iterator() {
        let shape = [2, 3];
        let indices: Vec<Vec<usize>> = IndexIterator::new(&shape).collect();

        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[1], vec![0, 1]);
        assert_eq!(indices[2], vec![0, 2]);
        assert_eq!(indices[3], vec![1, 0]);
        assert_eq!(indices[4], vec![1, 1]);
        assert_eq!(indices[5], vec![1, 2]);
    }

    #[test]
    fn test_index_iterator_empty() {
        let shape = [0, 3];
        let indices: Vec<Vec<usize>> = IndexIterator::new(&shape).collect();
        assert!(indices.is_empty());
    }
}
