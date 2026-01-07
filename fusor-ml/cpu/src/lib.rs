use aligned_vec::ABox;
use generativity::Id;
use pulp::{Scalar128b, Scalar256b, Scalar512b, Simd, m8, m16, m32, m64};
use pulp::{m8x16, m8x32, m8x64, m16x8, m16x16, m16x32, m32x4, m32x8, m32x16, m64x2, m64x4, m64x8};
use pulp::{f32x4, f32x8, f32x16, f64x2, f64x4, f64x8};
use pulp::{i8x16, i8x32, i8x64, i16x8, i16x16, i16x32, i32x4, i32x8, i32x16, i64x2, i64x4, i64x8};
use pulp::{u8x16, u8x32, u8x64, u16x8, u16x16, u16x32, u32x4, u32x8, u32x16, u64x2, u64x4, u64x8};

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
        assert!(Self::Concrete::RANK == Self::RANK, "Tensor rank mismatch in ConcreteTensor");
    };
    fn to_concrete(&self) -> Self::Concrete;
}

trait ResolvedTensor: Tensor {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn data(&self) -> &ABox<[Self::Elem]>;
}

trait ResolvedTensorExt: ResolvedTensor {
    fn visit_simds<S: SimdExt>(&self, simd: S)
    where
        Self::Elem: SimdElement,
    {
        // sort the strides to find the best dimension to vectorize over
        let mut stride_indices: Box<[usize]> = (0..Self::RANK).collect();
        stride_indices.sort_by_key(|&i| self.strides()[i]);

        let smallest_stride = stride_indices.first().map_or(usize::MAX, |&i| self.strides()[i]);
        let smallest_stride_len = stride_indices.first().map_or(0, |&i| self.shape()[i]);
        if smallest_stride == 1 {
            // we can vectorize over this dimension
            let lane_count = S::Wide::<Self::Elem>::LANES;
            let simd_chunks = smallest_stride_len / lane_count;
            let remainder = smallest_stride_len % lane_count;

            for chunk_idx in 0..simd_chunks {
                // process a full SIMD chunk
            }
        }
    }
}
impl<T: ResolvedTensor> ResolvedTensorExt for T {}

#[derive(Clone)]
struct Layout {
    shape: Box<[usize]>,
    strides: Box<[usize]>,
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
    fn data(&self) -> &ABox<[Self::Elem]> {
        &self.backing
    }
}

struct Add<const RANK: usize, T1: Tensor, T2: Tensor> {
    lhs: T1,
    rhs: T2,
}

impl<const RANK: usize, T1, T2> Tensor for Add<RANK, T1, T2>
where
    T1: Tensor,
    T2: Tensor<Elem = T1::Elem>,
{
    type Elem = T1::Elem;
    const RANK: usize = {
        assert!(T2::RANK == T1::RANK, "Tensor rank mismatch in Add");
        T1::RANK
    };
    type Concrete = ConcreteTensor<Self::Elem, RANK>;
    fn to_concrete(&self) -> Self::Concrete {
        let lhs_concrete = self.lhs.to_concrete();
        let rhs_concrete = self.rhs.to_concrete();
        todo!()
    }
}

trait SimdExt: Simd {
    type Wide<T: SimdElement>: Wide;
}

impl SimdExt for Scalar128b {
    type Wide<T: SimdElement>
        = <T as SimdElement>::Wide128b;
}

impl SimdExt for Scalar256b {
    type Wide<T: SimdElement>
        = <T as SimdElement>::Wide256b;
}

impl SimdExt for Scalar512b {
    type Wide<T: SimdElement>
        = <T as SimdElement>::Wide512b;
}

trait SimdElement: SimdElementMarker + Sized + Copy {
    type Wide128b: Wide;
    type Wide256b: Wide;
    type Wide512b: Wide;
}

impl<T> SimdElement for T where T: SimdElementMarker + Sized + Copy, Scalar128b: WideForSimd<T>, Scalar256b: WideForSimd<T>, Scalar512b: WideForSimd<T> {
    type Wide128b = <Scalar128b as WideForSimd<T>>::Wide;
    type Wide256b = <Scalar256b as WideForSimd<T>>::Wide;
    type Wide512b = <Scalar512b as WideForSimd<T>>::Wide;
}

trait Wide {
    const LANES: usize;
    type Mask: Wide;
}

macro_rules! impl_wide_scalar {
    ($($ty:ty [$base:ty] => $mask:ty),* $(,)?) => {
        $(
            impl Wide for $ty {
                const LANES: usize = std::mem::size_of::<$ty>() / std::mem::size_of::<$base>();
                type Mask = $mask;
            }
        )*
    };
}

impl_wide_scalar!(
    // f32 vectors
    f32x4 [f32] => m32x4,
    f32x8 [f32] => m32x8,
    f32x16 [f32] => m32x16,
    // f64 vectors
    f64x2 [f64] => m64x2,
    f64x4 [f64] => m64x4,
    f64x8 [f64] => m64x8,
    // i8 vectors
    i8x16 [i8] => m8x16,
    i8x32 [i8] => m8x32,
    i8x64 [i8] => m8x64,
    // i16 vectors
    i16x8 [i16] => m16x8,
    i16x16 [i16] => m16x16,
    i16x32 [i16] => m16x32,
    // i32 vectors
    i32x4 [i32] => m32x4,
    i32x8 [i32] => m32x8,
    i32x16 [i32] => m32x16,
    // i64 vectors
    i64x2 [i64] => m64x2,
    i64x4 [i64] => m64x4,
    i64x8 [i64] => m64x8,
    // u8 vectors
    u8x16 [u8] => m8x16,
    u8x32 [u8] => m8x32,
    u8x64 [u8] => m8x64,
    // u16 vectors
    u16x8 [u16] => m16x8,
    u16x16 [u16] => m16x16,
    u16x32 [u16] => m16x32,
    // u32 vectors
    u32x4 [u32] => m32x4,
    u32x8 [u32] => m32x8,
    u32x16 [u32] => m32x16,
    // u64 vectors
    u64x2 [u64] => m64x2,
    u64x4 [u64] => m64x4,
    u64x8 [u64] => m64x8,
    // scalar masks
    m8 [m8] => m8,
    m16 [m16] => m16,
    m32 [m32] => m32,
    m64 [m64] => m64,
    // m8 vectors
    m8x16 [m8] => m8x16,
    m8x32 [m8] => m8x32,
    m8x64 [m8] => m8x64,
    // m16 vectors
    m16x8 [m16] => m16x8,
    m16x16 [m16] => m16x16,
    m16x32 [m16] => m16x32,
    // m32 vectors
    m32x4 [m32] => m32x4,
    m32x8 [m32] => m32x8,
    m32x16 [m32] => m32x16,
    // m64 vectors
    m64x2 [m64] => m64x2,
    m64x4 [m64] => m64x4,
    m64x8 [m64] => m64x8,
    // scalars
    f32 [f32] => m32,
    f64 [f64] => m64,
    i8 [i8] => m8,
    i16 [i16] => m16,
    i32 [i32] => m32,
    i64 [i64] => m64,
    u8 [u8] => m8,
    u16 [u16] => m16,
    u32 [u32] => m32,
    u64 [u64] => m64,
);

trait WideForSimd<T: SimdElementMarker>: Simd {
    type Wide: Wide;
}

macro_rules! impl_wide_for_simd {
    ($simd:ty => { $($ty:ty => $wide:ident),* }) => {
        $(
            impl WideForSimd<$ty> for $simd {
                type Wide = <$simd as Simd>::$wide;
            }
        )*
    };
}

impl_wide_for_simd!(Scalar128b => {
    f32 => f32s,
    f64 => f64s,
    i16 => i16s,
    i32 => i32s,
    i64 => i64s,
    i8 => i8s,
    u16 => u16s,
    u32 => u32s,
    u64 => u64s,
    u8 => u8s,
    m8 => m8s,
    m16 => m16s,
    m32 => m32s,
    m64 => m64s
});

impl_wide_for_simd!(Scalar256b => {
    f32 => f32s,
    f64 => f64s,
    i16 => i16s,
    i32 => i32s,
    i64 => i64s,
    i8 => i8s,
    u16 => u16s,
    u32 => u32s,
    u64 => u64s,
    u8 => u8s,
    m8 => m8s,
    m16 => m16s,
    m32 => m32s,
    m64 => m64s
});

impl_wide_for_simd!(Scalar512b => {
    f32 => f32s,
    f64 => f64s,
    i16 => i16s,
    i32 => i32s,
    i64 => i64s,
    i8 => i8s,
    u16 => u16s,
    u32 => u32s,
    u64 => u64s,
    u8 => u8s,
    m8 => m8s,
    m16 => m16s,
    m32 => m32s,
    m64 => m64s
});

macro_rules! impl_simd_element {
    ($($ty:ty),*) => {
        mod sealed {
            use super::*;

            pub trait Sealed {}

            $(
                impl Sealed for $ty {}
            )*
        }

        trait SimdElementMarker: sealed::Sealed {}

        $(
            impl SimdElementMarker for $ty {}
        )*
    };
}

impl_simd_element!(
    u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, m8, m16, m32, m64
);
