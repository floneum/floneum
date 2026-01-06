use generativity::Id;
use pulp::{Scalar256b, Simd, m8, m16, m32, m64};

struct Dim<'a> {
    id: Id<'a>,
}

trait SimdExt: Simd {
    type Wide<T: SimdElement>: Wide
    where
        Self: WideForSimd<T>;
}

trait Wide {
    const LANES: usize;
    type Mask: Wide<LANES = Self::LANES>;
}

macro_rules! impl_wide_scalar {
    ($($ty:ty [$base:ty] => $mask:ty),* $(,)?) => {
        $(
            impl Wide for $ty {
                const LANES: usize = std::mem::size_of::<$ty>() / std::mem::size_of::<$mask>();
                type Mask = $mask;
            }
        )*
    };
}

impl_wide_scalar!(
    f32x16 [f32] => m32x16,
    f32x4 [f32] => m32x4,
    f32x8 [f32] => m32x8,
    f64x2 [f64] => m64x2,
    f64x4 [f64] => m64x4,
    f64x8 [f64] => m64x8,
    i16x16 [i16] => m16x16,
    i16x32
    i16x8
    i32x16
    i32x4
    i32x8
    i64x2
    i64x4
    i64x8
    i8x16
    i8x32
    i8x64
    m16
    m16x16
    m16x32
    m16x8
    m32
    m32x16
    m32x4
    m32x8
    m64
    m64x2
    m64x4
    m64x8
    m8
    m8x16
    m8x32
    m8x64
    u16x16
    u16x32
    u16x8
    u32x16
    u32x4
    u32x8
    u64x2
    u64x4
    u64x8
    u8x16
    u8x32
    u8x64
);

impl<S: Simd> SimdExt for S {
    type Wide<T: SimdElement>
        = <Self as WideForSimd<T>>::Wide
    where
        Self: WideForSimd<T>;
}

trait WideForSimd<T: SimdElement>: Simd {
    type Wide: Wide;
}

macro_rules! impl_wide_for_simd {
    ($simd:ty => { $($ty:ty => $wide:ident),* }) => {
        $(
            impl<S: Simd> WideForSimd<$ty> for S {
                type Wide = <S as Simd>::$wide;
            }
        )*
    };
}

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

macro_rules! impl_simd_element {
    ($($ty:ty),*) => {
        mod sealed {
            use super::*;

            pub trait Sealed {}

            $(
                impl Sealed for $ty {}
            )*
        }

        trait SimdElement: sealed::Sealed {}

        $(
            impl SimdElement for $ty {}
        )*
    };
}

impl_simd_element!(
    u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, m8, m16, m32, m64
);
