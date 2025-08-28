use crate::{DataType, Tensor};

pub trait NextRankInner {
    type NextRank;
}

pub trait NextRank<const R: usize, D>: NextRankInner<NextRank = Tensor<R, D>> {}

impl<const R: usize, D, T> NextRank<R, D> for T where T: NextRankInner<NextRank = Tensor<R, D>> {}

pub trait SmallerRankInner<const R: usize> {
    type SmallerRank;
    type SmallerByArray;
}

pub trait SmallerRank<const R: usize, const S: usize, D>:
    SmallerRankInner<R, SmallerRank = Tensor<S, D>, SmallerByArray = [usize; R]>
{
}

impl<const R: usize, const S: usize, D, T> SmallerRank<R, S, D> for T where
    T: SmallerRankInner<R, SmallerRank = Tensor<S, D>, SmallerByArray = [usize; R]>
{
}

pub trait LastRankInner {
    type LastRank;
}

pub trait LastRank<const R: usize, D>: LastRankInner<LastRank = Tensor<R, D>> {}

impl<const R: usize, D, T> LastRank<R, D> for T where T: LastRankInner<LastRank = Tensor<R, D>> {}

pub trait LargerRankInner<const R: usize> {
    type LargerRank;
    type LargerByArray;
}

pub trait LargerRank<const R: usize, const L: usize, D>:
    LargerRankInner<R, LargerRank = Tensor<L, D>, LargerByArray = [usize; R]>
{
}

impl<const R: usize, const L: usize, D, T> LargerRank<R, L, D> for T where
    T: LargerRankInner<R, LargerRank = Tensor<L, D>, LargerByArray = [usize; R]>
{
}

pub trait MaxRankInner {
    type MaxRank;
}

pub trait MaxRank<const R: usize, D>: MaxRankInner<MaxRank = Tensor<R, D>> {}

impl<const R: usize, D, T> MaxRank<R, D> for T where T: MaxRankInner<MaxRank = Tensor<R, D>> {}

macro_rules! impl_next_last {
    ($($smaller:literal, )* [0] $(, $larger:literal)*) => {
        $(
            impl<D: DataType> SmallerRankInner<{0 - $smaller}> for Tensor<0, D> {
                type SmallerRank = Tensor<$smaller, D>;
                type SmallerByArray = [usize; {0 - $smaller}];
            }
        )*

        impl<D: DataType> NextRankInner for Tensor<0, D> {
            type NextRank = Tensor<1, D>;
        }

        impl<D: DataType> MaxRankInner for (Tensor<0, D>, Tensor<0, D>) {
            type MaxRank = Tensor<0, D>;
        }

        $(
            impl<D: DataType> LargerRankInner<{$larger - 0}> for Tensor<0, D> {
                type LargerRank = Tensor<$larger, D>;
                type LargerByArray = [usize; {$larger - 0}];
            }

            impl<D: DataType> MaxRankInner for (Tensor<0, D>, Tensor<$larger, D>) {
                type MaxRank = Tensor<$larger, D>;
            }

            impl<D: DataType> MaxRankInner for (Tensor<$larger, D>, Tensor<0, D>) {
                type MaxRank = Tensor<$larger, D>;
            }
        )*
    };

    ($($smaller:literal, )* [$R:literal] $(, $larger:literal)*) => {
        $(
            impl<D: DataType> SmallerRankInner<{$R - $smaller}> for Tensor<$R, D> {
                type SmallerRank = Tensor<$smaller, D>;
                type SmallerByArray = [usize; {$R - $smaller}];
            }
        )*

        impl<D: DataType> NextRankInner for Tensor<$R, D> {
            type NextRank = Tensor<{ $R + 1 }, D>;
        }

        impl<D: DataType> LastRankInner for Tensor<$R, D> {
            type LastRank = Tensor<{ $R - 1 }, D>;
        }

        impl<D: DataType> MaxRankInner for (Tensor<$R, D>, Tensor<$R, D>) {
            type MaxRank = Tensor<$R, D>;
        }

        $(
            impl<D: DataType> LargerRankInner<{$larger - $R}> for Tensor<$R, D> {
                type LargerRank = Tensor<$larger, D>;
                type LargerByArray = [usize; {$larger - $R}];
            }

            impl<D: DataType> MaxRankInner for (Tensor<$R, D>, Tensor<$larger, D>) {
                type MaxRank = Tensor<$larger, D>;
            }

            impl<D: DataType> MaxRankInner for (Tensor<$larger, D>, Tensor<$R, D>) {
                type MaxRank = Tensor<$larger, D>;
            }
        )*
    };
}

#[rustfmt::skip]
mod impls {
    use super::*;

    impl_next_last!([0], 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, [1], 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, [2], 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, [3], 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, [4], 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, [5], 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, [6], 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, [7], 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, [8], 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, [9], 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, [10], 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [11], 12, 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, [12], 13, 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, [13], 14, 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, [14], 15, 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, [15], 16, 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, [16], 17, 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, [17], 18, 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, [18], 19, 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, [19], 20);
    impl_next_last!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, [20]);
}
