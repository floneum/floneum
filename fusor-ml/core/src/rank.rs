use crate::{DataType, Tensor};

pub trait NextRankInner {
    type NextRank;
}

pub trait NextRank<const R: usize, D>: NextRankInner<NextRank = Tensor<R, D>> {}

impl<const R: usize, D, T> NextRank<R, D> for T where T: NextRankInner<NextRank = Tensor<R, D>> {}

pub trait LastRankInner {
    type LastRank;
}

pub trait LastRank<const R: usize, D>: LastRankInner<LastRank = Tensor<R, D>> {}

impl<const R: usize, D, T> LastRank<R, D> for T where T: LastRankInner<LastRank = Tensor<R, D>> {}

macro_rules! impl_next_last {
    ($R:expr) => {
        impl<D: DataType> NextRankInner for Tensor<$R, D> {
            type NextRank = Tensor<{ $R + 1 }, D>;
        }

        impl<D: DataType> LastRankInner for Tensor<$R, D> {
            type LastRank = Tensor<{ $R - 1 }, D>;
        }
    };
}

impl<D: DataType> NextRankInner for Tensor<0, D> {
    type NextRank = Tensor<1, D>;
}

impl_next_last!(1);
impl_next_last!(2);
impl_next_last!(3);
impl_next_last!(4);
impl_next_last!(5);
impl_next_last!(6);
impl_next_last!(7);
impl_next_last!(8);
impl_next_last!(9);
impl_next_last!(10);
impl_next_last!(11);
impl_next_last!(12);
impl_next_last!(13);
impl_next_last!(14);
impl_next_last!(15);
impl_next_last!(16);
impl_next_last!(17);
impl_next_last!(18);
impl_next_last!(19);
impl_next_last!(20);
