use crate::{DataType, LastRank, Tensor};
use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Index into a tensor with a tuple of ranges and indices
    /// This implements indexing similar to PyTorch's tensor[(..., 0, ...)] syntax
    pub fn i<I: TensorIndex<R, D, Phantom>, Phantom>(&self, indices: I) -> I::Output {
        indices.index(self)
    }
}

pub trait TensorIndex<const R: usize, D: DataType, Priv = ()> {
    type Output;

    fn index(&self, tensor: &Tensor<R, D>) -> Self::Output;
}

#[doc(hidden)]
pub struct PhantomConst<const N: usize, T = ()>(PhantomData<T>);

impl<
    const R1: usize,
    const R2: usize,
    T1: TensorIndexComponent<1, R1, D, Output = Tensor<R2, D>>,
    D: DataType,
> TensorIndex<1, D, PhantomConst<R1, PhantomConst<R2>>> for (T1,)
{
    type Output = Tensor<R2, D>;

    fn index(&self, tensor: &Tensor<1, D>) -> Self::Output {
        let [t1_shape] = *tensor.shape();
        let (t1,) = self;

        let slices = [t1.range(t1_shape)];

        let sliced = tensor.slice(slices);
        t1.squeeze(sliced, 0)
    }
}

impl<
    const R1: usize,
    const R2: usize,
    T1: TensorIndexComponent<R1, R2, D, Output = Tensor<R2, D>>,
    T2: TensorIndexComponent<2, R1, D, Output = Tensor<R1, D>>,
    D: DataType,
> TensorIndex<2, D, PhantomConst<R1, PhantomConst<R2>>> for (T1, T2)
{
    type Output = Tensor<R2, D>;

    fn index(&self, tensor: &Tensor<2, D>) -> Self::Output {
        let [t1_shape, t2_shape] = *tensor.shape();
        let (t1, t2) = self;

        let slices = [t1.range(t1_shape), t2.range(t2_shape)];

        let sliced = tensor.slice(slices);
        let out = t2.squeeze(sliced, 1);
        t1.squeeze(out, 0)
    }
}

impl<
    const R1: usize,
    const R2: usize,
    const R3: usize,
    T1: TensorIndexComponent<R2, R3, D, Output = Tensor<R3, D>>,
    T2: TensorIndexComponent<R1, R2, D, Output = Tensor<R2, D>>,
    T3: TensorIndexComponent<3, R1, D, Output = Tensor<R1, D>>,
    D: DataType,
> TensorIndex<3, D, PhantomConst<R1, PhantomConst<R2, PhantomConst<R3>>>> for (T1, T2, T3)
{
    type Output = Tensor<R3, D>;

    fn index(&self, tensor: &Tensor<3, D>) -> Self::Output {
        let [t1_shape, t2_shape, t3_shape] = *tensor.shape();
        let (t1, t2, t3) = self;

        let slices = [t1.range(t1_shape), t2.range(t2_shape), t3.range(t3_shape)];

        let sliced = tensor.slice(slices);
        let out = t3.squeeze(sliced, 2);
        let out = t2.squeeze(out, 1);
        t1.squeeze(out, 0)
    }
}

impl<
    const R1: usize,
    const R2: usize,
    const R3: usize,
    const R4: usize,
    T1: TensorIndexComponent<R3, R4, D, Output = Tensor<R4, D>>,
    T2: TensorIndexComponent<R2, R3, D, Output = Tensor<R3, D>>,
    T3: TensorIndexComponent<R1, R2, D, Output = Tensor<R2, D>>,
    T4: TensorIndexComponent<4, R1, D, Output = Tensor<R1, D>>,
    D: DataType,
> TensorIndex<4, D, PhantomConst<R1, PhantomConst<R2, PhantomConst<R3, PhantomConst<R4>>>>>
    for (T1, T2, T3, T4)
{
    type Output = Tensor<R4, D>;

    fn index(&self, tensor: &Tensor<4, D>) -> Self::Output {
        let [t1_shape, t2_shape, t3_shape, t4_shape] = *tensor.shape();
        let (t1, t2, t3, t4) = self;
        let slices = [
            t1.range(t1_shape),
            t2.range(t2_shape),
            t3.range(t3_shape),
            t4.range(t4_shape),
        ];

        let sliced = tensor.slice(slices);
        let out = t4.squeeze(sliced, 3);
        let out = t3.squeeze(out, 2);
        let out = t2.squeeze(out, 1);
        t1.squeeze(out, 0)
    }
}

impl<
    const R1: usize,
    const R2: usize,
    const R3: usize,
    const R4: usize,
    const R5: usize,
    T1: TensorIndexComponent<R4, R5, D, Output = Tensor<R5, D>>,
    T2: TensorIndexComponent<R3, R4, D, Output = Tensor<R4, D>>,
    T3: TensorIndexComponent<R2, R3, D, Output = Tensor<R3, D>>,
    T4: TensorIndexComponent<R1, R2, D, Output = Tensor<R2, D>>,
    T5: TensorIndexComponent<5, R1, D, Output = Tensor<R1, D>>,
    D: DataType,
> TensorIndex<5, D, PhantomConst<R1, PhantomConst<R2, PhantomConst<R3, PhantomConst<R4, PhantomConst<R5>>>>>> for (T1, T2, T3, T4, T5)
{
    type Output = Tensor<R5, D>;

    fn index(&self, tensor: &Tensor<5, D>) -> Self::Output {
        let [t1_shape, t2_shape, t3_shape, t4_shape, t5_shape] = *tensor.shape();
        let (t1, t2, t3, t4, t5) = self;
        let slices = [
            t1.range(t1_shape),
            t2.range(t2_shape),
            t3.range(t3_shape),
            t4.range(t4_shape),
            t5.range(t5_shape),
        ];

        let sliced = tensor.slice(slices);
        let out = t5.squeeze(sliced, 4);
        let out = t4.squeeze(out, 3);
        let out = t3.squeeze(out, 2);
        let out = t2.squeeze(out, 1);
        t1.squeeze(out, 0)
    }
}

trait TensorIndexComponent<const R: usize, const R2: usize, D> {
    // type Input<const R: usize, const R2: usize, D>: LastRank<R2, D>;
    type Output;
    fn squeeze(&self, tensor: Tensor<R, D>, axis: usize) -> Self::Output;
    fn range(&self, size: usize) -> Range<usize>;
}

impl<const R: usize, D: DataType> TensorIndexComponent<R, R, D> for RangeFull {
    type Output = Tensor<R, D>;
    fn squeeze(&self, tensor: Tensor<R, D>, _: usize) -> Self::Output {
        tensor
    }
    fn range(&self, size: usize) -> Range<usize> {
        0..size
    }
}

impl<const R: usize, D: DataType> TensorIndexComponent<R, R, D> for RangeTo<usize> {
    type Output = Tensor<R, D>;
    fn squeeze(&self, tensor: Tensor<R, D>, _: usize) -> Self::Output {
        tensor
    }
    fn range(&self, _: usize) -> Range<usize> {
        0..self.end
    }
}

impl<const R: usize, D: DataType> TensorIndexComponent<R, R, D> for RangeFrom<usize> {
    type Output = Tensor<R, D>;
    fn squeeze(&self, tensor: Tensor<R, D>, _: usize) -> Self::Output {
        tensor
    }
    fn range(&self, size: usize) -> Range<usize> {
        self.start..size
    }
}

impl<const R: usize, D: DataType> TensorIndexComponent<R, R, D> for Range<usize> {
    type Output = Tensor<R, D>;
    fn squeeze(&self, tensor: Tensor<R, D>, _: usize) -> Self::Output {
        tensor
    }
    fn range(&self, _: usize) -> Range<usize> {
        self.clone()
    }
}

impl<const R: usize, const R2: usize, D: DataType> TensorIndexComponent<R, R2, D> for usize
where
    Tensor<R, D>: LastRank<R2, D>,
{
    type Output = Tensor<R2, D>;
    fn squeeze(&self, tensor: Tensor<R, D>, axis: usize) -> Self::Output {
        tensor.squeeze(axis)
    }
    fn range(&self, _: usize) -> Range<usize> {
        *self..(*self + 1)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_index() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
    let tensor = Tensor::new(&device, &data);
    let indexed = tensor.i((.., 0, ..));

    let indexed_slice = indexed.as_slice().await.unwrap();
    assert_eq!(indexed_slice[[0, 0]], 1.);
    assert_eq!(indexed_slice[[0, 1]], 2.);
    assert_eq!(indexed_slice[[1, 0]], 5.);
    assert_eq!(indexed_slice[[1, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_4d() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]];
    let tensor = Tensor::new(&device, &data);
    let indexed = tensor.i((.., ..1, 0, 0..));

    let indexed_slice = indexed.as_slice().await.unwrap();
    println!("{:?}", indexed_slice);
    assert_eq!(indexed_slice[[0, 0, 0]], 1.);
    assert_eq!(indexed_slice[[0, 0, 1]], 2.);
}
