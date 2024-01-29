use candle_core::{Device, Tensor};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;

static SILU_CACHE: once_cell::sync::Lazy<Vec<f32>> = once_cell::sync::Lazy::new(|| {
    let f16_count = 2 << 16;
    let mut cache = Vec::with_capacity(f16_count);
    for i in 0..f16_count {
        let x = half::f16::from_bits(i as u16).to_f32();
        cache.push(x / (1. + (-x).exp()));
    }
    cache
});

#[inline(always)]
fn silu(x: &mut f32) {
    let cache: &[f32] = SILU_CACHE.as_ref();
    let as_f16 = half::f16::from_f32(*x);
    let as_f16 = as_f16.to_bits();
    let as_f16 = as_f16 as usize;
    *x = cache[as_f16];
}

#[inline(always)]
fn silu_chunk(chunk: &mut [f32; 16]) {
    for entry in chunk {
        silu(entry);
    }
}

pub(crate) fn fast_cpu_silu(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let shape = tensor.shape();

    let mut as_vec = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut iter = as_vec.par_chunks_exact_mut(16);
    for item in iter.remainder() {
        silu(item)
    }
    iter.for_each(|chunk| {
        let chunk: &mut [f32; 16] = unsafe { chunk.try_into().unwrap_unchecked() };
        silu_chunk(chunk)
    });

    Tensor::from_vec(as_vec, shape, &Device::Cpu)
}
