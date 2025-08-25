use crate::sgemv::SgemvParams;

#[inline]
pub fn gemv_parameters(m: usize, _: usize, k: usize) -> SgemvParams {
    let m = m as f32;
    let k = k as f32;
    match (m as u32, k as u32) {
        (..=512, ..=512) => SgemvParams::new(2, 4),
        (..=512, ..=1024) => SgemvParams::new(2, 4),
        (..=512, ..=2048) => SgemvParams::new(2, 4),
        (..=512, ..=4096) => SgemvParams::new(2, 4),
        (..=512, _) => SgemvParams::new(2, 4),

        (..=1024, ..=512) => SgemvParams::new(8, 2),
        (..=1024, ..=1024) => SgemvParams::new(16, 4),
        (..=1024, ..=2048) => SgemvParams::new(8, 2),
        (..=1024, ..=4096) => SgemvParams::new(16, 4),
        (..=1024, _) => SgemvParams::new(16, 4),

        (..=2048, ..=512) => SgemvParams::new(4, 4),
        (..=2048, ..=1024) => SgemvParams::new(16, 4),
        (..=2048, ..=2048) => SgemvParams::new(16, 2),
        (..=2048, ..=4096) => SgemvParams::new(32, 1),
        (..=2048, _) => SgemvParams::new(32, 1),

        (..=4096, ..=512) => SgemvParams::new(8, 2),
        (..=4096, ..=1024) => SgemvParams::new(16, 2),
        (..=4096, ..=2048) => SgemvParams::new(1, 4),
        (..=4096, ..=4096) => SgemvParams::new(32, 2),
        (..=4096, _) => SgemvParams::new(32, 2),

        (_, _) => SgemvParams::new(32, 2),
    }
}
