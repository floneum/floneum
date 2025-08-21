use crate::sgemv::SgemvParams;

#[inline]
pub fn gemv_parameters(m: usize, n: usize, k: usize) -> SgemvParams {
    let m = m as f32;
    let k = k as f32;
    if k <= 384f32 {
        SgemvParams::new(2u32, 2u32)
    } else {
        if m <= 768f32 {
            SgemvParams::new(2u32, 2u32)
        } else {
            if m <= 1536f32 {
                SgemvParams::new(2u32, 2u32)
            } else {
                if k <= 1536f32 {
                    SgemvParams::new(4u32, 4u32)
                } else {
                    SgemvParams::new(4u32, 4u32)
                }
            }
        }
    }
}
