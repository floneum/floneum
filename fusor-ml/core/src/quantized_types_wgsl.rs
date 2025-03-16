use std::fmt::Write;

use fusor_gguf::{BlockQ4K, BlockQ4_0, BlockQ5_0, BlockQ6K, BlockQ8_0, GgmlType};

fn byte_array_array_u32<W: Write>(f: &mut W, name: &str, byte_size: usize) -> std::fmt::Result {
    assert!(byte_size % 4 == 0);
    let size = byte_size / 4;
    writeln!(f, "    {}: array<u32, {}>,", name, size)
}

pub(crate) fn write_q4_0_type<W: Write>(f: &mut W) -> std::fmt::Result {
    let q4_0 = GgmlType::Q4_0;
    writeln!(f, "struct {q4_0} {{")?;
    writeln!(f, "    scale: f16,")?;
    byte_array_array_u32(f, "data", BlockQ4_0::WEIGHTS_SIZE)?;
    writeln!(f, "}};")?;

    Ok(())
}


pub(crate) fn write_q5_0_type<W: Write>(f: &mut W) -> std::fmt::Result {
    let q5_0 = GgmlType::Q5_0;
    writeln!(f, "struct {q5_0} {{")?;
        writeln!(f, "    scale: f16,")?;
        byte_array_array_u32(f, "data_high_bits", BlockQ5_0::WEIGHTS_HIGH_BITS_SIZE)?;
        byte_array_array_u32(f, "data_low_bits", BlockQ5_0::WEIGHTS_LOW_BITS_SIZE)?;
        writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q8_0_type<W: Write>(f: &mut W) -> std::fmt::Result {
    let q8_0 = GgmlType::Q8_0;
            writeln!(f, "struct {q8_0} {{")?;
            writeln!(f, "    scale: f16,")?;
            byte_array_array_u32(f, "data", BlockQ8_0::WEIGHTS_SIZE)?;
            writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q4_k_type<W: Write>(f: &mut W) -> std::fmt::Result {
let q4_k = GgmlType::Q4K;
            writeln!(f, "struct {q4_k} {{")?;
            writeln!(f, "    scale: f16,")?;
            writeln!(f, "    min: f16,")?;
            byte_array_array_u32(f, "scales", BlockQ4K::SCALES_SIZE)?;
            byte_array_array_u32(f, "data", BlockQ4K::WEIGHTS_SIZE)?;
            writeln!(f, "}};")?;

            Ok(())
}

pub(crate) fn write_q6_k_type<W: Write>(f: &mut W) -> std::fmt::Result {
let q6_k = GgmlType::Q6K;
            writeln!(f, "struct {q6_k} {{")?;
            byte_array_array_u32(f, "data_low_bits", BlockQ6K::WEIGHTS_LOW_BITS_SIZE)?;
            byte_array_array_u32(f, "data_high_bits", BlockQ6K::WEIGHTS_HIGH_BITS_SIZE)?;
            byte_array_array_u32(f, "scales", BlockQ6K::SCALES_SIZE)?;
            writeln!(f, "    scale: f16,")?;
            writeln!(f, "}};")?;

            Ok(())
}