use crate::{
    DataTypeEnum, Device,
    quantized_types_wgsl::{
        write_q4_0_type, write_q4_k_type, write_q5_0_type, write_q6_k_type, write_q8_0_type,
    },
};
use fusor_gguf::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock, GgufMetadata,
    GgufReadError, GgufTensorMetadata,
};
use std::{
    fmt::{Display, Write},
    sync::Arc,
};
use wgpu::util::DeviceExt;

pub(crate) mod dequantize;
pub(crate) mod matmul;

#[derive(Clone, Debug)]
pub struct QMatrix {
    device: Device,
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType,
}

impl PartialEq for QMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.datatype == other.datatype && self.buffer == other.buffer
    }
}

impl QMatrix {
    pub fn read_from_file<R: std::io::Read + std::io::Seek>(
        device: &Device,
        metadata: &GgufMetadata,
        reader: &mut R,
        key: &str,
    ) -> Result<Option<Self>, GgufReadError> {
        Ok(match metadata.tensor_infos.get(key) {
            Some(rope_freq_weight) => {
                let rope_freq_weight = QMatrix::read(
                    device,
                    rope_freq_weight,
                    reader,
                    metadata.tensor_data_offset,
                )?;
                Some(rope_freq_weight)
            }
            None => None,
        })
    }

    pub fn read<R: std::io::Read + std::io::Seek>(
        device: &Device,
        metadata: &GgufTensorMetadata,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Self, GgufReadError> {
        let bytes = metadata.read_tensor_bytes(reader, tensor_data_offset)?;
        let shape = metadata.shape.iter().map(|x| *x as usize).collect();
        let ty = metadata.ty;
        QMatrix::from_parts(device, &bytes, shape, ty)
    }

    pub(crate) fn from_parts(
        device: &Device,
        bytes: &[u8],
        shape: Box<[usize]>,
        ty: GgmlType,
    ) -> Result<Self, GgufReadError> {
        let bytes: Box<[u8]> = match ty {
            GgmlType::Q4_0 => bytemuck::cast_slice::<_, BlockQ4_0>(bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q5_0 => bytemuck::cast_slice::<_, BlockQ5_0>(bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q8_0 => bytemuck::cast_slice::<_, BlockQ8_0>(bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q4K => bytemuck::cast_slice::<_, BlockQ4K>(bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::Q6K => bytemuck::cast_slice::<_, BlockQ6K>(bytes)
                .iter()
                .flat_map(|block| block.into_wgsl_bytes())
                .collect(),
            GgmlType::F16 | GgmlType::F32 => bytes.into(),
            _ => todo!(),
        };
        let buffer = Arc::new(device.wgpu_device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("QMatrix Buffer"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        Ok(QMatrix {
            device: device.clone(),
            shape,
            buffer,
            datatype: ty,
        })
    }

    pub(crate) fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn datatype(&self) -> GgmlType {
        self.datatype
    }
}

pub(crate) fn dequantize_block<W: Write>(
    kernel: &mut W,
    ty: GgmlType,
    chunk: String,
    datatype: DataTypeEnum,
    mut process_element: impl FnMut(String, String, &mut W),
) {
    match ty {
        GgmlType::Q4_0 => BlockQ4_0::dequantize_block(chunk, datatype, process_element, kernel),
        GgmlType::Q5_0 => BlockQ5_0::dequantize_block(chunk, datatype, process_element, kernel),
        GgmlType::Q8_0 => BlockQ8_0::dequantize_block(chunk, datatype, process_element, kernel),
        GgmlType::Q4K => BlockQ4K::dequantize_block(chunk, datatype, process_element, kernel),
        GgmlType::Q6K => BlockQ6K::dequantize_block(chunk, datatype, process_element, kernel),
        GgmlType::F16 | GgmlType::F32 => {
            process_element("0".to_string(), format!("{datatype}({chunk})"), kernel);
        }
        _ => todo!(),
    }
}

#[allow(unused)]
pub(crate) fn unrolled_dequantize_block<W: Write>(
    kernel: &mut W,
    ty: GgmlType,
    chunk: String,
    datatype: DataTypeEnum,
    mut process_element: impl FnMut(u32, String, &mut W),
) {
    match ty {
        GgmlType::Q4_0 => {
            BlockQ4_0::unrolled_dequantize_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q5_0 => {
            BlockQ5_0::unrolled_dequantize_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q8_0 => {
            BlockQ8_0::unrolled_dequantize_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q4K => {
            BlockQ4K::unrolled_dequantize_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q6K => {
            BlockQ6K::unrolled_dequantize_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::F16 | GgmlType::F32 => {
            process_element(0, format!("{datatype}({chunk})"), kernel);
        }
        _ => todo!(),
    }
}

pub(crate) fn dequantize_vec4_block<W: Write>(
    kernel: &mut W,
    ty: GgmlType,
    chunk: String,
    datatype: DataTypeEnum,
    process_element: impl FnMut(String, String, &mut W),
) {
    match ty {
        GgmlType::Q4_0 => {
            BlockQ4_0::dequantize_vec4_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q5_0 => {
            BlockQ5_0::dequantize_vec4_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q8_0 => {
            BlockQ8_0::dequantize_vec4_block(chunk, datatype, process_element, kernel)
        }
        GgmlType::Q4K => BlockQ4K::dequantize_vec4_block(chunk, datatype, process_element, kernel),
        GgmlType::Q6K => BlockQ6K::dequantize_vec4_block(chunk, datatype, process_element, kernel),
        _ => todo!(),
    }
}

pub(crate) fn dequantize_mat4x4_block<W: Write>(
    kernel: &mut W,
    ty: GgmlType,
    index: &str,
    chunk: String,
    datatype: DataTypeEnum,
    process_element: impl FnOnce(String, &mut W),
) -> bool {
    match ty {
        GgmlType::Q4_0 => {
            BlockQ4_0::dequantize_4x4_block(index, chunk, datatype, process_element, kernel);
            true
        }
        GgmlType::Q5_0 => {
            BlockQ5_0::dequantize_4x4_block(index, chunk, datatype, process_element, kernel);
            true
        }
        GgmlType::Q8_0 => {
            BlockQ8_0::dequantize_4x4_block(index, chunk, datatype, process_element, kernel);
            true
        }
        GgmlType::Q4K => {
            BlockQ4K::dequantize_4x4_block(index, chunk, datatype, process_element, kernel);
            true
        }
        GgmlType::Q6K => {
            BlockQ6K::dequantize_4x4_block(index, chunk, datatype, process_element, kernel);
            true
        }
        _ => false,
    }
}

pub(crate) fn dequantize_mat4x4_block_count(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_0 => BlockQ4_0::MATRIX_BLOCKS,
        GgmlType::Q5_0 => BlockQ5_0::MATRIX_BLOCKS,
        GgmlType::Q8_0 => BlockQ8_0::MATRIX_BLOCKS,
        GgmlType::Q4K => BlockQ4K::MATRIX_BLOCKS,
        GgmlType::Q6K => BlockQ6K::MATRIX_BLOCKS,
        _ => 0,
    }
}

trait WgslQuantizedType: GgufBlock {
    const GGML_TYPE: GgmlType;

    fn dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    );

    fn unrolled_dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnMut(u32, String, &mut W),
        code: &mut W,
    );

    fn dequantize_vec4_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        Self::unrolled_dequantize_block(
            chunk,
            datatype,
            |index, data, code| {
                writeln!(code, "let dequantized_{index} = {data};").unwrap();
            },
            code,
        );

        let chunk_blocks = Self::GGML_TYPE.block_size() / 4;

        // Pack the individual dequantized values into vectors
        for i in 0..chunk_blocks {
            write!(code, "let dequantized_vec_{i} = vec4<{datatype}>(").unwrap();
            for j in 0..4 {
                if j > 0 {
                    write!(code, ", ").unwrap();
                }
                let index = i * 4 + j;
                write!(code, "dequantized_{index}").unwrap();
            }
            writeln!(code, ");").unwrap();

            process_element(i.to_string(), format!("dequantized_vec_{i}"), code);
        }
    }

    const MATRIX_BLOCKS: usize = 0;

    fn dequantize_4x4_block<W: Write>(
        _index: &str,
        _chunk: String,
        _datatype: DataTypeEnum,
        _process_element: impl FnOnce(String, &mut W),
        _code: &mut W,
    );

    // This is used in the fuzzing test
    #[allow(unused)]
    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result;
}

const fn center_of_bit_space(bits: u8) -> u8 {
    1 << (bits - 1)
}

pub(crate) const fn shift_right_scale(shift_bits: u8) -> f32 {
    1.0 / (1 << shift_bits) as f32
}

impl WgslQuantizedType for BlockQ4_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q4_0;

    const MATRIX_BLOCKS: usize = 2;

    fn dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        const CENTER: u8 = center_of_bit_space(4);
        const RIGHT_SHIFT_4_SCALE: f32 = shift_right_scale(4);

        let half_block_size = BlockQ4_0::BLOCK_SIZE as u8 / 2;
        let weights_size_u32 = BlockQ4_0::WEIGHTS_SIZE as u8 / 4;
        writeln!(
            code,
            "const SHIFT_SCALES = vec2({datatype}(1.0), {datatype}({RIGHT_SHIFT_4_SCALE}));"
        )
        .unwrap();
        writeln!(code, "const CENTER = vec2({datatype}({CENTER}));").unwrap();

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "var output_index = 0u;").unwrap();
        writeln!(code, "for (var i = 0u; i < {weights_size_u32}; i++) {{").unwrap();
        writeln!(code, "let weight_chunk = {chunk}.data[i];").unwrap();
        writeln!(code, "let weight_chunk_bytes = unpack4xU8(weight_chunk);").unwrap();
        for offset in 0..4 {
            writeln!(code, "let byte{offset} = weight_chunk_bytes[{offset}];").unwrap();
            writeln!(
                code,
                "let data{offset} = vec2(byte{offset} & 0x0F, byte{offset} & 0xF0);"
            )
            .unwrap();
            writeln!(
            code,
            "let data_float{offset} = ((vec2<{datatype}>(data{offset}) * SHIFT_SCALES) - CENTER) * scale;"
        )
        .unwrap();
            process_element(
                "output_index".to_string(),
                format!("data_float{offset}.x"),
                code,
            );
            process_element(
                format!("output_index + {half_block_size}"),
                format!("data_float{offset}.y"),
                code,
            );
            writeln!(code, "output_index += 1;").unwrap();
        }
        writeln!(code, "}}").unwrap();
    }

    fn unrolled_dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(u32, String, &mut W),
        code: &mut W,
    ) {
        const CENTER: u8 = center_of_bit_space(4);
        const RIGHT_SHIFT_4_SCALE: f32 = shift_right_scale(4);

        let half_block_size = BlockQ4_0::BLOCK_SIZE as u8 / 2;
        let weights_size_u32 = BlockQ4_0::WEIGHTS_SIZE as u8 / 4;
        writeln!(
            code,
            "const SHIFT_SCALES = vec2({datatype}(1.0), {datatype}({RIGHT_SHIFT_4_SCALE}));"
        )
        .unwrap();
        writeln!(code, "const CENTER = vec2({datatype}({CENTER}));").unwrap();

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        let mut output_index = 0;
        for i in 0..weights_size_u32 {
            writeln!(
                code,
                "let weight_chunk_bytes_{i} = unpack4xU8({chunk}.data[{i}]);"
            )
            .unwrap();
            for offset in 0..4 {
                writeln!(
                    code,
                    "let byte_{offset}_{i} = weight_chunk_bytes_{i}[{offset}];"
                )
                .unwrap();
                writeln!(
                    code,
                    "let data_{offset}_{i} = vec2(byte_{offset}_{i} & 0x0F, byte_{offset}_{i} & 0xF0);"
                )
                .unwrap();
                writeln!(
                    code,
                    "let data_float_{offset}_{i} = ((vec2<{datatype}>(data_{offset}_{i}) * SHIFT_SCALES) - CENTER) * scale;"
                )
                .unwrap();
                process_element(output_index, format!("data_float_{offset}_{i}.x"), code);
                process_element(
                    output_index + half_block_size as u32,
                    format!("data_float_{offset}_{i}.y"),
                    code,
                );
                output_index += 1;
            }
        }
    }

    fn dequantize_4x4_block<W: Write>(
        index: &str,
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnOnce(String, &mut W),
        code: &mut W,
    ) {
        const CENTER: u8 = center_of_bit_space(4);
        const RIGHT_SHIFT_4_SCALE: f32 = shift_right_scale(4);

        writeln!(code, "const CENTER = {datatype}({CENTER});").unwrap();

        writeln!(
            code,
            "let shift_scale = select({datatype}(1.0), {datatype}({RIGHT_SHIFT_4_SCALE}), {index} == 1u);"
        )
        .unwrap();
        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(
            code,
            "let mask = select(0x0F0F0F0F, 0xF0F0F0F0u, {index} == 1u);"
        )
        .unwrap();

        // Load 16 bytes (4 u32s) which contain our 16 elements
        // Group the results into a mat4x4
        write!(code, "let result = mat4x4<{datatype}>(").unwrap();
        for i in 0..4 {
            if i > 0 {
                writeln!(code, ", ").unwrap();
            }
            write!(
                code,
                "(vec4<{datatype}>(unpack4xU8({chunk}.data[{i}] & mask)) * shift_scale - CENTER) * scale"
            )
            .unwrap();
        }
        writeln!(code, ");").unwrap();

        process_element("result".to_string(), code);
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q4_0_type(f)
    }
}

impl WgslQuantizedType for BlockQ5_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q5_0;

    const MATRIX_BLOCKS: usize = 2;

    fn dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        const CENTER: u8 = center_of_bit_space(5);
        const FIFTH_BIT: u8 = 0x10;

        let half_block_size = BlockQ5_0::BLOCK_SIZE as u8 / 2;
        let low_weights_size_u32 = BlockQ5_0::WEIGHTS_LOW_BITS_SIZE as u8 / 4;
        writeln!(code, "const CENTER = vec2({datatype}({CENTER}));").unwrap();

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "var output_index = 0u;").unwrap();
        writeln!(code, "let high_bits = {chunk}.data_high_bits[0];").unwrap();
        writeln!(code, "for (var i = 0u; i < {low_weights_size_u32}; i++) {{").unwrap();
        writeln!(code, "let low_weight_chunk = {chunk}.data_low_bits[i];").unwrap();
        writeln!(
            code,
            "let low_weight_chunk_bytes = unpack4xU8(low_weight_chunk);"
        )
        .unwrap();
        for offset in 0..4 {
            writeln!(code, "let byte{offset} = low_weight_chunk_bytes[{offset}];").unwrap();
            writeln!(
                code,
                "let data{offset} = vec2((byte{offset} & 0x0F) | (((high_bits >> (i*4 + {offset})) << 4) & {FIFTH_BIT}), (byte{offset} >> 4) | ((high_bits >> (i*4 + {offset} + 12)) & {FIFTH_BIT}));"
            )
            .unwrap();
            writeln!(
                code,
                "let data_float{offset} = (vec2<{datatype}>(data{offset}) - CENTER) * scale;"
            )
            .unwrap();
            process_element(
                "output_index".to_string(),
                format!("data_float{offset}.x"),
                code,
            );
            process_element(
                format!("output_index + {half_block_size}"),
                format!("data_float{offset}.y"),
                code,
            );
            writeln!(code, "output_index += 1;").unwrap();
        }
        writeln!(code, "}}").unwrap();
    }

    fn unrolled_dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(u32, String, &mut W),
        code: &mut W,
    ) {
        const CENTER: u8 = center_of_bit_space(5);
        const FIFTH_BIT: u8 = 0x10;

        let half_block_size = BlockQ5_0::BLOCK_SIZE as u8 / 2;
        let low_weights_size_u32 = BlockQ5_0::WEIGHTS_LOW_BITS_SIZE as u8 / 4;
        writeln!(code, "const CENTER = vec2({datatype}({CENTER}));").unwrap();

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "let high_bits = {chunk}.data_high_bits[0];").unwrap();
        let mut output_index = 0;
        for i in 0..low_weights_size_u32 {
            writeln!(
                code,
                "let low_weight_chunk_{i} = {chunk}.data_low_bits[{i}];"
            )
            .unwrap();
            writeln!(
                code,
                "let low_weight_chunk_bytes_{i} = unpack4xU8(low_weight_chunk_{i});"
            )
            .unwrap();
            for offset in 0..4 {
                writeln!(
                    code,
                    "let byte_{offset}_{i} = low_weight_chunk_bytes_{i}[{offset}];"
                )
                .unwrap();
                writeln!(
                code,
                "let data_{offset}_{i} = vec2((byte_{offset}_{i} & 0x0F) | (((high_bits >> ({i}*4 + {offset})) << 4) & {FIFTH_BIT}), (byte_{offset}_{i} >> 4) | ((high_bits >> ({i}*4 + {offset} + 12)) & {FIFTH_BIT}));"
            )
            .unwrap();
                writeln!(
                code,
                "let data_float_{offset}_{i} = (vec2<{datatype}>(data_{offset}_{i}) - CENTER) * scale;"
            )
            .unwrap();
                process_element(output_index, format!("data_float_{offset}_{i}.x"), code);
                process_element(
                    output_index + half_block_size as u32,
                    format!("data_float_{offset}_{i}.y"),
                    code,
                );
                output_index += 1;
            }
        }
    }

    fn dequantize_4x4_block<W: Write>(
        index: &str,
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnOnce(String, &mut W),
        code: &mut W,
    ) {
        const CENTER: u8 = center_of_bit_space(5);
        const FIFTH_BIT: u8 = 0x10;

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "const CENTER = {datatype}({CENTER});").unwrap();
        writeln!(code, "let high_bits = {chunk}.data_high_bits[0];").unwrap();

        // index = 0 means low nibbles (first 16 elements)
        // index = 1 means high nibbles (second 16 elements)
        writeln!(code, "let is_high_half = {index} == 1u;").unwrap();

        // Load 16 bytes (4 u32s) which contain our 16 elements
        for i in 0..4 {
            writeln!(code, "let low_data_{i} = ").unwrap();
            for j in 0..4 {
                if j > 0 {
                    write!(code, " | ").unwrap();
                }
                let bit_index = i * 4 + j;
                write!(
                    code,
                    "(((high_bits >> {bit_index}) << 4) & {FIFTH_BIT}) << {}",
                    j * 8
                )
                .unwrap();
            }
            writeln!(code, ";").unwrap();
            writeln!(code, "let high_data_{i} = ").unwrap();
            for j in 0..4 {
                if j > 0 {
                    write!(code, " | ").unwrap();
                }
                let bit_index = i * 4 + j;
                write!(
                    code,
                    "((high_bits >> ({bit_index} + 12)) & {FIFTH_BIT}) << {}",
                    j * 8
                )
                .unwrap();
            }
            writeln!(code, ";").unwrap();
            const FIRST_4_MASK: u32 = 0x0F0F_0F0F;
            const LAST_4_MASK: u32 = 0xF0F0_F0F0;
            writeln!(code, "let raw_low_bits_{i} = {chunk}.data_low_bits[{i}];").unwrap();
            writeln!(code, "let low_bits_{i} = select(raw_low_bits_{i} & {FIRST_4_MASK}, (raw_low_bits_{i} & {LAST_4_MASK}) >> 4, is_high_half);").unwrap();
            writeln!(
                code,
                "let high_bits_{i} = select(low_data_{i}, high_data_{i}, is_high_half);"
            )
            .unwrap();
            writeln!(code, "let bits_{i} = low_bits_{i} | high_bits_{i};").unwrap();
        }

        // Group the results into a mat4x4
        write!(code, "let result = mat4x4<{datatype}>(").unwrap();
        for i in 0..4 {
            if i > 0 {
                writeln!(code, ", ").unwrap();
            }
            writeln!(
                code,
                "(vec4<{datatype}>(unpack4xU8(bits_{i})) - CENTER) * scale"
            )
            .unwrap();
        }
        writeln!(code, ");").unwrap();

        process_element("result".to_string(), code);
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q5_0_type(f)
    }
}

impl WgslQuantizedType for BlockQ8_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q8_0;

    const MATRIX_BLOCKS: usize = 2;

    fn dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        let weights_size_u32 = BlockQ8_0::WEIGHTS_SIZE as u8 / 4;

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "var output_index = 0u;").unwrap();
        writeln!(code, "for (var i = 0u; i < {weights_size_u32}; i++) {{").unwrap();
        writeln!(code, "let weight_chunk = {chunk}.data[i];").unwrap();
        writeln!(code, "let weight_chunk_bytes = unpack4xI8(weight_chunk);").unwrap();
        for offset in 0..4 {
            writeln!(code, "let data{offset} = weight_chunk_bytes[{offset}];").unwrap();
            writeln!(
                code,
                "let data_float{offset} = {datatype}(data{offset}) * scale;"
            )
            .unwrap();
            process_element(
                "output_index".to_string(),
                format!("data_float{offset}"),
                code,
            );
            writeln!(code, "output_index += 1;").unwrap();
        }
        writeln!(code, "}}").unwrap();
    }

    fn unrolled_dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(u32, String, &mut W),
        code: &mut W,
    ) {
        let weights_size_u32 = BlockQ8_0::WEIGHTS_SIZE as u8 / 4;

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        let mut output_index = 0;
        for i in 0..weights_size_u32 {
            writeln!(code, "let weight_chunk_{i} = {chunk}.data[{i}];").unwrap();
            writeln!(
                code,
                "let weight_chunk_bytes_{i} = unpack4xI8(weight_chunk_{i});"
            )
            .unwrap();
            for offset in 0..4 {
                writeln!(
                    code,
                    "let data_{offset}_{i} = weight_chunk_bytes_{i}[{offset}];"
                )
                .unwrap();
                writeln!(
                    code,
                    "let data_float_{offset}_{i} = {datatype}(data_{offset}_{i}) * scale;"
                )
                .unwrap();
                process_element(output_index, format!("data_float_{offset}_{i}"), code);
                output_index += 1;
            }
        }
    }

    fn dequantize_vec4_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        let weights_size_u32 = BlockQ8_0::WEIGHTS_SIZE as u8 / 4;

        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();
        for i in 0..weights_size_u32 {
            writeln!(
            code,
            "let weight_chunk_bytes_{i} = vec4<{datatype}>(unpack4xI8({chunk}.data[{i}])) * scale;"
        )
        .unwrap();
            process_element(i.to_string(), format!("weight_chunk_bytes_{i}"), code);
        }
    }

    fn dequantize_4x4_block<W: Write>(
        index: &str,
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnOnce(String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();

        // index = 0 means first 16 elements (u32s 0-3)
        // index = 1 means second 16 elements (u32s 4-7)
        writeln!(code, "let base_index = {index} * 4u;").unwrap();

        // Load 16 bytes (4 u32s) which contain our 16 elements
        // Group the results into a mat4x4
        write!(code, "let result = mat4x4<{datatype}>(").unwrap();
        for i in 0..4 {
            if i > 0 {
                writeln!(code, ", ").unwrap();
            }
            write!(
                code,
                "vec4<{datatype}>(unpack4xI8({chunk}.data[base_index + {i}u])) * scale"
            )
            .unwrap();
        }
        writeln!(code, ");").unwrap();

        process_element("result".to_string(), code);
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q8_0_type(f)
    }
}

const SIX_BITS_MASK: u32 = 0b0011_1111_0011_1111_0011_1111_0011_1111;
const MSB_TWO_BITS_MASK: u32 = 0b1100_0000_1100_0000_1100_0000_1100_0000;
const LOW_FOUR_BITS: u32 = 0b0000_1111_0000_1111_0000_1111_0000_1111;
const HIGH_FOUR_BITS: u32 = 0b1111_0000_1111_0000_1111_0000_1111_0000;
const MSB_SCALES_MASK: u32 = LOW_FOUR_BITS;
const MSB_OFFSET_MASK: u32 = HIGH_FOUR_BITS;

impl WgslQuantizedType for BlockQ4K {
    const GGML_TYPE: GgmlType = GgmlType::Q4K;

    const MATRIX_BLOCKS: usize = 16;

    fn dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let super_block_scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "let super_block_min = {datatype}({chunk}.min);").unwrap();

        writeln!(code, "let first_four_bytes = {chunk}.scales[0];").unwrap();
        writeln!(code, "let middle_four_bytes = {chunk}.scales[1];").unwrap();
        writeln!(code, "let last_four_bytes = {chunk}.scales[2];").unwrap();

        // Extracts this bit pattern. The first 6 bits of the first
        // 4 bytes are the scales. The first 6 bits of the second 4
        // bytes are the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // __000000|__111111|__222222|__333333|__000000|__111111
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // __222222|__333333|________|________|________|________
        writeln!(
            code,
            "let first_scales = first_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();
        writeln!(
            code,
            "let first_offsets = middle_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();

        // Extracts this bit pattern. The first 2 bits of the first
        // 4 bytes are the scales. The first 2 bits of the second 4
        // bytes are the offsets.
        // The first 4 bits of the last 4 bytes are the lower 4 bits
        // of the scales. The second 4 bits of the last 4 bytes are
        // the lower 4 bits of the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // 44______|55______|66______|77______|44______|55______
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // 66______|77______|44444444|55555555|66666666|77777777
        writeln!(
            code,
            "let msb_scales = (first_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
        )
        .unwrap();
        writeln!(
            code,
            "let lsb_scales = last_four_bytes & {MSB_SCALES_MASK};"
        )
        .unwrap();
        writeln!(code, "let second_scales = msb_scales | lsb_scales;").unwrap();
        writeln!(
            code,
            "let msb_offsets = (middle_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
        )
        .unwrap();
        writeln!(
            code,
            "let lsb_offsets = (last_four_bytes & {MSB_OFFSET_MASK}) >> 4;"
        )
        .unwrap();
        writeln!(code, "let second_offsets = msb_offsets | lsb_offsets;").unwrap();

        writeln!(code, "var weight_index = 0u;").unwrap();
        writeln!(code, "var output_index = 0u;").unwrap();

        let mut run_chunk = |scales: &str, offsets: &str| {
            writeln!(code, "{{").unwrap();
            writeln!(
                code,
                "let scales = vec4<{datatype}>(unpack4xU8({scales})) * super_block_scale;"
            )
            .unwrap();
            writeln!(code, "let low_scales = scales.xz;").unwrap();
            writeln!(
                code,
                "let high_scales = scales.yw * {};",
                shift_right_scale(4)
            )
            .unwrap();
            writeln!(
                code,
                "let offsets = vec4<{datatype}>(unpack4xU8({offsets})) * super_block_min;"
            )
            .unwrap();
            writeln!(code, "let low_offsets = offsets.xz;").unwrap();
            writeln!(code, "let high_offsets = offsets.yw;").unwrap();

            for suffix in ["x", "y"] {
                writeln!(code, "for (var i = 0u; i < 32u; i += 4) {{").unwrap();
                writeln!(code, "let weight_chunk = {chunk}.data[weight_index];").unwrap();
                writeln!(code, "let weight_chunk_low = vec4<{datatype}>(unpack4xU8(weight_chunk & {LOW_FOUR_BITS}));").unwrap();
                writeln!(code, "let weight_chunk_high = vec4<{datatype}>(unpack4xU8(weight_chunk & {HIGH_FOUR_BITS}));").unwrap();
                writeln!(code, "let low_result = weight_chunk_low * low_scales.{suffix} - low_offsets.{suffix};").unwrap();
                writeln!(code, "let high_result = weight_chunk_high * high_scales.{suffix} - high_offsets.{suffix};").unwrap();
                for i in 0..4 {
                    process_element(
                        format!("output_index + i + {i}u"),
                        format!("low_result[{i}]"),
                        code,
                    );
                    process_element(
                        format!("output_index + i + {i}u + 32u"),
                        format!("high_result[{i}]"),
                        code,
                    );
                }
                writeln!(code, "weight_index += 1;").unwrap();
                writeln!(code, "}}").unwrap();
                writeln!(code, "output_index += 64;").unwrap();
            }
            writeln!(code, "}}").unwrap();
        };

        run_chunk("first_scales", "first_offsets");
        run_chunk("second_scales", "second_offsets");
    }

    fn unrolled_dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(u32, String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let super_block_scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "let super_block_min = {datatype}({chunk}.min);").unwrap();

        writeln!(code, "let first_four_bytes = {chunk}.scales[0];").unwrap();
        writeln!(code, "let middle_four_bytes = {chunk}.scales[1];").unwrap();
        writeln!(code, "let last_four_bytes = {chunk}.scales[2];").unwrap();

        // Extracts this bit pattern. The first 6 bits of the first
        // 4 bytes are the scales. The first 6 bits of the second 4
        // bytes are the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // __000000|__111111|__222222|__333333|__000000|__111111
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // __222222|__333333|________|________|________|________
        writeln!(
            code,
            "let first_scales = first_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();
        writeln!(
            code,
            "let first_offsets = middle_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();

        // Extracts this bit pattern. The first 2 bits of the first
        // 4 bytes are the scales. The first 2 bits of the second 4
        // bytes are the offsets.
        // The first 4 bits of the last 4 bytes are the lower 4 bits
        // of the scales. The second 4 bits of the last 4 bytes are
        // the lower 4 bits of the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // 44______|55______|66______|77______|44______|55______
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // 66______|77______|44444444|55555555|66666666|77777777
        writeln!(
            code,
            "let msb_scales = (first_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
        )
        .unwrap();
        writeln!(
            code,
            "let lsb_scales = last_four_bytes & {MSB_SCALES_MASK};"
        )
        .unwrap();
        writeln!(code, "let second_scales = msb_scales | lsb_scales;").unwrap();
        writeln!(
            code,
            "let msb_offsets = (middle_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
        )
        .unwrap();
        writeln!(
            code,
            "let lsb_offsets = (last_four_bytes & {MSB_OFFSET_MASK}) >> 4;"
        )
        .unwrap();
        writeln!(code, "let second_offsets = msb_offsets | lsb_offsets;").unwrap();

        let mut weight_index = 0;
        let mut output_index = 0;

        let mut run_chunk = |scales: &str,
                             offsets: &str,
                             suffix: &str,
                             output_index: &mut u32,
                             weight_index: &mut u32| {
            writeln!(
                code,
                "let scales_{suffix} = vec4<{datatype}>(unpack4xU8({scales})) * super_block_scale;"
            )
            .unwrap();
            writeln!(code, "let low_scales_{suffix} = scales_{suffix}.xz;").unwrap();
            writeln!(
                code,
                "let high_scales_{suffix} = scales_{suffix}.yw * {};",
                shift_right_scale(4)
            )
            .unwrap();
            writeln!(
                code,
                "let offsets_{suffix} = vec4<{datatype}>(unpack4xU8({offsets})) * super_block_min;"
            )
            .unwrap();
            writeln!(code, "let low_offsets_{suffix} = offsets_{suffix}.xz;").unwrap();
            writeln!(code, "let high_offsets_{suffix} = offsets_{suffix}.yw;").unwrap();

            for local_suffix in ["x", "y"] {
                for index in (0..32).step_by(4) {
                    writeln!(
                        code,
                        "let weight_chunk_{local_suffix}_{suffix}_{index} = {chunk}.data[{weight_index}];"
                    )
                    .unwrap();
                    writeln!(code, "let weight_chunk_low_{local_suffix}_{suffix}_{index} = vec4<{datatype}>(unpack4xU8(weight_chunk_{local_suffix}_{suffix}_{index} & {LOW_FOUR_BITS}));").unwrap();
                    writeln!(code, "let weight_chunk_high_{local_suffix}_{suffix}_{index} = vec4<{datatype}>(unpack4xU8(weight_chunk_{local_suffix}_{suffix}_{index} & {HIGH_FOUR_BITS}));").unwrap();
                    writeln!(code, "let low_result_{local_suffix}_{suffix}_{index} = weight_chunk_low_{local_suffix}_{suffix}_{index} * low_scales_{suffix}.{local_suffix} - low_offsets_{suffix}.{local_suffix};").unwrap();
                    writeln!(code, "let high_result_{local_suffix}_{suffix}_{index} = weight_chunk_high_{local_suffix}_{suffix}_{index} * high_scales_{suffix}.{local_suffix} - high_offsets_{suffix}.{local_suffix};").unwrap();
                    for i in 0..4 {
                        process_element(
                            *output_index + i + index,
                            format!("low_result_{local_suffix}_{suffix}_{index}[{i}]"),
                            code,
                        );
                        process_element(
                            *output_index + i + index + 32,
                            format!("high_result_{local_suffix}_{suffix}_{index}[{i}]"),
                            code,
                        );
                    }
                    *weight_index += 1;
                }
                *output_index += 64;
            }
        };

        run_chunk(
            "first_scales",
            "first_offsets",
            "first",
            &mut output_index,
            &mut weight_index,
        );
        run_chunk(
            "second_scales",
            "second_offsets",
            "second",
            &mut output_index,
            &mut weight_index,
        );
    }

    fn dequantize_vec4_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let super_block_scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "let super_block_min = {datatype}({chunk}.min);").unwrap();

        writeln!(code, "let first_four_bytes = {chunk}.scales[0];").unwrap();
        writeln!(code, "let middle_four_bytes = {chunk}.scales[1];").unwrap();
        writeln!(code, "let last_four_bytes = {chunk}.scales[2];").unwrap();

        // Extracts this bit pattern. The first 6 bits of the first
        // 4 bytes are the scales. The first 6 bits of the second 4
        // bytes are the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // __000000|__111111|__222222|__333333|__000000|__111111
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // __222222|__333333|________|________|________|________
        writeln!(
            code,
            "let first_scales = first_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();
        writeln!(
            code,
            "let first_offsets = middle_four_bytes & {SIX_BITS_MASK};"
        )
        .unwrap();

        // Extracts this bit pattern. The first 2 bits of the first
        // 4 bytes are the scales. The first 2 bits of the second 4
        // bytes are the offsets.
        // The first 4 bits of the last 4 bytes are the lower 4 bits
        // of the scales. The second 4 bits of the last 4 bytes are
        // the lower 4 bits of the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // 44______|55______|66______|77______|44______|55______
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // 66______|77______|44444444|55555555|66666666|77777777
        writeln!(
            code,
            "let msb_scales = (first_four_bytes & {MSB_TWO_BITS_MASK}) >> 2u;"
        )
        .unwrap();
        writeln!(
            code,
            "let lsb_scales = last_four_bytes & {MSB_SCALES_MASK};"
        )
        .unwrap();
        writeln!(code, "let second_scales = msb_scales | lsb_scales;").unwrap();
        writeln!(
            code,
            "let msb_offsets = (middle_four_bytes & {MSB_TWO_BITS_MASK}) >> 2u;"
        )
        .unwrap();
        writeln!(
            code,
            "let lsb_offsets = (last_four_bytes & {MSB_OFFSET_MASK}) >> 4u;"
        )
        .unwrap();
        writeln!(code, "let second_offsets = msb_offsets | lsb_offsets;").unwrap();

        writeln!(code, "var weight_index = 0u;").unwrap();
        writeln!(code, "var output_index = 0u;").unwrap();

        let mut run_chunk = |scales: &str, offsets: &str| {
            writeln!(code, "{{").unwrap();
            writeln!(
                code,
                "let scales = vec4<{datatype}>(unpack4xU8({scales})) * super_block_scale;"
            )
            .unwrap();
            writeln!(code, "let low_scales = scales.xz;").unwrap();
            writeln!(
                code,
                "let high_scales = scales.yw * {};",
                shift_right_scale(4)
            )
            .unwrap();
            writeln!(
                code,
                "let offsets = vec4<{datatype}>(unpack4xU8({offsets})) * super_block_min;"
            )
            .unwrap();
            writeln!(code, "let low_offsets = offsets.xz;").unwrap();
            writeln!(code, "let high_offsets = offsets.yw;").unwrap();

            for suffix in ["x", "y"] {
                writeln!(code, "for (var i = 0u; i < 8u; i += 1u) {{").unwrap();
                writeln!(code, "let weight_chunk = {chunk}.data[weight_index];").unwrap();
                writeln!(code, "let weight_chunk_low = vec4<{datatype}>(unpack4xU8(weight_chunk & {LOW_FOUR_BITS}));").unwrap();
                writeln!(code, "let weight_chunk_high = vec4<{datatype}>(unpack4xU8(weight_chunk & {HIGH_FOUR_BITS}));").unwrap();
                writeln!(code, "let low_result = weight_chunk_low * low_scales.{suffix} - low_offsets.{suffix};").unwrap();
                writeln!(code, "let high_result = weight_chunk_high * high_scales.{suffix} - high_offsets.{suffix};").unwrap();
                process_element(
                    "output_index + i".to_string(),
                    "low_result".to_string(),
                    code,
                );
                process_element(
                    "output_index + i + 8u".to_string(),
                    "high_result".to_string(),
                    code,
                );

                writeln!(code, "weight_index += 1u;").unwrap();
                writeln!(code, "}}").unwrap();
                writeln!(code, "output_index += 16u;").unwrap();
            }
            writeln!(code, "}}").unwrap();
        };

        run_chunk("first_scales", "first_offsets");
        run_chunk("second_scales", "second_offsets");
    }

    fn dequantize_4x4_block<W: Write>(
        index: &str,
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnOnce(String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let super_block_scale = {datatype}({chunk}.scale);").unwrap();
        writeln!(code, "let super_block_min = {datatype}({chunk}.min);").unwrap();

        // Determine which of the 4 64-element groups this index belongs to
        writeln!(code, "let scale_group = {index} / 4u;").unwrap();
        writeln!(code, "let sub_chunk = {index} % 4u;").unwrap();

        // Select the appropriate scales and offsets
        {
            writeln!(code, "let first_four_bytes = {chunk}.scales[0];").unwrap();
            writeln!(code, "let middle_four_bytes = {chunk}.scales[1];").unwrap();
            writeln!(code, "let last_four_bytes = {chunk}.scales[2];").unwrap();

            writeln!(
                code,
                "let first_scales = first_four_bytes & {SIX_BITS_MASK};"
            )
            .unwrap();
            writeln!(
                code,
                "let first_offsets = middle_four_bytes & {SIX_BITS_MASK};"
            )
            .unwrap();

            writeln!(
                code,
                "let msb_scales = (first_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
            )
            .unwrap();
            writeln!(
                code,
                "let lsb_scales = last_four_bytes & {MSB_SCALES_MASK};"
            )
            .unwrap();
            writeln!(code, "let second_scales = msb_scales | lsb_scales;").unwrap();
            writeln!(
                code,
                "let msb_offsets = (middle_four_bytes & {MSB_TWO_BITS_MASK}) >> 2;"
            )
            .unwrap();
            writeln!(
                code,
                "let lsb_offsets = (last_four_bytes & {MSB_OFFSET_MASK}) >> 4;"
            )
            .unwrap();
            writeln!(code, "let second_offsets = msb_offsets | lsb_offsets;").unwrap();
            writeln!(
                code,
                "let scales_u32 = select(first_scales, second_scales, scale_group >= 2u);"
            )
            .unwrap();
            writeln!(
                code,
                "let offsets_u32 = select(first_offsets, second_offsets, scale_group >= 2u);"
            )
            .unwrap();
        }
        writeln!(code, "let scale_component_idx = scale_group % 2u;").unwrap();

        writeln!(
            code,
            "let scales_vec = vec4<{datatype}>(unpack4xU8(scales_u32)) * super_block_scale;"
        )
        .unwrap();
        writeln!(
            code,
            "let offsets_vec = vec4<{datatype}>(unpack4xU8(offsets_u32)) * super_block_min;"
        )
        .unwrap();

        writeln!(
            code,
            "let scale = scales_vec[sub_chunk / 2 + (scale_group % 2)*2] * select(1.0, {}, sub_chunk >= 2u);",
            shift_right_scale(4)
        )
        .unwrap();
        writeln!(
            code,
            "let offset = offsets_vec[sub_chunk / 2 + (scale_group % 2)*2];",
        )
        .unwrap();

        // Calculate the weight base index
        writeln!(
            code,
            "let weight_base = (scale_group * 8u) + ((sub_chunk & 1u) * 4u);"
        )
        .unwrap();

        writeln!(code, "let is_high = sub_chunk >= 2u;").unwrap();

        // Load 16 elements (4 u32s, each with 4 bytes)
        // Group the results into a mat4x4
        write!(code, "let result = mat4x4<{datatype}>(").unwrap();
        for i in 0..4 {
            if i > 0 {
                writeln!(code, ", ").unwrap();
            }
            writeln!(
                code,
                "vec4<{datatype}>(unpack4xU8({chunk}.data[weight_base + {i}u] & select({LOW_FOUR_BITS}u, {HIGH_FOUR_BITS}u, is_high))) * scale - offset"
            )
            .unwrap();
        }
        writeln!(code, ");").unwrap();

        process_element("result".to_string(), code);
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q4_k_type(f)
    }
}

const CENTER_SIX_BIT: i8 = center_of_bit_space(6) as i8;
const FIRST_TWO_BITS: u8 = 0b11000000;
const SECOND_TWO_BITS: u8 = 0b00110000;
const THIRD_TWO_BITS: u8 = 0b00001100;
const FOURTH_TWO_BITS: u8 = 0b00000011;

const FIRST_HALF_BITS: u8 = 0b11110000;
const SECOND_HALF_BITS: u8 = 0b00001111;

fn index_signed_bytes(u32_array: impl Display, byte_index: impl Display) -> String {
    format!("unpack4xI8({u32_array}[({byte_index}) / 4u])[({byte_index}) % 4]")
}

impl WgslQuantizedType for BlockQ6K {
    const GGML_TYPE: GgmlType = GgmlType::Q6K;

    fn dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();

        writeln!(
            code,
            "for (var raw_chunk_index = 0u; raw_chunk_index < 16u; raw_chunk_index += 1u) {{",
        )
        .unwrap();
        {
            writeln!(code, "let low_index = (raw_chunk_index / 8u) * 16u + ((raw_chunk_index / 2u) & 1u) * 8u + (raw_chunk_index & 1u) * 4u;").unwrap();
            writeln!(
                code,
                "let high_index = (raw_chunk_index / 8u) * 8u + (raw_chunk_index & 1u) * 4u;"
            )
            .unwrap();
            writeln!(
                code,
                "let scale_index = (raw_chunk_index % 2u) + (raw_chunk_index / 2u) * 2u;"
            )
            .unwrap();
            writeln!(
                code,
                "let chunk_index = (raw_chunk_index / 2u) & {FOURTH_TWO_BITS}u;"
            )
            .unwrap();

            writeln!(
                code,
                "let chunk_raw_scale = {datatype}({});",
                index_signed_bytes(format!("{chunk}.scales"), "scale_index")
            )
            .unwrap();
            writeln!(code, "let high_mask = select(select({FOURTH_TWO_BITS}u, {THIRD_TWO_BITS}u, chunk_index > 0u), select({SECOND_TWO_BITS}u, {FIRST_TWO_BITS}u, chunk_index > 2u), chunk_index > 1u);").unwrap();
            writeln!(
                code,
                "let low_mask = select({SECOND_HALF_BITS}u, {FIRST_HALF_BITS}u, chunk_index > 1u);"
            )
            .unwrap();
            let shift_4 = shift_right_scale(4);
            writeln!(code, "let coefficient = select({datatype}(1.0), {datatype}({shift_4}), chunk_index > 1u);").unwrap();

            writeln!(
                code,
                "let chunk_midpoint = scale * chunk_raw_scale * {datatype}({CENTER_SIX_BIT});"
            )
            .unwrap();
            writeln!(
                code,
                "let chunk_scale = scale * chunk_raw_scale * coefficient;"
            )
            .unwrap();

            writeln!(
                code,
                "for (var vec4_index = 0u; vec4_index < 4u; vec4_index += 1u) {{"
            )
            .unwrap();
            {
                writeln!(
                    code,
                    "let low_chunk = unpack4xU8({chunk}.data_low_bits[low_index + vec4_index]);",
                )
                .unwrap();
                writeln!(
                    code,
                    "let high_chunk = unpack4xU8({chunk}.data_high_bits[high_index + vec4_index]);",
                )
                .unwrap();

                for offset in 0..4 {
                    writeln!(
                        code,
                        "let low_byte_{offset} = low_chunk[{offset}] & low_mask;",
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "let high_byte_{offset} = high_chunk[{offset}] & high_mask;",
                    )
                    .unwrap();

                    writeln!(
                        code,
                        "let merged_{offset} = {datatype}(low_byte_{offset} | select(high_byte_{offset} << 4, high_byte_{offset} << 2, (chunk_index & 1u) == 1u));",
                    )
                    .unwrap();

                    writeln!(
                        code,
                        "let scaled_{offset} = {datatype}(chunk_scale * merged_{offset} - chunk_midpoint);"
                    )
                    .unwrap();

                    process_element(
                        format!("raw_chunk_index * 16u + vec4_index * 4u + {offset}"),
                        format!("scaled_{offset}"),
                        code,
                    );
                }
            }
            writeln!(code, "}}").unwrap();
        }
        writeln!(code, "}}").unwrap();
    }

    fn unrolled_dequantize_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(u32, String, &mut W),
        code: &mut W,
    ) {
        writeln!(code, "let scale = {datatype}({chunk}.scale);").unwrap();

        for raw_chunk_index in 0..16 {
            let low_index = (raw_chunk_index / 8) * 16
                + ((raw_chunk_index / 2) & 1) * 8
                + (raw_chunk_index & 1) * 4;
            let high_index = (raw_chunk_index / 8) * 8 + (raw_chunk_index & 1) * 4;
            let scale_index = (raw_chunk_index % 2) + (raw_chunk_index / 2) * 2;
            let chunk_index = (raw_chunk_index / 2) & FOURTH_TWO_BITS as u32;

            writeln!(
                code,
                "let chunk_raw_scale_{raw_chunk_index} = {datatype}({});",
                index_signed_bytes(format!("{chunk}.scales"), scale_index)
            )
            .unwrap();
            let high_mask = match chunk_index {
                0 => FOURTH_TWO_BITS,
                1 => THIRD_TWO_BITS,
                2 => SECOND_TWO_BITS,
                _ => FIRST_TWO_BITS,
            };
            let low_mask = if chunk_index > 1 {
                FIRST_HALF_BITS
            } else {
                SECOND_HALF_BITS
            };
            let coefficient = if chunk_index > 1 {
                shift_right_scale(4)
            } else {
                1.0
            };

            writeln!(
                code,
                "let chunk_midpoint_{raw_chunk_index} = scale * chunk_raw_scale_{raw_chunk_index} * {datatype}({CENTER_SIX_BIT});"
            )
            .unwrap();
            writeln!(
                code,
                "let chunk_scale_{raw_chunk_index} = scale * chunk_raw_scale_{raw_chunk_index} * {coefficient};"
            )
            .unwrap();

            for vec4_index in 0..4 {
                writeln!(
                    code,
                    "let low_chunk_{raw_chunk_index}_{vec4_index} = unpack4xU8({chunk}.data_low_bits[{low_index} + {vec4_index}]);",
                )
                .unwrap();
                writeln!(
                    code,
                    "let high_chunk_{raw_chunk_index}_{vec4_index} = unpack4xU8({chunk}.data_high_bits[{high_index} + {vec4_index}]);",
                )
                .unwrap();

                for offset in 0..4 {
                    writeln!(
                        code,
                        "let low_byte_{raw_chunk_index}_{vec4_index}_{offset} = low_chunk_{raw_chunk_index}_{vec4_index}[{offset}] & {low_mask};",
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "let high_byte_{raw_chunk_index}_{vec4_index}_{offset} = high_chunk_{raw_chunk_index}_{vec4_index}[{offset}] & {high_mask};",
                    )
                    .unwrap();

                    let shift = if chunk_index & 1 == 1 { 2 } else { 4 };

                    writeln!(
                        code,
                        "let merged_{raw_chunk_index}_{vec4_index}_{offset} = {datatype}(low_byte_{raw_chunk_index}_{vec4_index}_{offset} | (high_byte_{raw_chunk_index}_{vec4_index}_{offset} << {shift}));",
                    )
                    .unwrap();

                    writeln!(
                        code,
                        "let scaled_{raw_chunk_index}_{vec4_index}_{offset} = {datatype}(chunk_scale_{raw_chunk_index} * merged_{raw_chunk_index}_{vec4_index}_{offset} - chunk_midpoint_{raw_chunk_index});"
                    )
                    .unwrap();

                    process_element(
                        raw_chunk_index * 16 + vec4_index * 4 + offset,
                        format!("scaled_{raw_chunk_index}_{vec4_index}_{offset}"),
                        code,
                    );
                }
            }
        }
    }

    fn dequantize_vec4_block<W: Write>(
        chunk: String,
        datatype: DataTypeEnum,
        mut process_element: impl FnMut(String, String, &mut W),
        code: &mut W,
    ) {
        _ = writeln!(
            code,
            "for (var raw_chunk_index = 0u; raw_chunk_index < 16u; raw_chunk_index += 1u) {{",
        );
        _ = write_q6_k_dequant(code, datatype, &chunk, "raw_chunk_index", |chunk, code| {
            for offset in 0..4 {
                process_element(
                    format!("raw_chunk_index * 4u + {offset}"),
                    format!("{chunk}[{offset}]"),
                    code,
                );
            }
        });
        _ = writeln!(code, "}}");
    }

    const MATRIX_BLOCKS: usize = 16;

    fn dequantize_4x4_block<W: Write>(
        index: &str,
        chunk: String,
        datatype: DataTypeEnum,
        process_element: impl FnOnce(String, &mut W),
        code: &mut W,
    ) {
        _ = write_q6_k_dequant(code, datatype, &chunk, index, process_element);
    }

    fn write_type<W: Write>(f: &mut W) -> std::fmt::Result {
        write_q6_k_type(f)
    }
}

fn write_q6_k_dequant<W: Write>(
    code: &mut W,
    datatype: DataTypeEnum,
    chunk: &str,
    index: &str,
    process_element: impl FnOnce(String, &mut W),
) -> std::fmt::Result {
    _ = writeln!(code, "let scale = {datatype}({chunk}.scale);");
    writeln!(
        code,
        "let low_index = ({index} / 8u) * 16u + (({index} / 2u) & 1u) * 8u + ({index} & 1u) * 4u;"
    )?;
    writeln!(
        code,
        "let high_index = ({index} / 8u) * 8u + ({index} & 1u) * 4u;"
    )?;
    writeln!(
        code,
        "let scale_index = ({index} % 2u) + ({index} / 2u) * 2u;"
    )?;
    writeln!(
        code,
        "let chunk_index = ({index} / 2u) & {FOURTH_TWO_BITS}u;"
    )?;

    writeln!(
        code,
        "let chunk_raw_scale = {datatype}({});",
        index_signed_bytes(format!("{chunk}.scales"), "scale_index")
    )?;
    writeln!(
        code,
        "let high_mask = select(select({FOURTH_TWO_BITS}u, {THIRD_TWO_BITS}u, chunk_index > 0u), select({SECOND_TWO_BITS}u, {FIRST_TWO_BITS}u, chunk_index > 2u), chunk_index > 1u);"
    )?;
    writeln!(
        code,
        "let low_mask = select({SECOND_HALF_BITS}u, {FIRST_HALF_BITS}u, chunk_index > 1u);"
    )?;
    let shift_4 = shift_right_scale(4);
    writeln!(
        code,
        "let coefficient = select({datatype}(1.0), {datatype}({shift_4}), chunk_index > 1u);"
    )?;

    writeln!(
        code,
        "let chunk_midpoint = scale * chunk_raw_scale * {datatype}({CENTER_SIX_BIT});"
    )?;
    writeln!(
        code,
        "let chunk_scale = scale * chunk_raw_scale * coefficient;"
    )?;

    for vec4_index in 0..4 {
        writeln!(
            code,
            "let low_chunk_{vec4_index} = unpack4xU8({chunk}.data_low_bits[low_index + {vec4_index}]);",
        )?;
        writeln!(
            code,
            "let high_chunk_{vec4_index} = unpack4xU8({chunk}.data_high_bits[high_index + {vec4_index}]);",
        )?;

        for offset in 0..4 {
            writeln!(
                code,
                "let low_byte_{vec4_index}_{offset} = low_chunk_{vec4_index}[{offset}] & low_mask;",
            )?;
            writeln!(
                code,
                "let high_byte_{vec4_index}_{offset} = high_chunk_{vec4_index}[{offset}] & high_mask;",
            )?;

            writeln!(
                code,
                "let merged_{vec4_index}_{offset} = {datatype}(low_byte_{vec4_index}_{offset} | select(high_byte_{vec4_index}_{offset} << 4, high_byte_{vec4_index}_{offset} << 2, (chunk_index & 1u) == 1u));",
            )?;

            writeln!(
                code,
                "let scaled_{vec4_index}_{offset} = {datatype}(merged_{vec4_index}_{offset});"
            )?;
        }
    }
    // Group the results into a mat4x4
    write!(code, "let scaled = mat4x4<{datatype}>(")?;
    for row in 0..4 {
        if row > 0 {
            writeln!(code, ", ")?;
        }
        write!(code, "vec4(")?;
        for col in 0..4 {
            if col > 0 {
                write!(code, ", ")?;
            }
            write!(code, "scaled_{row}_{col}")?;
        }
        write!(code, ") * chunk_scale - chunk_midpoint")?;
    }
    writeln!(code, ");")?;
    process_element("scaled".to_string(), code);
    Ok(())
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_de_quantize_4_0_block() {
    fuzz_de_quantize::<BlockQ4_0>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_de_quantize_5_0_block() {
    fuzz_de_quantize::<BlockQ5_0>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_de_quantize_8_0_block() {
    fuzz_de_quantize::<BlockQ8_0>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_de_quantize_4_k_block() {
    fuzz_de_quantize::<BlockQ4K>().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_de_quantize_6_k_block() {
    fuzz_de_quantize::<BlockQ6K>().await;
}

#[cfg(test)]
async fn fuzz_de_quantize<B: WgslQuantizedType + PartialEq + std::fmt::Debug>()
where
    rand::distr::StandardUniform:
        rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
{
    use pretty_assertions::assert_eq;
    use wgpu::util::DownloadBuffer;

    use crate::FloatDataType;

    println!("testing f32 compact...");
    test_fuzz_de_quantize_block_inner::<B, f32>(false, false).await;
    println!("testing f32 vec4...");
    test_fuzz_de_quantize_block_inner::<B, f32>(false, true).await;
    println!("testing f32 unrolled...");
    test_fuzz_de_quantize_block_inner::<B, f32>(true, false).await;
    println!("testing f32 mat4x4...");
    test_fuzz_de_quantize_mat4x4_inner::<B, f32>().await;

    async fn test_fuzz_de_quantize_block_inner<
        B: WgslQuantizedType + PartialEq + std::fmt::Debug,
        T: FloatDataType,
    >(
        unrolled: bool,
        vec4: bool,
    ) where
        rand::distr::StandardUniform:
            rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
    {
        let dtype = T::WGSL_TYPE;
        let device = crate::Device::new().await.unwrap();
        let mut kernel_body = String::new();
        if unrolled {
            B::unrolled_dequantize_block(
                "block".to_string(),
                dtype,
                |i, data, code| {
                    writeln!(code, "output[{i}] = {data};",).unwrap();
                },
                &mut kernel_body,
            )
        } else if vec4 {
            B::dequantize_vec4_block(
                "block".to_string(),
                dtype,
                |i, data, code| {
                    for v in 0..4 {
                        writeln!(code, "output[({i})*4u + {v}] = {data}[{v}];",).unwrap();
                    }
                },
                &mut kernel_body,
            )
        } else {
            B::dequantize_block(
                "block".to_string(),
                dtype,
                |i, data, code| {
                    writeln!(code, "output[{i}] = {data};",).unwrap();
                },
                &mut kernel_body,
            )
        };
        let mut kernel = String::new();
        writeln!(&mut kernel, "enable f16;").unwrap();
        B::write_type(&mut kernel).unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(0) var<storage, read> block: {};",
            B::GGML_TYPE
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(1) var<storage, read_write> output: array<{dtype}, {}>;",
            B::BLOCK_SIZE
        )
        .unwrap();
        writeln!(&mut kernel, "@compute @workgroup_size(1, 1, 1)").unwrap();
        writeln!(&mut kernel, "fn main() {{").unwrap();
        writeln!(&mut kernel, "{kernel_body}").unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        let bind_group_layout =
            device
                .wgpu_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let compute_pipeline_layout =
            device
                .wgpu_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let module = device.create_shader_module(kernel);
        let pipeline =
            device
                .wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&compute_pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                });
        for _ in 0..200 {
            let block_bytes: B::AsBytes = rand::random();
            let block = bytemuck::pod_read_unaligned::<B>(block_bytes.as_ref());
            if !block.finite() {
                continue;
            }
            let block_wgsl = block.into_wgsl_bytes();
            assert_eq!(block, B::from_wgsl_bytes(block_wgsl));
            let output =
                device
                    .wgpu_device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&vec![T::zero(); B::BLOCK_SIZE]),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
            let bind_group = device
                .wgpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &device.wgpu_device().create_buffer_init(
                                    &wgpu::util::BufferInitDescriptor {
                                        label: None,
                                        contents: block_wgsl.as_ref(),
                                        usage: wgpu::BufferUsages::STORAGE
                                            | wgpu::BufferUsages::COPY_SRC,
                                    },
                                ),
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &output,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                });

            let mut encoder = device
                .wgpu_device()
                .create_command_encoder(&Default::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = [1, 1, 1];
                cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z);
            }
            device.wgpu_queue().submit(Some(encoder.finish()));

            let (sender, receiver) = futures_channel::oneshot::channel();
            DownloadBuffer::read_buffer(
                device.wgpu_device(),
                device.wgpu_queue(),
                &output.slice(..),
                move |result| {
                    _ = sender.send(result);
                },
            );
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
            let output = receiver
                .await
                .map_err(|_| wgpu::BufferAsyncError)
                .unwrap()
                .unwrap();

            let ouptut_as_floats = bytemuck::cast_slice::<_, T>(&output);
            let expected_result = block
                .dequantize()
                .as_ref()
                .iter()
                .map(|x| T::from_f32(*x))
                .collect::<Vec<_>>();
            assert_eq!(
                ouptut_as_floats, expected_result,
                "Block: {block:?}, Output: {ouptut_as_floats:?}, Expected: {expected_result:?}, kernel: {kernel_body}"
            );
        }
    }

    async fn test_fuzz_de_quantize_mat4x4_inner<
        B: WgslQuantizedType + PartialEq + std::fmt::Debug,
        T: FloatDataType,
    >()
    where
        rand::distr::StandardUniform:
            rand::prelude::Distribution<<B as fusor_gguf::GgufBlock>::AsBytes>,
    {
        let dtype = T::WGSL_TYPE;
        let device = crate::Device::new().await.unwrap();
        let mut kernel_body = String::new();

        // Dequantize each mat4x4 block and write it to output
        for i in 0..B::MATRIX_BLOCKS {
            // Wrap each block in its own scope to avoid variable name conflicts
            writeln!(&mut kernel_body, "{{").unwrap();
            B::dequantize_4x4_block(
                &format!("{i}u"),
                "block".to_string(),
                dtype,
                |mat, code| {
                    // Extract each element from the mat4x4 and write to output
                    // mat4x4 in WGSL is column-major, so mat[col][row] accesses column col, row row
                    for col in 0..4 {
                        for row in 0..4 {
                            let idx = i * 16 + col * 4 + row;
                            writeln!(code, "output[{idx}] = {mat}[{col}][{row}];").unwrap();
                        }
                    }
                },
                &mut kernel_body,
            );
            writeln!(&mut kernel_body, "}}").unwrap();
        }

        let mut kernel = String::new();
        writeln!(&mut kernel, "enable f16;").unwrap();
        B::write_type(&mut kernel).unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(0) var<storage, read> block: {};",
            B::GGML_TYPE
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "@group(0) @binding(1) var<storage, read_write> output: array<{dtype}, {}>;",
            B::BLOCK_SIZE
        )
        .unwrap();
        writeln!(&mut kernel, "@compute @workgroup_size(1, 1, 1)").unwrap();
        writeln!(&mut kernel, "fn main() {{").unwrap();
        writeln!(&mut kernel, "{kernel_body}").unwrap();
        writeln!(&mut kernel, "}}").unwrap();

        let bind_group_layout =
            device
                .wgpu_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let compute_pipeline_layout =
            device
                .wgpu_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let module = device.create_shader_module(kernel);
        let pipeline =
            device
                .wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&compute_pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                });
        for _ in 0..200 {
            let block_bytes: B::AsBytes = rand::random();
            let block = bytemuck::pod_read_unaligned::<B>(block_bytes.as_ref());
            if !block.finite() {
                continue;
            }
            let block_wgsl = block.into_wgsl_bytes();
            assert_eq!(block, B::from_wgsl_bytes(block_wgsl));
            let output =
                device
                    .wgpu_device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&vec![T::zero(); B::BLOCK_SIZE]),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
            let bind_group = device
                .wgpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &device.wgpu_device().create_buffer_init(
                                    &wgpu::util::BufferInitDescriptor {
                                        label: None,
                                        contents: block_wgsl.as_ref(),
                                        usage: wgpu::BufferUsages::STORAGE
                                            | wgpu::BufferUsages::COPY_SRC,
                                    },
                                ),
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &output,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                });

            let mut encoder = device
                .wgpu_device()
                .create_command_encoder(&Default::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let [workgroup_size_x, workgroup_size_y, workgroup_size_z] = [1, 1, 1];
                cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z);
            }
            device.wgpu_queue().submit(Some(encoder.finish()));

            let (sender, receiver) = futures_channel::oneshot::channel();
            DownloadBuffer::read_buffer(
                device.wgpu_device(),
                device.wgpu_queue(),
                &output.slice(..),
                move |result| {
                    _ = sender.send(result);
                },
            );
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
            let output = receiver
                .await
                .map_err(|_| wgpu::BufferAsyncError)
                .unwrap()
                .unwrap();

            let ouptut_as_floats = bytemuck::cast_slice::<_, T>(&output);
            let expected_result = block
                .dequantize()
                .as_ref()
                .iter()
                .map(|x| T::from_f32(*x))
                .collect::<Vec<_>>();
            assert_eq!(
                ouptut_as_floats, expected_result,
                "Block: {block:?}, Output: {ouptut_as_floats:?}, Expected: {expected_result:?}, kernel: {kernel_body}"
            );
        }
    }
}
