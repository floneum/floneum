//! Support for [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files

// Tensor layout is described at https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
// Modified from https://github.com/huggingface/candle/blob/e286cf7cc9e34bc426a542264b818e35e6eed05b/candle-core/src/quantized/gguf_file.rs#L31

use bytemuck::{AnyBitPattern, Contiguous, bytes_of};
use rustc_hash::FxHashMap;

const GGUF_MAGIC_BYTES: [u8; 4] = *b"GGUF";

fn check_magic<R: std::io::Read>(reader: &mut R) -> Result<(), GgufReadError> {
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if magic == GGUF_MAGIC_BYTES
        || magic
            .iter()
            .rev()
            .zip(GGUF_MAGIC_BYTES.iter())
            .all(|(a, b)| a == b)
    {
        Ok(())
    } else {
        Err(GgufReadError::MagicBytesMismatch)
    }
}

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl TryFrom<u32> for GgmlDType {
    type Error = GgufReadError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            _ => Err(GgufReadError::UnsupportedDType(value)),
        }
    }
}

impl GgmlDType {
    /// The number of elements in each block
    pub const fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// The size of each block in bytes
    pub fn block_allocation_size(&self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::F16 => std::mem::size_of::<half::f16>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            _ => todo!("implement block_allocation_size for {self:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufVersion {
    V1,
    V2,
    V3,
}

impl GgufVersion {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self, GgufReadError> {
        check_magic(reader)?;
        let version = read_le_u32(reader)?;
        Self::try_from(version)
    }
}

impl TryFrom<u32> for GgufVersion {
    type Error = GgufReadError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            _ => Err(GgufReadError::UnsupportedVersion(value)),
        }
    }
}

fn read_le_u8<R: std::io::Read>(reader: &mut R) -> Result<u8, GgufReadError> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(u8::from_le_bytes(bytes))
}

fn read_le_u16<R: std::io::Read>(reader: &mut R) -> Result<u16, GgufReadError> {
    let mut bytes = [0; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_le_u32<R: std::io::Read>(reader: &mut R) -> Result<u32, GgufReadError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_le_u64<R: std::io::Read>(reader: &mut R) -> Result<u64, GgufReadError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_le_i8<R: std::io::Read>(reader: &mut R) -> Result<i8, GgufReadError> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(i8::from_le_bytes(bytes))
}

fn read_le_i16<R: std::io::Read>(reader: &mut R) -> Result<i16, GgufReadError> {
    let mut bytes = [0; 2];
    reader.read_exact(&mut bytes)?;
    Ok(i16::from_le_bytes(bytes))
}

fn read_le_i32<R: std::io::Read>(reader: &mut R) -> Result<i32, GgufReadError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_le_i64<R: std::io::Read>(reader: &mut R) -> Result<i64, GgufReadError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_le_f32<R: std::io::Read>(reader: &mut R) -> Result<f32, GgufReadError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_le_f64<R: std::io::Read>(reader: &mut R) -> Result<f64, GgufReadError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn read_array_length<R: std::io::Read>(
    reader: &mut R,
    version: GgufVersion,
) -> Result<usize, GgufReadError> {
    Ok(match version {
        GgufVersion::V1 => read_le_u32(reader)? as usize,
        GgufVersion::V2 | GgufVersion::V3 => read_le_u64(reader)? as usize,
    })
}

#[derive(thiserror::Error, Debug)]
pub enum GgufReadError {
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic bytes")]
    MagicBytesMismatch,
    #[error("unsupported magic {0}")]
    UnsupportedVersion(u32),
    #[error("unsupported dtype {0}")]
    UnsupportedDType(u32),
    #[error("invalid bool value")]
    InvalidBool,
    #[error("invalid value type {0}")]
    InvalidValueType(#[from] InvalidValueType),
    #[error("tensor size ({tensor_elems}) is not a multiple of the block size ({block_size})")]
    InvalidTensorSize {
        tensor_elems: usize,
        block_size: usize,
    },
}

#[derive(Debug)]
pub struct TensorMetadata {
    pub ggml_dtype: GgmlDType,
    pub shape: Box<[u32]>,
    pub offset: u64,
}

impl TensorMetadata {
    fn read<R: std::io::Read + std::io::Seek>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Box<[u8]>, GgufReadError> {
        let tensor_elems = self.shape.iter().copied().product::<u32>() as usize;
        let block_size = self.ggml_dtype.block_size();
        if tensor_elems % block_size != 0 {
            return Err(GgufReadError::InvalidTensorSize {
                tensor_elems,
                block_size,
            });
        }
        let size_in_bytes = (tensor_elems / block_size) * self.ggml_dtype.block_allocation_size();
        let mut raw_data = vec![0u8; size_in_bytes].into_boxed_slice();
        reader.seek(std::io::SeekFrom::Start(tensor_data_offset + self.offset))?;
        reader.read_exact(&mut raw_data)?;
        Ok(raw_data)
    }
}

#[derive(Debug)]
pub struct GgufMetadata {
    pub version: GgufVersion,
    pub metadata: FxHashMap<Box<str>, GgufValue>,
    pub tensor_infos: FxHashMap<Box<str>, TensorMetadata>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read>(
    reader: &mut R,
    magic: GgufVersion,
) -> Result<Box<str>, GgufReadError> {
    let len = read_array_length(reader, magic)?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = bytes.last() {
        bytes.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&bytes).into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Contiguous)]
#[repr(u8)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8 = 0,
    // The value is a 8-bit signed integer.
    I8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    U16 = 2,
    // The value is a 16-bit signed little-endian integer.
    I16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    U32 = 4,
    // The value is a 32-bit signed little-endian integer.
    I32 = 5,
    // The value is a 64-bit unsigned little-endian integer.
    U64 = 10,
    // The value is a 64-bit signed little-endian integer.
    I64 = 11,
    // The value is a 32-bit IEEE754 floating point number.
    F32 = 6,
    // The value is a 64-bit IEEE754 floating point number.
    F64 = 12,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String = 8,
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array = 9,
}

#[derive(thiserror::Error, Debug)]
#[error("invalid value type {0}")]
pub struct InvalidValueType(u32);

impl TryFrom<u32> for ValueType {
    type Error = InvalidValueType;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let as_u8 = value.try_into().map_err(|_| InvalidValueType(value))?;
        Self::from_integer(as_u8).ok_or(InvalidValueType(value))
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(Box<str>),
    Array(Box<[GgufValue]>),
}

impl GgufValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: ValueType,
        version: GgufVersion,
    ) -> Result<Self, GgufReadError> {
        let v = match value_type {
            ValueType::U8 => Self::U8(read_le_u8(reader)?),
            ValueType::I8 => Self::I8(read_le_i8(reader)?),
            ValueType::U16 => Self::U16(read_le_u16(reader)?),
            ValueType::I16 => Self::I16(read_le_i16(reader)?),
            ValueType::U32 => Self::U32(read_le_u32(reader)?),
            ValueType::I32 => Self::I32(read_le_i32(reader)?),
            ValueType::U64 => Self::U64(read_le_u64(reader)?),
            ValueType::I64 => Self::I64(read_le_i64(reader)?),
            ValueType::F32 => Self::F32(read_le_f32(reader)?),
            ValueType::F64 => Self::F64(read_le_f64(reader)?),
            ValueType::Bool => match read_le_u8(reader)? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                _ => return Err(GgufReadError::InvalidBool),
            },
            ValueType::String => Self::String(read_string(reader, version)?),
            ValueType::Array => {
                let value_type = read_le_u32(reader)?;
                let value_type = ValueType::try_from(value_type)?;
                let len = read_array_length(reader, version)?;
                let vs: Result<Box<[_]>, _> = (0..len)
                    .map(|_| GgufValue::read(reader, value_type, version))
                    .collect();
                Self::Array(vs?)
            }
        };
        Ok(v)
    }
}

impl GgufMetadata {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self, GgufReadError> {
        let version = GgufVersion::read(reader)?;

        let tensor_count = read_array_length(reader, version)?;
        let metadata_kv_count = read_array_length(reader, version)?;

        let mut metadata = FxHashMap::default();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, version)?;
            let value_type = read_le_u32(reader)?;
            let value_type = ValueType::try_from(value_type)?;
            let value = GgufValue::read(reader, value_type, version)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = FxHashMap::default();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, version)?;
            let dimensions = read_le_u32(reader)? as usize;

            let shape: Result<Box<[u32]>, _> = (0..dimensions)
                .map(|_| read_array_length(reader, version).map(|i| i as u32))
                .collect();
            let mut shape = shape?;

            shape.reverse();
            let ggml_dtype = read_le_u32(reader)?;
            let ggml_dtype = GgmlDType::try_from(ggml_dtype)?;
            let offset = read_le_u64(reader)?;
            tensor_infos.insert(
                tensor_name,
                TensorMetadata {
                    shape,
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U8(v)) => *v as u64,
            Some(GgufValue::U16(v)) => *v as u64,
            Some(GgufValue::U32(v)) => *v as u64,
            Some(GgufValue::I8(v)) if *v >= 0 => *v as u64,
            Some(GgufValue::I16(v)) if *v >= 0 => *v as u64,
            Some(GgufValue::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self {
            version,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }
}

const Q4_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) scale: half::f16,
    pub(crate) qs: [u8; Q4_0_BLOCK_SIZE / 2],
}

impl BlockQ4_0 {
    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L255
    fn dequantize(&self) -> [f32; Q4_0_BLOCK_SIZE] {
        const CENTER_FOUR_BIT: i8 = 8;

        let scale = self.scale.to_f32();
        let mut data = [0.0; Q4_0_BLOCK_SIZE];

        for (i, byte) in self.qs.iter().enumerate() {
            let low_data = (byte & 0x0F) as i8 - CENTER_FOUR_BIT;
            let high_data = (byte >> 4) as i8 - CENTER_FOUR_BIT;

            let low_data = low_data as f32 * scale;
            let high_data = high_data as f32 * scale;

            data[i] = low_data;
            data[i + Q4_0_BLOCK_SIZE / 2] = high_data;
        }

        data
    }
}

const Q5_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) scale: half::f16,
    // The highest bit for each of the 5 bit values
    pub(crate) data_high_bits: [u8; 4],
    // The low four bits for each of the 5 bit values
    pub(crate) data_low_bits: [u8; Q5_0_BLOCK_SIZE / 2],
}

impl BlockQ5_0 {
    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L296
    fn dequantize(&self) -> [f32; Q5_0_BLOCK_SIZE] {
        const FIFTH_BIT: u8 = 0x10;
        const CENTER_FIVE_BIT: i8 = 16;

        let scale = self.scale.to_f32();
        let high_bits: u32 = bytemuck::cast(self.data_high_bits);
        let mut out = [0.0; Q5_0_BLOCK_SIZE];

        for (i, byte) in self.data_low_bits.iter().enumerate() {
            let low = byte & 0x0F;
            let high = byte >> 4;

            // Get the nth bit from the start and make it our fifth bit
            let low_bit = ((high_bits >> i) << 4) as u8 & FIFTH_BIT;
            // Get the nth bit from the last half of the high bits and make it our fifth bit
            let high_bit = (high_bits >> (i + 12)) as u8 & FIFTH_BIT;

            // Merge the two chunks together. The _0 quant variants are always centered at the
            // middle of the bit range
            let low_data = (low | low_bit) as i8 - CENTER_FIVE_BIT;
            let high_data = (high | high_bit) as i8 - CENTER_FIVE_BIT;

            out[i] = low_data as f32 * scale;
            out[i + Q5_0_BLOCK_SIZE / 2] = high_data as f32 * scale;
        }

        out
    }
}

const Q8_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) scale: half::f16,
    pub(crate) data: [i8; Q8_0_BLOCK_SIZE],
}

impl BlockQ8_0 {
    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L349
    fn dequantize(&self) -> [f32; Q8_0_BLOCK_SIZE] {
        let scale = self.scale.to_f32();

        std::array::from_fn(|i| self.data[i] as f32 * scale)
    }
}

const K_BLOCK_SIZE: usize = 256;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ4K {
    scale: half::f16,
    min: half::f16,
    scales: [u8; 12],
    weights: [u8; K_BLOCK_SIZE / 2],
}

impl BlockQ4K {
    fn dequantize(&self) -> [f32; K_BLOCK_SIZE] {
        let weights = &self.weights;
        let super_block_scale = self.scale.to_f32();
        let super_block_min = self.min.to_f32();
        let scales = bytemuck::cast_slice(&self.scales);

        let mut data = [0.0; K_BLOCK_SIZE];
        let (first_scales, first_offset) = first_scales_min_k4(scales);
        let (second_scales, second_offset) = second_scales_min_k4(scales);
        let scales: [u8; 8] = bytemuck::cast([first_scales, second_scales]);
        let offsets: [u8; 8] = bytemuck::cast([first_offset, second_offset]);
        let mut pair_index = 0;
        for chunk_index in (0..K_BLOCK_SIZE / 2).step_by(32) {
            let out_chunk_index = chunk_index * 2;
            let low_scale = scales[pair_index];
            let low_offset = offsets[pair_index];
            let low_scale = low_scale as f32 * super_block_scale;
            let low_offset = low_offset as f32 * super_block_min;
            pair_index += 1;
            let high_scale = scales[pair_index];
            let high_offset = offsets[pair_index];
            let high_scale = high_scale as f32 * super_block_scale;
            let high_offset = high_offset as f32 * super_block_min;
            pair_index += 1;

            for offset in 0..32 {
                let weight = weights[chunk_index + offset];
                data[out_chunk_index + offset] = low_scale * (weight & 0xF) as f32 - low_offset;
                data[out_chunk_index + offset + 32] =
                    high_scale * (weight >> 4) as f32 - high_offset;
            }
        }

        data
    }
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ6K {
    // The 4 low bits of the each value
    pub(crate) data_low_bits: [u8; K_BLOCK_SIZE / 2],
    // The 2 high bits of the each value
    pub(crate) data_high_bits: [u8; K_BLOCK_SIZE / 4],
    // Full byte scales for each block of 16 values
    pub(crate) scales: [i8; K_BLOCK_SIZE / 16],
    // The scale of the super block
    pub(crate) scale: half::f16,
}

impl BlockQ6K {
    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L1690
    fn dequantize(&self) -> [f32; K_BLOCK_SIZE] {
        const CENTER_SIX_BIT: i8 = 32;
        const TWO_BITS: u8 = 0b11;
        const FOUR_BITS: u8 = 0b1111;

        let scale = self.scale.to_f32();
        let mut data = [0.0; K_BLOCK_SIZE];

        for chunk_index in 0..2 {
            let output_index = chunk_index * 128;
            let lower_index = chunk_index * 64;
            let high_index = chunk_index * 32;
            let scale_index = chunk_index * 8;
            // Load 128 6 bit values into data in groups of 4
            for high_byte_index in 0..32 {
                let scale_index = scale_index + high_byte_index / 16;
                let high_byte = self.data_high_bits[high_index + high_byte_index];
                let first_low_byte = self.data_low_bits[lower_index + high_byte_index];
                let second_low_byte = self.data_low_bits[lower_index + high_byte_index + 32];

                let first_two_bits = high_byte & TWO_BITS;
                let first_high_nibble = first_low_byte & FOUR_BITS;
                let first_merged =
                    ((first_two_bits << 4) | first_high_nibble) as i8 - CENTER_SIX_BIT;
                data[output_index + high_byte_index] =
                    scale * self.scales[scale_index] as f32 * first_merged as f32;

                let second_two_bits = (high_byte >> 2) & TWO_BITS;
                let second_high_nibble = second_low_byte & FOUR_BITS;
                let second_merged =
                    ((second_two_bits << 4) | second_high_nibble) as i8 - CENTER_SIX_BIT;
                data[output_index + high_byte_index + 32] =
                    scale * self.scales[scale_index + 2] as f32 * second_merged as f32;

                let third_two_bits = (high_byte >> 4) & TWO_BITS;
                let third_high_nibble = first_low_byte >> 4;
                let third_merged =
                    ((third_two_bits << 4) | third_high_nibble) as i8 - CENTER_SIX_BIT;
                data[output_index + high_byte_index + 64] =
                    scale * self.scales[scale_index + 4] as f32 * third_merged as f32;

                let fourth_two_bits = (high_byte >> 6) & TWO_BITS;
                let fourth_high_nibble = second_low_byte >> 4;
                let fourth_merged =
                    ((fourth_two_bits << 4) | fourth_high_nibble) as i8 - CENTER_SIX_BIT;
                data[output_index + high_byte_index + 96] =
                    scale * self.scales[scale_index + 6] as f32 * fourth_merged as f32;
            }
        }

        data
    }
}

const SIX_BITS_MASK: u32     = 0b0011_1111_0011_1111_0011_1111_0011_1111;
const MSB_TWO_BITS_MASK: u32 = 0b1100_0000_1100_0000_1100_0000_1100_0000;
const MSB_SCALES_MASK: u32 = 0b0000_1111_0000_1111_0000_1111_0000_1111;
const MSB_OFFSET_MASK: u32 = 0b1111_0000_1111_0000_1111_0000_1111_0000;

fn first_scales_min_k4(packed_scales: &[u32]) -> (u32, u32) {
    let first_four_bytes = packed_scales[0];
    let middle_four_bytes = packed_scales[1];

    let first_scales = first_four_bytes & SIX_BITS_MASK;
    let first_offsets = middle_four_bytes & SIX_BITS_MASK;

    (first_scales, first_offsets)
}

fn second_scales_min_k4(packed_scales: &[u32]) -> (u32, u32) {
    let first_four_bytes = packed_scales[0];
    let middle_four_bytes = packed_scales[1];
    let last_four_bytes = packed_scales[2];

    let msb_scales = (first_four_bytes & MSB_TWO_BITS_MASK) >> 2;
    let lsb_scales = last_four_bytes & MSB_SCALES_MASK;
    let second_scales = msb_scales | lsb_scales;

    let msb_offsets = (middle_four_bytes & MSB_TWO_BITS_MASK) >> 2;
    let lsb_offsets = (last_four_bytes & MSB_OFFSET_MASK) >> 4;
    let second_offsets = msb_offsets | lsb_offsets;

    (second_scales, second_offsets)
}

const SIX_BITS: u8 = 0b0011_1111;
const FOUR_BITS: u8 = 0b0000_1111;

// pair_index is in the range [0, 7]. It is the index of the pair of 6 bit (scale, offset) pairs
fn get_scale_min_k4(pair_index: u8, packed_scales: &[u8; 12]) -> (u8, u8) {
    if pair_index < 4 {
        // Extracts this bit pattern. The first 6 bits of the first
        // 4 bytes are the scales. The first 6 bits of the second 4
        // bytes are the offsets.
        //
        // dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
        // __000000|__111111|__222222|__333333|__000000|__111111
        //
        // mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
        // __222222|__333333|________|________|________|________
        let scale = packed_scales[pair_index as usize] & SIX_BITS;
        let offset = packed_scales[(pair_index + 4) as usize] & SIX_BITS;

        (scale, offset)
    } else {
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
        // Get the byte with the 4 LSB of the scale and offset
        let shared_byte = packed_scales[(pair_index + 4) as usize];

        let scale_bottom_four_bits = shared_byte & FOUR_BITS;
        // Get the 2 MSB of the scale
        let scale_top_two_bits = packed_scales[(pair_index - 4) as usize] >> 6;
        let scale = scale_bottom_four_bits | scale_top_two_bits << 4;

        let offset_bottom_four_bits = shared_byte >> 4;
        // Get the 2 MSB of the offset
        let offset_top_two_bits = packed_scales[pair_index as usize] >> 6;
        let offset = offset_bottom_four_bits | offset_top_two_bits << 4;

        (scale, offset)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_load_tiny_llama() {
    use pretty_assertions::assert_eq;

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let device = candle_core::Device::Cpu;
    for (name, candle_tensor) in candle_metadata.tensor_infos {
        let tensor = metadata.tensor_infos.get(&*name).unwrap();
        println!("{}: {:?}", name, tensor);
        let tensor_bytes = tensor
            .read(&mut reader, metadata.tensor_data_offset)
            .unwrap();
        let candle_tensor = candle_tensor
            .read(&mut reader, candle_metadata.tensor_data_offset, &device)
            .unwrap();
        let candle_tensor_de_quantized = candle_tensor.dequantize(&device).unwrap();
        let candle_tensor_data: Vec<f32> = candle_tensor_de_quantized
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        match tensor.ggml_dtype {
            GgmlDType::Q4_0 => {
                let blocks: &[BlockQ4_0] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in blocks
                    .iter()
                    .zip(candle_tensor_data.chunks(Q4_0_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {:?}", dequantized);
                            println!("candle: {:?}", candle_block);
                            assert_eq!(dequantized, candle_block);
                            panic!();
                        }
                    }
                }
            }
            GgmlDType::Q5_0 => {
                let blocks: &[BlockQ5_0] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in blocks
                    .iter()
                    .zip(candle_tensor_data.chunks(Q5_0_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {:?}", dequantized);
                            println!("candle: {:?}", candle_block);
                            assert_eq!(dequantized, candle_block);
                            panic!();
                        }
                    }
                }
            }
            GgmlDType::Q8_0 => {
                let blocks: &[BlockQ8_0] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in blocks
                    .iter()
                    .zip(candle_tensor_data.chunks(Q8_0_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {:?}", dequantized);
                            println!("candle: {:?}", candle_block);
                            assert_eq!(dequantized, candle_block);
                            panic!();
                        }
                    }
                }
            }
            GgmlDType::Q4K => {
                let blocks: &[BlockQ4K] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in
                    blocks.iter().zip(candle_tensor_data.chunks(K_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {:?}", dequantized);
                            println!("candle: {:?}", candle_block);
                            assert_eq!(dequantized, candle_block);
                            panic!();
                        }
                    }
                }
            }
            GgmlDType::Q6K => {
                let blocks: &[BlockQ6K] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in
                    blocks.iter().zip(candle_tensor_data.chunks(K_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {:?}", dequantized);
                            println!("candle: {:?}", candle_block);
                            assert_eq!(dequantized, candle_block);
                            panic!();
                        }
                    }
                }
            }
            GgmlDType::F32 => {
                let blocks: &[f32] = bytemuck::cast_slice(&tensor_bytes);
                for (a, b) in blocks.iter().zip(candle_tensor_data) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
            _ => todo!(),
        }
    }
}
