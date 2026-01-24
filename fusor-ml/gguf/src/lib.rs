//! Support for [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files

// Enable NEON dot product intrinsics for faster quantized matmul on Apple Silicon
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]

// Tensor layout is described at https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
// Modified from https://github.com/huggingface/candle/blob/e286cf7cc9e34bc426a542264b818e35e6eed05b/candle-core/src/quantized/gguf_file.rs#L31

use std::{fmt::Display, mem::offset_of};

use bytemuck::{AnyBitPattern, Contiguous, Pod, Zeroable};
use enumset::EnumSetType;
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

#[derive(EnumSetType, Debug, Hash)]
pub enum GgmlType {
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

impl Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Q4_0 => write!(f, "Q40"),
            Self::Q4_1 => write!(f, "Q41"),
            Self::Q5_0 => write!(f, "Q50"),
            Self::Q5_1 => write!(f, "Q51"),
            Self::Q8_0 => write!(f, "Q80"),
            Self::Q8_1 => write!(f, "Q81"),
            Self::Q2K => write!(f, "Q2k"),
            Self::Q3K => write!(f, "Q3k"),
            Self::Q4K => write!(f, "Q4k"),
            Self::Q5K => write!(f, "Q5k"),
            Self::Q6K => write!(f, "Q6k"),
            Self::Q8K => write!(f, "Q8k"),
        }
    }
}

impl TryFrom<u32> for GgmlType {
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

impl GgmlType {
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
    V1 = 1,
    V2 = 2,
    V3 = 3,
}

impl GgufVersion {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self, GgufReadError> {
        check_magic(reader)?;
        let version = read_le_u32(reader)?;
        Self::try_from(version)
    }

    fn write<W: std::io::Write>(&self, writer: &mut W) -> Result<(), GgufWriteError> {
        writer.write_all(&GGUF_MAGIC_BYTES)?;
        let version = *self as u32;
        write_le_u32(writer, version)
    }
}

#[test]
fn test_gguf_version_round_trip() {
    let versions = [GgufVersion::V1, GgufVersion::V2, GgufVersion::V3];
    for version in versions {
        let mut buf = Vec::new();
        version.write(&mut buf).unwrap();
        println!("buf: {buf:?}");
        let mut cursor = std::io::Cursor::new(buf);
        let read_version = GgufVersion::read(&mut cursor).unwrap();
        assert_eq!(version, read_version);
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

fn write_le_u8<W: std::io::Write>(writer: &mut W, value: u8) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_u16<W: std::io::Write>(writer: &mut W, value: u16) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_u32<W: std::io::Write>(writer: &mut W, value: u32) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_u64<W: std::io::Write>(writer: &mut W, value: u64) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_i8<W: std::io::Write>(writer: &mut W, value: i8) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_i16<W: std::io::Write>(writer: &mut W, value: i16) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_i32<W: std::io::Write>(writer: &mut W, value: i32) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_i64<W: std::io::Write>(writer: &mut W, value: i64) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_f32<W: std::io::Write>(writer: &mut W, value: f32) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_le_f64<W: std::io::Write>(writer: &mut W, value: f64) -> Result<(), GgufWriteError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_array_length<W: std::io::Write>(
    writer: &mut W,
    version: GgufVersion,
    value: usize,
) -> Result<(), GgufWriteError> {
    match version {
        GgufVersion::V1 => write_le_u32(writer, value as u32),
        GgufVersion::V2 | GgufVersion::V3 => write_le_u64(writer, value as u64),
    }
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

/// Error type for GGUF writing
#[derive(thiserror::Error, Debug)]
pub enum GgufWriteError {
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    #[error("unknown tensor {0}")]
    UnknownTensor(String),
    #[error("cannot write empty array (type cannot be determined)")]
    EmptyArray,
}

#[derive(Debug, PartialEq)]
pub struct GgufTensorMetadata {
    pub ty: GgmlType,
    pub shape: Box<[u32]>,
    pub offset: u64,
}

impl GgufTensorMetadata {
    pub fn read_tensor_bytes<R: std::io::Read + std::io::Seek>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Box<[u8]>, GgufReadError> {
        let size_in_bytes = self.byte_size()?;
        let mut raw_data = vec![0u8; size_in_bytes].into_boxed_slice();
        reader.seek(std::io::SeekFrom::Start(tensor_data_offset + self.offset))?;
        reader.read_exact(&mut raw_data)?;
        Ok(raw_data)
    }

    fn byte_size(&self) -> Result<usize, GgufReadError> {
        let tensor_elems = self.shape.iter().copied().product::<u32>() as usize;
        let block_size = self.ty.block_size();
        if !tensor_elems.is_multiple_of(block_size) {
            return Err(GgufReadError::InvalidTensorSize {
                tensor_elems,
                block_size,
            });
        }
        Ok((tensor_elems / block_size) * self.ty.block_allocation_size())
    }
}

#[derive(Debug, PartialEq)]
pub struct GgufMetadata {
    pub version: GgufVersion,
    pub metadata: FxHashMap<Box<str>, GgufValue>,
    pub tensor_infos: FxHashMap<Box<str>, GgufTensorMetadata>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read>(
    reader: &mut R,
    version: GgufVersion,
) -> Result<Box<str>, GgufReadError> {
    let len = read_array_length(reader, version)?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = bytes.last() {
        bytes.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&bytes).into())
}

fn write_string<W: std::io::Write>(
    writer: &mut W,
    value: &str,
    version: GgufVersion,
) -> Result<(), GgufWriteError> {
    let len = value.len();
    write_array_length(writer, version, len)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Contiguous)]
#[repr(u8)]
pub enum GgufMetadataValueType {
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

impl TryFrom<u32> for GgufMetadataValueType {
    type Error = InvalidValueType;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let as_u8 = value.try_into().map_err(|_| InvalidValueType(value))?;
        Self::from_integer(as_u8).ok_or(InvalidValueType(value))
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    pub fn value_type(&self) -> GgufMetadataValueType {
        match self {
            Self::U8(_) => GgufMetadataValueType::U8,
            Self::I8(_) => GgufMetadataValueType::I8,
            Self::U16(_) => GgufMetadataValueType::U16,
            Self::I16(_) => GgufMetadataValueType::I16,
            Self::U32(_) => GgufMetadataValueType::U32,
            Self::I32(_) => GgufMetadataValueType::I32,
            Self::U64(_) => GgufMetadataValueType::U64,
            Self::I64(_) => GgufMetadataValueType::I64,
            Self::F32(_) => GgufMetadataValueType::F32,
            Self::F64(_) => GgufMetadataValueType::F64,
            Self::Bool(_) => GgufMetadataValueType::Bool,
            Self::String(_) => GgufMetadataValueType::String,
            Self::Array(_) => GgufMetadataValueType::Array,
        }
    }

    fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: GgufMetadataValueType,
        version: GgufVersion,
    ) -> Result<Self, GgufReadError> {
        let v = match value_type {
            GgufMetadataValueType::U8 => Self::U8(read_le_u8(reader)?),
            GgufMetadataValueType::I8 => Self::I8(read_le_i8(reader)?),
            GgufMetadataValueType::U16 => Self::U16(read_le_u16(reader)?),
            GgufMetadataValueType::I16 => Self::I16(read_le_i16(reader)?),
            GgufMetadataValueType::U32 => Self::U32(read_le_u32(reader)?),
            GgufMetadataValueType::I32 => Self::I32(read_le_i32(reader)?),
            GgufMetadataValueType::U64 => Self::U64(read_le_u64(reader)?),
            GgufMetadataValueType::I64 => Self::I64(read_le_i64(reader)?),
            GgufMetadataValueType::F32 => Self::F32(read_le_f32(reader)?),
            GgufMetadataValueType::F64 => Self::F64(read_le_f64(reader)?),
            GgufMetadataValueType::Bool => match read_le_u8(reader)? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                _ => return Err(GgufReadError::InvalidBool),
            },
            GgufMetadataValueType::String => Self::String(read_string(reader, version)?),
            GgufMetadataValueType::Array => {
                let value_type = read_le_u32(reader)?;
                let value_type = GgufMetadataValueType::try_from(value_type)?;
                let len = read_array_length(reader, version)?;
                let vs: Result<Box<[_]>, _> = (0..len)
                    .map(|_| GgufValue::read(reader, value_type, version))
                    .collect();
                Self::Array(vs?)
            }
        };
        Ok(v)
    }

    fn write<W: std::io::Write>(
        &self,
        writer: &mut W,
        version: GgufVersion,
    ) -> Result<(), GgufWriteError> {
        match self {
            Self::U8(v) => write_le_u8(writer, *v),
            Self::I8(v) => write_le_i8(writer, *v),
            Self::U16(v) => write_le_u16(writer, *v),
            Self::I16(v) => write_le_i16(writer, *v),
            Self::U32(v) => write_le_u32(writer, *v),
            Self::I32(v) => write_le_i32(writer, *v),
            Self::U64(v) => write_le_u64(writer, *v),
            Self::I64(v) => write_le_i64(writer, *v),
            Self::F32(v) => write_le_f32(writer, *v),
            Self::F64(v) => write_le_f64(writer, *v),
            Self::Bool(v) => write_le_u8(writer, if *v { 1 } else { 0 }),
            Self::String(s) => write_string(writer, s, version),
            Self::Array(arr) => {
                let first = arr.first().ok_or(GgufWriteError::EmptyArray)?;
                let value_type = first.value_type();
                write_le_u32(writer, value_type as u32)?;
                let len = arr.len();
                write_array_length(writer, version, len)?;
                for v in arr.iter() {
                    v.write(writer, version)?;
                }
                Ok(())
            }
        }
    }
}

macro_rules! try_into_gguf_value {
    ($convert: ident, $variant:ident, $value:ty $(=> from $($from:ident)+)?) => {
        impl GgufValue {
            #[doc = concat!("Convert the GGUF value to a `", stringify!($value), "` if possible.")]
            pub fn $convert(&self) -> Result<$value, GgufReadError> {
                self.try_into()
            }
        }

        impl TryInto<$value> for GgufValue {
            type Error = GgufReadError;

            fn try_into(self) -> Result<$value, Self::Error> {
                match self {
                    GgufValue::$variant(v) => Ok(v),
                    $(
                        $(
                            GgufValue::$from(v) => Ok(v as $value),
                        )+
                    )?
                    _ => Err(GgufReadError::InvalidValueType(InvalidValueType(
                        self.value_type() as u32,
                    ))),
                }
            }
        }

        impl<'a> TryInto<$value> for &'a GgufValue {
            type Error = GgufReadError;

            fn try_into(self) -> Result<$value, Self::Error> {
                match self {
                    GgufValue::$variant(v) => Ok(v.clone()),
                    $(
                        $(
                            GgufValue::$from(v) => Ok(v.clone() as $value),
                        )+
                    )?
                    _ => Err(GgufReadError::InvalidValueType(InvalidValueType(
                        self.value_type() as u32,
                    ))),
                }
            }
        }
    };
}
try_into_gguf_value!(to_u8, U8, u8 => from U16 U32 U64 I8 I16 I32 I64);
try_into_gguf_value!(to_i8, I8, i8 => from I16 I32 I64 U8 U16 U32 U64);
try_into_gguf_value!(to_u16, U16, u16 => from U8 U32 U64 I8 I16 I32 I64);
try_into_gguf_value!(to_i16, I16, i16 => from I8 I32 I64 U8 U16 U32 U64);
try_into_gguf_value!(to_u32, U32, u32 => from U8 U16 U64 I8 I16 I32 I64);
try_into_gguf_value!(to_i32, I32, i32 => from I8 I16 I64 U8 U16 U32 U64);
try_into_gguf_value!(to_u64, U64, u64 => from U8 U16 U32 I8 I16 I32 I64);
try_into_gguf_value!(to_i64, I64, i64 => from I8 I16 I32 U8 U16 U32 U64);
try_into_gguf_value!(to_f32, F32, f32 => from F64);
try_into_gguf_value!(to_f64, F64, f64 => from F32);
try_into_gguf_value!(to_bool, Bool, bool);
try_into_gguf_value!(to_string, String, Box<str>);
try_into_gguf_value!(to_array, Array, Box<[GgufValue]>);

impl GgufMetadata {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self, GgufReadError> {
        let version = GgufVersion::read(reader)?;

        let tensor_count = read_array_length(reader, version)?;
        let metadata_kv_count = read_array_length(reader, version)?;

        let mut metadata = FxHashMap::default();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, version)?;
            let value_type = read_le_u32(reader)?;
            let value_type = GgufMetadataValueType::try_from(value_type)?;
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
            let ggml_dtype = GgmlType::try_from(ggml_dtype)?;
            let offset = read_le_u64(reader)?;
            tensor_infos.insert(
                tensor_name,
                GgufTensorMetadata {
                    shape,
                    offset,
                    ty: ggml_dtype,
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

    pub fn write<'a, W: std::io::Write + std::io::Seek>(
        &self,
        writer: &mut W,
        tensors: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> Result<(), GgufWriteError> {
        self.version.write(writer)?;

        // Write the tensor count
        write_array_length(writer, self.version, self.tensor_infos.len())?;
        // Write the metadata key-value count
        write_array_length(writer, self.version, self.metadata.len())?;

        // Write the metadata key-value pairs
        for (key, value) in &self.metadata {
            write_string(writer, key, self.version)?;
            let value_type = value.value_type();
            write_le_u32(writer, value_type as u32)?;
            value.write(writer, self.version)?;
        }

        // Write the tensor metadata
        for (tensor_name, tensor_metadata) in &self.tensor_infos {
            write_string(writer, tensor_name, self.version)?;
            let dimensions = tensor_metadata.shape.len() as u32;
            write_le_u32(writer, dimensions)?;
            for dim in tensor_metadata.shape.iter().rev() {
                write_array_length(writer, self.version, *dim as usize)?;
            }
            write_le_u32(writer, tensor_metadata.ty as u32)?;
            write_le_u64(writer, tensor_metadata.offset)?;
        }

        let tensor_data_offset = writer.stream_position()?;

        // Write the tensor data
        for (tensor_name, tensor_data) in tensors {
            let tensor_metadata = self
                .tensor_infos
                .get(tensor_name)
                .ok_or(GgufWriteError::UnknownTensor(tensor_name.to_string()))?;
            let tensor_data_offset = tensor_data_offset + tensor_metadata.offset;
            writer.seek(std::io::SeekFrom::Start(tensor_data_offset))?;

            // Write the tensor data
            writer.write_all(tensor_data)?;
        }

        Ok(())
    }
}

pub trait GgufBlock: Pod + Sized {
    const BLOCK_SIZE: usize;

    type Bytes: AsRef<[u8]> + Copy;
    type BytesF32: AsRef<[u8]> + Copy;
    type AsBytes: AsRef<[u8]> + Copy;
    type Dequantized: AsRef<[f32]> + Copy;

    /// The activation block type for integer dot product.
    /// For 32-element blocks (Q4_0, Q5_0, Q8_0), this is BlockQ8_0.
    /// For 256-element K-quant blocks (Q4_K, Q6_K), this is BlockQ8K.
    type ActivationBlock: Pod + Sized;

    fn finite(&self) -> bool;
    fn into_wgsl_bytes(self) -> Self::Bytes;
    fn into_wgsl_bytes_f32(self) -> Self::BytesF32;
    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self;

    fn dequantize(&self) -> Self::Dequantized;

    /// Compute integer dot product with a quantized activation block.
    ///
    /// This performs the dot product directly on quantized values and applies
    /// the scale factors at the end, avoiding the overhead of full dequantization.
    /// Returns the dot product result with scales already applied.
    fn vec_dot(&self, activation: &Self::ActivationBlock) -> f32;

    /// Quantize a slice of f32 values into an activation block.
    ///
    /// The slice must have exactly `BLOCK_SIZE` elements.
    fn quantize_activation(data: &[f32]) -> Self::ActivationBlock;
}

const Q4_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ4_0Wgsl {
    pub(crate) scale: half::f16,
    pub(crate) data: [u32; Q4_0_BLOCK_SIZE / 8],
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ4_0WgslF32 {
    pub(crate) scale: f32,
    pub(crate) data: [u32; Q4_0_BLOCK_SIZE / 8],
}

#[derive(Zeroable, Pod, Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) scale: half::f16,
    pub(crate) data: [u8; Q4_0_BLOCK_SIZE / 2],
}

impl BlockQ4_0 {
    pub const WEIGHTS_SIZE: usize = Q4_0_BLOCK_SIZE / 2;
    pub const BLOCK_SIZE: usize = Q4_0_BLOCK_SIZE;
}

impl GgufBlock for BlockQ4_0 {
    const BLOCK_SIZE: usize = Q4_0_BLOCK_SIZE;

    type Bytes = [u8; std::mem::size_of::<BlockQ4_0Wgsl>()];
    type BytesF32 = [u8; std::mem::size_of::<BlockQ4_0WgslF32>()];
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; Q4_0_BLOCK_SIZE];
    type ActivationBlock = BlockQ8_0;

    fn finite(&self) -> bool {
        self.scale.is_finite()
    }

    fn into_wgsl_bytes_f32(self) -> Self::BytesF32 {
        let mut bytes = [0; std::mem::size_of::<BlockQ4_0WgslF32>()];
        let scale_f32 = self.scale.to_f32();
        bytes[0..4].copy_from_slice(bytemuck::bytes_of(&scale_f32));
        bytes[4..].copy_from_slice(bytemuck::cast_slice(&self.data));
        bytes
    }

    fn into_wgsl_bytes(self) -> Self::Bytes {
        let mut bytes = [0; std::mem::size_of::<BlockQ4_0Wgsl>()];
        let scale_offset = offset_of!(BlockQ4_0Wgsl, scale);
        let scale_bytes = bytemuck::bytes_of(&self.scale);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let data_offset = offset_of!(BlockQ4_0Wgsl, data);
        let data_bytes = bytemuck::cast_slice(&self.data);
        bytes[data_offset..data_offset + data_bytes.len()].copy_from_slice(data_bytes);
        bytes
    }

    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self {
        let scale_offset = offset_of!(BlockQ4_0Wgsl, scale);
        let scale_bytes = &bytes[scale_offset..scale_offset + std::mem::size_of::<half::f16>()];
        let scale = *bytemuck::from_bytes(scale_bytes);
        let data_offset = offset_of!(BlockQ4_0Wgsl, data);
        let data_bytes =
            &bytes[data_offset..data_offset + std::mem::size_of::<[u32; Q4_0_BLOCK_SIZE / 8]>()];
        let data = *bytemuck::from_bytes(data_bytes);
        Self { scale, data }
    }

    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L255
    fn dequantize(&self) -> Self::Dequantized {
        const CENTER_FOUR_BIT: i8 = 8;

        let scale = self.scale.to_f32();
        let mut data = [0.0; Q4_0_BLOCK_SIZE];

        for (i, byte) in self.data.iter().enumerate() {
            let low_data = (byte & 0x0F) as i8 - CENTER_FOUR_BIT;
            let high_data = (byte >> 4) as i8 - CENTER_FOUR_BIT;

            let low_data = low_data as f32 * scale;
            let high_data = high_data as f32 * scale;

            data[i] = low_data;
            data[i + Q4_0_BLOCK_SIZE / 2] = high_data;
        }

        data
    }

    #[cfg(target_arch = "aarch64")]
    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        use std::arch::aarch64::*;
        const CENTER: i8 = 8;
        unsafe {
            // Load 16 packed Q4 bytes
            let q4_bytes = vld1q_u8(self.data.as_ptr());

            // Unpack low and high nibbles
            let mask_low = vdupq_n_u8(0x0F);
            let q4_lo_u8 = vandq_u8(q4_bytes, mask_low);
            let q4_hi_u8 = vshrq_n_u8(q4_bytes, 4);

            // Convert to signed and subtract center (8)
            let center = vdupq_n_s8(CENTER);
            let q4_lo = vsubq_s8(vreinterpretq_s8_u8(q4_lo_u8), center);
            let q4_hi = vsubq_s8(vreinterpretq_s8_u8(q4_hi_u8), center);

            // Load activation values (first 16 for low nibbles, next 16 for high nibbles)
            let y_lo = vld1q_s8(y.data.as_ptr());
            let y_hi = vld1q_s8(y.data.as_ptr().add(16));

            // Use vdotq_s32 for fast i8x4 dot products
            let acc = vdupq_n_s32(0);
            let acc = vdotq_s32(acc, q4_lo, y_lo);
            let acc = vdotq_s32(acc, q4_hi, y_hi);
            let sum = vaddvq_s32(acc);

            let scale = self.scale.to_f32() * y.scale.to_f32();
            (sum as f32) * scale
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        const CENTER: i8 = 8;
        let mut sum: i32 = 0;

        for i in 0..16 {
            let byte = self.data[i];
            let q4_lo = (byte & 0x0F) as i8 - CENTER;
            let q4_hi = (byte >> 4) as i8 - CENTER;

            sum += (q4_lo as i32) * (y.data[i] as i32);
            sum += (q4_hi as i32) * (y.data[i + 16] as i32);
        }

        let scale = self.scale.to_f32() * y.scale.to_f32();
        (sum as f32) * scale
    }

    fn quantize_activation(data: &[f32]) -> Self::ActivationBlock {
        let arr: &[f32; 32] = data.try_into().expect("data must have 32 elements");
        BlockQ8_0::quantize(arr)
    }
}

const Q5_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ5_0Wgsl {
    pub(crate) scale: half::f16,
    pub(crate) data_high_bits: [u32; (Q5_0_BLOCK_SIZE / 8) / 4],
    pub(crate) data_low_bits: [u32; (Q5_0_BLOCK_SIZE / 2) / 4],
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ5_0WgslF32 {
    pub(crate) scale: f32,
    pub(crate) data_high_bits: [u32; (Q5_0_BLOCK_SIZE / 8) / 4],
    pub(crate) data_low_bits: [u32; (Q5_0_BLOCK_SIZE / 2) / 4],
}

#[derive(Zeroable, Pod, Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) scale: half::f16,
    // The highest bit for each of the 5 bit values
    pub(crate) data_high_bits: [u8; Q5_0_BLOCK_SIZE / 8],
    // The low four bits for each of the 5 bit values
    pub(crate) data_low_bits: [u8; Q5_0_BLOCK_SIZE / 2],
}

impl BlockQ5_0 {
    pub const BLOCK_SIZE: usize = Q5_0_BLOCK_SIZE;
    pub const WEIGHTS_HIGH_BITS_SIZE: usize = Q5_0_BLOCK_SIZE / 8;
    pub const WEIGHTS_LOW_BITS_SIZE: usize = Q5_0_BLOCK_SIZE / 2;
}

impl GgufBlock for BlockQ5_0 {
    const BLOCK_SIZE: usize = Q5_0_BLOCK_SIZE;

    type Bytes = [u8; std::mem::size_of::<BlockQ5_0Wgsl>()];
    type BytesF32 = [u8; std::mem::size_of::<BlockQ5_0WgslF32>()];
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; Q5_0_BLOCK_SIZE];
    type ActivationBlock = BlockQ8_0;

    fn finite(&self) -> bool {
        self.scale.is_finite()
    }

    fn into_wgsl_bytes_f32(self) -> Self::BytesF32 {
        let mut bytes = [0; std::mem::size_of::<BlockQ5_0WgslF32>()];
        let scale_offset = offset_of!(BlockQ5_0WgslF32, scale);
        let scale_f32 = self.scale.to_f32();
        let scale_bytes = bytemuck::bytes_of(&scale_f32);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let data_high_bits_offset = offset_of!(BlockQ5_0WgslF32, data_high_bits);
        let data_high_bits_bytes = bytemuck::cast_slice(&self.data_high_bits);
        bytes[data_high_bits_offset..data_high_bits_offset + data_high_bits_bytes.len()]
            .copy_from_slice(data_high_bits_bytes);
        let data_low_bits_offset = offset_of!(BlockQ5_0WgslF32, data_low_bits);
        let data_low_bits_bytes = bytemuck::cast_slice(&self.data_low_bits);
        bytes[data_low_bits_offset..data_low_bits_offset + data_low_bits_bytes.len()]
            .copy_from_slice(data_low_bits_bytes);
        bytes
    }

    fn into_wgsl_bytes(self) -> Self::Bytes {
        let mut bytes = [0; std::mem::size_of::<BlockQ5_0Wgsl>()];
        let scale_offset = offset_of!(BlockQ5_0Wgsl, scale);
        let scale_bytes = bytemuck::bytes_of(&self.scale);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let data_high_bits_offset = offset_of!(BlockQ5_0Wgsl, data_high_bits);
        let data_high_bits_bytes = bytemuck::cast_slice(&self.data_high_bits);
        bytes[data_high_bits_offset..data_high_bits_offset + data_high_bits_bytes.len()]
            .copy_from_slice(data_high_bits_bytes);
        let data_low_bits_offset = offset_of!(BlockQ5_0Wgsl, data_low_bits);
        let data_low_bits_bytes = bytemuck::cast_slice(&self.data_low_bits);
        bytes[data_low_bits_offset..data_low_bits_offset + data_low_bits_bytes.len()]
            .copy_from_slice(data_low_bits_bytes);
        bytes
    }

    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self {
        let scale_offset = offset_of!(BlockQ5_0Wgsl, scale);
        let scale_bytes = &bytes[scale_offset..scale_offset + std::mem::size_of::<half::f16>()];
        let scale = *bytemuck::from_bytes(scale_bytes);
        let data_high_bits_offset = offset_of!(BlockQ5_0Wgsl, data_high_bits);
        let data_high_bits_bytes = &bytes[data_high_bits_offset
            ..data_high_bits_offset + std::mem::size_of::<[u8; Q5_0_BLOCK_SIZE / 8]>()];
        let data_high_bits = *bytemuck::from_bytes(data_high_bits_bytes);
        let data_low_bits_offset = offset_of!(BlockQ5_0Wgsl, data_low_bits);
        let data_low_bits_bytes = &bytes[data_low_bits_offset
            ..data_low_bits_offset + std::mem::size_of::<[u8; Q5_0_BLOCK_SIZE / 2]>()];
        let data_low_bits = *bytemuck::from_bytes(data_low_bits_bytes);
        Self {
            scale,
            data_high_bits,
            data_low_bits,
        }
    }

    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L296
    fn dequantize(&self) -> Self::Dequantized {
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

    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        const CENTER: i8 = 16;
        const FIFTH_BIT: u8 = 0x10;
        let high_bits: u32 = bytemuck::cast(self.data_high_bits);
        let mut sum: i32 = 0;

        for i in 0..16 {
            let byte = self.data_low_bits[i];
            let low = byte & 0x0F;
            let high = byte >> 4;

            let low_bit = ((high_bits >> i) << 4) as u8 & FIFTH_BIT;
            let high_bit = (high_bits >> (i + 12)) as u8 & FIFTH_BIT;

            let q5_lo = (low | low_bit) as i8 - CENTER;
            let q5_hi = (high | high_bit) as i8 - CENTER;

            sum += (q5_lo as i32) * (y.data[i] as i32);
            sum += (q5_hi as i32) * (y.data[i + 16] as i32);
        }

        let scale = self.scale.to_f32() * y.scale.to_f32();
        (sum as f32) * scale
    }

    fn quantize_activation(data: &[f32]) -> Self::ActivationBlock {
        let arr: &[f32; 32] = data.try_into().expect("data must have 32 elements");
        BlockQ8_0::quantize(arr)
    }
}

const Q8_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ8_0Wgsl {
    pub(crate) scale: half::f16,
    pub(crate) data: [u32; Q8_0_BLOCK_SIZE / 4],
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ8_0WgslF32 {
    pub(crate) scale: f32,
    pub(crate) data: [u32; Q8_0_BLOCK_SIZE / 4],
}

#[derive(Zeroable, Pod, Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) scale: half::f16,
    pub(crate) data: [i8; Q8_0_BLOCK_SIZE],
}

impl BlockQ8_0 {
    pub const BLOCK_SIZE: usize = Q8_0_BLOCK_SIZE;
    pub const WEIGHTS_SIZE: usize = Q8_0_BLOCK_SIZE;

    /// Quantize 32 f32 values into a Q8_0 block.
    ///
    /// This finds the maximum absolute value, computes a scale factor,
    /// and quantizes each value to an i8 in the range [-127, 127].
    pub fn quantize(data: &[f32; 32]) -> Self {
        // Find max absolute value for scale
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if max_abs != 0.0 { 127.0 / max_abs } else { 0.0 };

        let mut qs = [0i8; 32];
        for (i, &v) in data.iter().enumerate() {
            qs[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }

        Self {
            scale: half::f16::from_f32(scale),
            data: qs,
        }
    }
}

impl GgufBlock for BlockQ8_0 {
    const BLOCK_SIZE: usize = Q8_0_BLOCK_SIZE;

    type Bytes = [u8; std::mem::size_of::<BlockQ8_0Wgsl>()];
    type BytesF32 = [u8; std::mem::size_of::<BlockQ8_0WgslF32>()];
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; Q8_0_BLOCK_SIZE];
    type ActivationBlock = BlockQ8_0;

    fn finite(&self) -> bool {
        self.scale.is_finite()
    }

    fn into_wgsl_bytes_f32(self) -> Self::BytesF32 {
        let mut bytes = [0; std::mem::size_of::<BlockQ8_0WgslF32>()];
        let scale_offset = offset_of!(BlockQ8_0WgslF32, scale);
        let scale_f32 = self.scale.to_f32();
        let scale_bytes = bytemuck::bytes_of(&scale_f32);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let data_offset = offset_of!(BlockQ8_0WgslF32, data);
        let data_bytes = bytemuck::cast_slice(&self.data);
        bytes[data_offset..data_offset + data_bytes.len()].copy_from_slice(data_bytes);
        bytes
    }

    fn into_wgsl_bytes(self) -> Self::Bytes {
        let mut bytes = [0; std::mem::size_of::<BlockQ8_0Wgsl>()];
        let scale_offset = offset_of!(BlockQ8_0Wgsl, scale);
        let scale_bytes = bytemuck::bytes_of(&self.scale);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let data_offset = offset_of!(BlockQ8_0Wgsl, data);
        let data_bytes = bytemuck::cast_slice(&self.data);
        bytes[data_offset..data_offset + data_bytes.len()].copy_from_slice(data_bytes);
        bytes
    }

    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self {
        let scale_offset = offset_of!(BlockQ8_0Wgsl, scale);
        let scale_bytes = &bytes[scale_offset..scale_offset + std::mem::size_of::<half::f16>()];
        let scale = *bytemuck::from_bytes(scale_bytes);
        let data_offset = offset_of!(BlockQ8_0Wgsl, data);
        let data_bytes =
            &bytes[data_offset..data_offset + std::mem::size_of::<[i8; Q8_0_BLOCK_SIZE]>()];
        let data = *bytemuck::from_bytes(data_bytes);
        Self { scale, data }
    }

    // https://github.com/ggml-org/llama.cpp/blob/80a02aa8588ef167d616f76f1781b104c245ace0/ggml/src/ggml-quants.c#L349
    fn dequantize(&self) -> [f32; Q8_0_BLOCK_SIZE] {
        let scale = self.scale.to_f32();

        std::array::from_fn(|i| self.data[i] as f32 * scale)
    }

    #[cfg(target_arch = "aarch64")]
    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        use std::arch::aarch64::*;
        unsafe {
            // Load 32 i8 values from each block (2 NEON registers each)
            let x0 = vld1q_s8(self.data.as_ptr());
            let x1 = vld1q_s8(self.data.as_ptr().add(16));
            let y0 = vld1q_s8(y.data.as_ptr());
            let y1 = vld1q_s8(y.data.as_ptr().add(16));

            // Use vdotq_s32: computes 4 simultaneous i8x4 dot products per instruction
            // Each lane accumulates: a[i*4+0]*b[i*4+0] + a[i*4+1]*b[i*4+1] + a[i*4+2]*b[i*4+2] + a[i*4+3]*b[i*4+3]
            let acc = vdupq_n_s32(0);
            let acc = vdotq_s32(acc, x0, y0);
            let acc = vdotq_s32(acc, x1, y1);

            // Horizontal sum of 4 i32 lanes
            let sum = vaddvq_s32(acc);

            let scale = self.scale.to_f32() * y.scale.to_f32();
            (sum as f32) * scale
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        let mut sum: i32 = 0;
        for i in 0..32 {
            sum += (self.data[i] as i32) * (y.data[i] as i32);
        }
        let scale = self.scale.to_f32() * y.scale.to_f32();
        (sum as f32) * scale
    }

    fn quantize_activation(data: &[f32]) -> Self::ActivationBlock {
        let arr: &[f32; 32] = data.try_into().expect("data must have 32 elements");
        BlockQ8_0::quantize(arr)
    }
}

const K_BLOCK_SIZE: usize = 256;

/// Q8_K block type for quantized activations in K-quant dot products.
///
/// This is used to quantize f32 activations for integer dot product
/// with Q4_K and Q6_K weight blocks. Each block stores 256 i8 values
/// with a single f16 scale factor.
#[derive(Zeroable, Pod, Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct BlockQ8K {
    /// Scale factor for the block
    pub scale: half::f16,
    /// Quantized i8 values
    pub data: [i8; K_BLOCK_SIZE],
}

impl BlockQ8K {
    pub const BLOCK_SIZE: usize = K_BLOCK_SIZE;

    /// Quantize 256 f32 values into a Q8_K block.
    ///
    /// This finds the maximum absolute value, computes a scale factor,
    /// and quantizes each value to an i8 in the range [-127, 127].
    pub fn quantize(data: &[f32; K_BLOCK_SIZE]) -> Self {
        // Find max absolute value for scale
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if max_abs != 0.0 { 127.0 / max_abs } else { 0.0 };

        let mut qs = [0i8; K_BLOCK_SIZE];
        for (i, &v) in data.iter().enumerate() {
            qs[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }

        Self {
            scale: half::f16::from_f32(scale),
            data: qs,
        }
    }

    /// Dequantize the block back to f32 values.
    pub fn dequantize(&self) -> [f32; K_BLOCK_SIZE] {
        let scale = self.scale.to_f32();
        std::array::from_fn(|i| self.data[i] as f32 * scale)
    }
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4KWgsl {
    scale: half::f16,
    min: half::f16,
    scales: [u32; 12 / 4],
    data: [u32; (K_BLOCK_SIZE / 2) / 4],
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4KWgslF32 {
    scale: f32,
    min: f32,
    scales: [u32; 12 / 4],
    data: [u32; (K_BLOCK_SIZE / 2) / 4],
}

#[derive(Zeroable, Pod, Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct BlockQ4K {
    scale: half::f16,
    min: half::f16,
    scales: [u8; 12],
    data: [u8; K_BLOCK_SIZE / 2],
}

impl BlockQ4K {
    pub const BLOCK_SIZE: usize = K_BLOCK_SIZE;
    pub const SCALES_SIZE: usize = 12;
    pub const WEIGHTS_SIZE: usize = K_BLOCK_SIZE / 2;

    pub fn scale(&self) -> f32 {
        self.scale.to_f32()
    }

    pub fn min(&self) -> f32 {
        self.min.to_f32()
    }
}

impl GgufBlock for BlockQ4K {
    const BLOCK_SIZE: usize = K_BLOCK_SIZE;

    type Bytes = [u8; std::mem::size_of::<BlockQ4KWgsl>()];
    type BytesF32 = [u8; std::mem::size_of::<BlockQ4KWgslF32>()];
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; K_BLOCK_SIZE];
    type ActivationBlock = BlockQ8K;

    fn finite(&self) -> bool {
        self.scale.is_finite() && self.min.is_finite()
    }

    fn into_wgsl_bytes_f32(self) -> Self::BytesF32 {
        let mut bytes = [0; std::mem::size_of::<BlockQ4KWgslF32>()];
        let scale_offset = offset_of!(BlockQ4KWgslF32, scale);
        let scale_f32 = self.scale.to_f32();
        let scale_bytes = bytemuck::bytes_of(&scale_f32);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let min_offset = offset_of!(BlockQ4KWgslF32, min);
        let min_f32 = self.min.to_f32();
        let min_bytes = bytemuck::bytes_of(&min_f32);
        bytes[min_offset..min_offset + min_bytes.len()].copy_from_slice(min_bytes);
        let scales_offset = offset_of!(BlockQ4KWgslF32, scales);
        let scales_bytes = bytemuck::bytes_of(&self.scales);
        bytes[scales_offset..scales_offset + scales_bytes.len()].copy_from_slice(scales_bytes);
        let data_offset = offset_of!(BlockQ4KWgslF32, data);
        let data_bytes = bytemuck::cast_slice(&self.data);
        bytes[data_offset..data_offset + data_bytes.len()].copy_from_slice(data_bytes);
        bytes
    }

    fn into_wgsl_bytes(self) -> Self::Bytes {
        let mut bytes = [0; std::mem::size_of::<BlockQ4KWgsl>()];
        let scale_offset = offset_of!(BlockQ4KWgsl, scale);
        let scale_bytes = bytemuck::bytes_of(&self.scale);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        let min_offset = offset_of!(BlockQ4KWgsl, min);
        let min_bytes = bytemuck::bytes_of(&self.min);
        bytes[min_offset..min_offset + min_bytes.len()].copy_from_slice(min_bytes);
        let scales_offset = offset_of!(BlockQ4KWgsl, scales);
        let scales_bytes = bytemuck::cast_slice(&self.scales);
        bytes[scales_offset..scales_offset + scales_bytes.len()].copy_from_slice(scales_bytes);
        let data_offset = offset_of!(BlockQ4KWgsl, data);
        let data_bytes = bytemuck::cast_slice(&self.data);
        bytes[data_offset..data_offset + data_bytes.len()].copy_from_slice(data_bytes);
        bytes
    }

    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self {
        let scale_offset = offset_of!(BlockQ4KWgsl, scale);
        let scale_bytes = &bytes[scale_offset..scale_offset + std::mem::size_of::<half::f16>()];
        let scale = *bytemuck::from_bytes(scale_bytes);
        let min_offset = offset_of!(BlockQ4KWgsl, min);
        let min_bytes = &bytes[min_offset..min_offset + std::mem::size_of::<half::f16>()];
        let min = *bytemuck::from_bytes(min_bytes);
        let scales_offset = offset_of!(BlockQ4KWgsl, scales);
        let scales_bytes = &bytes[scales_offset..scales_offset + std::mem::size_of::<[u8; 12]>()];
        let scales = *bytemuck::from_bytes(scales_bytes);
        let data_offset = offset_of!(BlockQ4KWgsl, data);
        let data_bytes =
            &bytes[data_offset..data_offset + std::mem::size_of::<[u32; K_BLOCK_SIZE / 2 / 4]>()];
        let data = *bytemuck::from_bytes(data_bytes);
        Self {
            scale,
            min,
            scales,
            data,
        }
    }

    fn dequantize(&self) -> [f32; K_BLOCK_SIZE] {
        let weights = &self.data;
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

    #[cfg(target_arch = "aarch64")]
    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        use std::arch::aarch64::*;

        let super_scale = self.scale.to_f32() * y.scale.to_f32();
        let super_min = self.min.to_f32() * y.scale.to_f32();

        // Extract scales and offsets (same as dequantize logic)
        let scales_bytes = bytemuck::cast_slice(&self.scales);
        let (first_scales, first_offset) = first_scales_min_k4(scales_bytes);
        let (second_scales, second_offset) = second_scales_min_k4(scales_bytes);
        let scales: [u8; 8] = bytemuck::cast([first_scales, second_scales]);
        let offsets: [u8; 8] = bytemuck::cast([first_offset, second_offset]);

        unsafe {
            let mask_low = vdupq_n_u8(0x0F);

            let mut scale_dot_sum = 0.0f32;
            let mut offset_y_sum = 0.0f32;
            let mut pair_index = 0;

            // Vector of ones for computing activation sums via dot product
            let ones = vdupq_n_s8(1);

            for chunk_index in (0..128).step_by(32) {
                let out_chunk_index = chunk_index * 2;

                // Load 32 weight bytes as two 16-byte vectors
                let w0 = vld1q_u8(self.data.as_ptr().add(chunk_index));
                let w1 = vld1q_u8(self.data.as_ptr().add(chunk_index + 16));

                // Extract low and high nibbles (values 0-15, fit in signed i8)
                let q_lo_0 = vreinterpretq_s8_u8(vandq_u8(w0, mask_low));
                let q_lo_1 = vreinterpretq_s8_u8(vandq_u8(w1, mask_low));
                let q_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                let q_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));

                // Load activation values
                let y_lo_0 = vld1q_s8(y.data.as_ptr().add(out_chunk_index));
                let y_lo_1 = vld1q_s8(y.data.as_ptr().add(out_chunk_index + 16));
                let y_hi_0 = vld1q_s8(y.data.as_ptr().add(out_chunk_index + 32));
                let y_hi_1 = vld1q_s8(y.data.as_ptr().add(out_chunk_index + 48));

                // Low nibbles integer dot product using vdotq_s32
                let acc_lo = vdupq_n_s32(0);
                let acc_lo = vdotq_s32(acc_lo, q_lo_0, y_lo_0);
                let acc_lo = vdotq_s32(acc_lo, q_lo_1, y_lo_1);
                let int_dot_lo = vaddvq_s32(acc_lo);

                // High nibbles integer dot product
                let acc_hi = vdupq_n_s32(0);
                let acc_hi = vdotq_s32(acc_hi, q_hi_0, y_hi_0);
                let acc_hi = vdotq_s32(acc_hi, q_hi_1, y_hi_1);
                let int_dot_hi = vaddvq_s32(acc_hi);

                // Sum of y values for offset correction using vdotq_s32 with ones
                let y_sum_acc_lo = vdupq_n_s32(0);
                let y_sum_acc_lo = vdotq_s32(y_sum_acc_lo, y_lo_0, ones);
                let y_sum_acc_lo = vdotq_s32(y_sum_acc_lo, y_lo_1, ones);
                let y_sum_lo = vaddvq_s32(y_sum_acc_lo);

                let y_sum_acc_hi = vdupq_n_s32(0);
                let y_sum_acc_hi = vdotq_s32(y_sum_acc_hi, y_hi_0, ones);
                let y_sum_acc_hi = vdotq_s32(y_sum_acc_hi, y_hi_1, ones);
                let y_sum_hi = vaddvq_s32(y_sum_acc_hi);

                // Apply group scales and accumulate
                let low_scale = scales[pair_index] as f32;
                let low_offset = offsets[pair_index] as f32;
                pair_index += 1;

                let high_scale = scales[pair_index] as f32;
                let high_offset = offsets[pair_index] as f32;
                pair_index += 1;

                scale_dot_sum += low_scale * (int_dot_lo as f32) + high_scale * (int_dot_hi as f32);
                offset_y_sum += low_offset * (y_sum_lo as f32) + high_offset * (y_sum_hi as f32);
            }

            super_scale * scale_dot_sum - super_min * offset_y_sum
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        let super_scale = self.scale.to_f32() * y.scale.to_f32();
        let super_min = self.min.to_f32() * y.scale.to_f32();

        // Extract scales and offsets (same as dequantize logic)
        let scales_bytes = bytemuck::cast_slice(&self.scales);
        let (first_scales, first_offset) = first_scales_min_k4(scales_bytes);
        let (second_scales, second_offset) = second_scales_min_k4(scales_bytes);
        let scales: [u8; 8] = bytemuck::cast([first_scales, second_scales]);
        let offsets: [u8; 8] = bytemuck::cast([first_offset, second_offset]);

        let mut sum = 0.0f32;
        let mut pair_index = 0;

        for chunk_index in (0..128).step_by(32) {
            let out_chunk_index = chunk_index * 2;

            let low_scale = scales[pair_index] as f32 * super_scale;
            let low_offset = offsets[pair_index] as f32 * super_min;
            pair_index += 1;

            let high_scale = scales[pair_index] as f32 * super_scale;
            let high_offset = offsets[pair_index] as f32 * super_min;
            pair_index += 1;

            for offset in 0..32 {
                let weight_byte = self.data[chunk_index + offset];
                let q_lo = (weight_byte & 0xF) as f32;
                let q_hi = (weight_byte >> 4) as f32;

                let w_lo = low_scale * q_lo - low_offset;
                let w_hi = high_scale * q_hi - high_offset;

                let y_lo = y.data[out_chunk_index + offset] as f32;
                let y_hi = y.data[out_chunk_index + offset + 32] as f32;

                sum += w_lo * y_lo + w_hi * y_hi;
            }
        }

        sum
    }

    fn quantize_activation(data: &[f32]) -> Self::ActivationBlock {
        let arr: &[f32; K_BLOCK_SIZE] = data.try_into().expect("data must have 256 elements");
        BlockQ8K::quantize(arr)
    }
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ6KWgsl {
    pub(crate) data_low_bits: [u32; (K_BLOCK_SIZE / 2) / 4],
    pub(crate) data_high_bits: [u32; (K_BLOCK_SIZE / 4) / 4],
    pub(crate) scales: [u32; (K_BLOCK_SIZE / 16) / 4],
    pub(crate) scale: half::f16,
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ6KWgslF32 {
    pub(crate) data_low_bits: [u32; (K_BLOCK_SIZE / 2) / 4],
    pub(crate) data_high_bits: [u32; (K_BLOCK_SIZE / 4) / 4],
    pub(crate) scales: [u32; (K_BLOCK_SIZE / 16) / 4],
    pub(crate) scale: f32,
}

#[derive(Zeroable, Pod, Clone, Copy, PartialEq, Debug)]
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
    pub const BLOCK_SIZE: usize = K_BLOCK_SIZE;
    pub const SCALES_SIZE: usize = K_BLOCK_SIZE / 16;
    pub const WEIGHTS_LOW_BITS_SIZE: usize = K_BLOCK_SIZE / 2;
    pub const WEIGHTS_HIGH_BITS_SIZE: usize = K_BLOCK_SIZE / 4;
}

impl GgufBlock for BlockQ6K {
    const BLOCK_SIZE: usize = K_BLOCK_SIZE;

    type Bytes = [u8; std::mem::size_of::<BlockQ6KWgsl>()];
    type BytesF32 = [u8; std::mem::size_of::<BlockQ6KWgslF32>()];
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; K_BLOCK_SIZE];
    type ActivationBlock = BlockQ8K;

    fn finite(&self) -> bool {
        self.scale.is_finite()
    }

    fn into_wgsl_bytes_f32(self) -> Self::BytesF32 {
        let mut bytes = [0; std::mem::size_of::<BlockQ6KWgslF32>()];
        let data_low_bits_offset = offset_of!(BlockQ6KWgslF32, data_low_bits);
        let data_low_bits_bytes = bytemuck::cast_slice(&self.data_low_bits);
        bytes[data_low_bits_offset..data_low_bits_offset + data_low_bits_bytes.len()]
            .copy_from_slice(data_low_bits_bytes);
        let data_high_bits_offset = offset_of!(BlockQ6KWgslF32, data_high_bits);
        let data_high_bits_bytes = bytemuck::cast_slice(&self.data_high_bits);
        bytes[data_high_bits_offset..data_high_bits_offset + data_high_bits_bytes.len()]
            .copy_from_slice(data_high_bits_bytes);
        let scales_offset = offset_of!(BlockQ6KWgslF32, scales);
        let scales_bytes = bytemuck::bytes_of(&self.scales);
        bytes[scales_offset..scales_offset + scales_bytes.len()].copy_from_slice(scales_bytes);
        let scale_offset = offset_of!(BlockQ6KWgslF32, scale);
        let scale_f32 = self.scale.to_f32();
        let scale_bytes = bytemuck::bytes_of(&scale_f32);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        bytes
    }

    fn into_wgsl_bytes(self) -> Self::Bytes {
        let mut bytes = [0; std::mem::size_of::<BlockQ6KWgsl>()];
        let data_low_bits_offset = offset_of!(BlockQ6KWgsl, data_low_bits);
        let data_low_bits_bytes = bytemuck::cast_slice(&self.data_low_bits);
        bytes[data_low_bits_offset..data_low_bits_offset + data_low_bits_bytes.len()]
            .copy_from_slice(data_low_bits_bytes);
        let data_high_bits_offset = offset_of!(BlockQ6KWgsl, data_high_bits);
        let data_high_bits_bytes = bytemuck::cast_slice(&self.data_high_bits);
        bytes[data_high_bits_offset..data_high_bits_offset + data_high_bits_bytes.len()]
            .copy_from_slice(data_high_bits_bytes);
        let scales_offset = offset_of!(BlockQ6KWgsl, scales);
        let scales_bytes = bytemuck::cast_slice(&self.scales);
        bytes[scales_offset..scales_offset + scales_bytes.len()].copy_from_slice(scales_bytes);
        let scale_offset = offset_of!(BlockQ6KWgsl, scale);
        let scale_bytes = bytemuck::bytes_of(&self.scale);
        bytes[scale_offset..scale_offset + scale_bytes.len()].copy_from_slice(scale_bytes);
        bytes
    }

    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self {
        let data_low_bits_offset = offset_of!(BlockQ6KWgsl, data_low_bits);
        let data_low_bits_bytes = &bytes[data_low_bits_offset
            ..data_low_bits_offset + std::mem::size_of::<[u8; K_BLOCK_SIZE / 2]>()];
        let data_low_bits = *bytemuck::from_bytes(data_low_bits_bytes);
        let data_high_bits_offset = offset_of!(BlockQ6KWgsl, data_high_bits);
        let data_high_bits_bytes = &bytes[data_high_bits_offset
            ..data_high_bits_offset + std::mem::size_of::<[u8; K_BLOCK_SIZE / 4]>()];
        let data_high_bits = *bytemuck::from_bytes(data_high_bits_bytes);
        let scales_offset = offset_of!(BlockQ6KWgsl, scales);
        let scales_bytes =
            &bytes[scales_offset..scales_offset + std::mem::size_of::<[i8; K_BLOCK_SIZE / 16]>()];
        let scales = *bytemuck::from_bytes(scales_bytes);
        let scale_offset = offset_of!(BlockQ6KWgsl, scale);
        let scale_bytes = &bytes[scale_offset..scale_offset + std::mem::size_of::<half::f16>()];
        let scale = *bytemuck::from_bytes(scale_bytes);
        Self {
            data_low_bits,
            data_high_bits,
            scales,
            scale,
        }
    }

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

    fn vec_dot(&self, y: &Self::ActivationBlock) -> f32 {
        const CENTER: i8 = 32;
        const TWO_BITS: u8 = 0b11;
        const FOUR_BITS: u8 = 0b1111;

        let scale = self.scale.to_f32() * y.scale.to_f32();
        let mut sum = 0.0f32;

        for chunk_index in 0..2 {
            let output_index = chunk_index * 128;
            let lower_index = chunk_index * 64;
            let high_index = chunk_index * 32;
            let base_scale_index = chunk_index * 8;

            for high_byte_index in 0..32 {
                let sc_idx = base_scale_index + high_byte_index / 16;
                let high_byte = self.data_high_bits[high_index + high_byte_index];
                let first_low = self.data_low_bits[lower_index + high_byte_index];
                let second_low = self.data_low_bits[lower_index + high_byte_index + 32];

                // Reconstruct 6-bit values
                let q1 = ((high_byte & TWO_BITS) << 4 | (first_low & FOUR_BITS)) as i8 - CENTER;
                let q2 = (((high_byte >> 2) & TWO_BITS) << 4 | (second_low & FOUR_BITS)) as i8 - CENTER;
                let q3 = (((high_byte >> 4) & TWO_BITS) << 4 | (first_low >> 4)) as i8 - CENTER;
                let q4 = (((high_byte >> 6) & TWO_BITS) << 4 | (second_low >> 4)) as i8 - CENTER;

                let s1 = self.scales[sc_idx] as f32;
                let s2 = self.scales[sc_idx + 2] as f32;
                let s3 = self.scales[sc_idx + 4] as f32;
                let s4 = self.scales[sc_idx + 6] as f32;

                sum += scale * s1 * (q1 as f32) * (y.data[output_index + high_byte_index] as f32);
                sum += scale * s2 * (q2 as f32) * (y.data[output_index + high_byte_index + 32] as f32);
                sum += scale * s3 * (q3 as f32) * (y.data[output_index + high_byte_index + 64] as f32);
                sum += scale * s4 * (q4 as f32) * (y.data[output_index + high_byte_index + 96] as f32);
            }
        }

        sum
    }

    fn quantize_activation(data: &[f32]) -> Self::ActivationBlock {
        let arr: &[f32; K_BLOCK_SIZE] = data.try_into().expect("data must have 256 elements");
        BlockQ8K::quantize(arr)
    }
}

const SIX_BITS_MASK: u32 = 0b0011_1111_0011_1111_0011_1111_0011_1111;
const MSB_TWO_BITS_MASK: u32 = 0b1100_0000_1100_0000_1100_0000_1100_0000;
const MSB_SCALES_MASK: u32 = 0b0000_1111_0000_1111_0000_1111_0000_1111;
const MSB_OFFSET_MASK: u32 = 0b1111_0000_1111_0000_1111_0000_1111_0000;

// Extracts this bit pattern. The first 6 bits of the first
// 4 bytes are the scales. The first 6 bits of the second 4
// bytes are the offsets.
//
// dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
// __000000|__111111|__222222|__333333|__000000|__111111
//
// mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
// __222222|__333333|________|________|________|________
fn first_scales_min_k4(packed_scales: &[u32]) -> (u32, u32) {
    let first_four_bytes = packed_scales[0];
    let middle_four_bytes = packed_scales[1];

    let first_scales = first_four_bytes & SIX_BITS_MASK;
    let first_offsets = middle_four_bytes & SIX_BITS_MASK;

    (first_scales, first_offsets)
}

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

#[cfg(test)]
mod vec_dot_tests {
    use super::*;

    #[test]
    fn test_q8_0_quantize_roundtrip() {
        let data: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
        let block = BlockQ8_0::quantize(&data);
        let dequant = block.dequantize();
        for (a, b) in data.iter().zip(&dequant) {
            assert!(
                (a - b).abs() < 0.02,
                "Quantization roundtrip error: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_q8_0_vec_dot() {
        // Create two Q8_0 blocks with known values
        let data_a: [f32; 32] = std::array::from_fn(|i| (i as f32) * 0.1);
        let data_b: [f32; 32] = std::array::from_fn(|i| ((i as f32) - 16.0) * 0.05);

        let block_a = BlockQ8_0::quantize(&data_a);
        let block_b = BlockQ8_0::quantize(&data_b);

        // Compute expected result using dequantized values
        let dequant_a = block_a.dequantize();
        let dequant_b = block_b.dequantize();
        let expected: f32 = dequant_a.iter().zip(&dequant_b).map(|(a, b)| a * b).sum();

        // Compute using vec_dot
        let actual = block_a.vec_dot(&block_b);

        // Allow small error due to quantization
        let tolerance = expected.abs().max(1.0) * 0.05;
        assert!(
            (actual - expected).abs() < tolerance,
            "vec_dot mismatch: expected {}, got {}, diff {}",
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    #[test]
    fn test_q8k_quantize_roundtrip() {
        let data: [f32; 256] = std::array::from_fn(|i| ((i as f32) - 128.0) * 0.01);
        let block = BlockQ8K::quantize(&data);
        let dequant = block.dequantize();
        for (a, b) in data.iter().zip(&dequant) {
            assert!(
                (a - b).abs() < 0.02,
                "Quantization roundtrip error: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_q4_k_vec_dot_vs_dequantize() {
        // Create a Q4_K block with known pattern
        let block = BlockQ4K {
            scale: half::f16::from_f32(0.1),
            min: half::f16::from_f32(0.05),
            scales: [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96],
            data: std::array::from_fn(|i| ((i * 3) % 256) as u8),
        };

        // Create Q8_K activation block
        let act_data: [f32; 256] = std::array::from_fn(|i| ((i as f32) - 128.0) * 0.01);
        let act_block = BlockQ8K::quantize(&act_data);

        // Compute expected using dequantize
        let dequant_weights = block.dequantize();
        let dequant_acts = act_block.dequantize();
        let expected: f32 = dequant_weights
            .iter()
            .zip(&dequant_acts)
            .map(|(w, a)| w * a)
            .sum();

        // Compute using vec_dot
        let actual = block.vec_dot(&act_block);

        // Allow for quantization error (Q4 has more error than Q8)
        let tolerance = expected.abs().max(1.0) * 0.10; // 10% tolerance for Q4_K
        assert!(
            (actual - expected).abs() < tolerance,
            "Q4_K vec_dot mismatch: expected {}, got {}, diff {} (tolerance {})",
            expected,
            actual,
            (actual - expected).abs(),
            tolerance
        );
    }
}

#[cfg(test)]
async fn tiny_llama() -> impl std::io::Read + std::io::Seek {
    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    std::io::Cursor::new(bytes)
}

#[cfg(test)]
#[tokio::test]
async fn test_round_trip_tiny_llama() {
    use std::{
        collections::HashMap,
        io::{Read, Seek},
    };

    // Load the model
    let mut reader = tiny_llama().await;
    // Copy the memory
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();

    // Reset the reader
    reader.seek(std::io::SeekFrom::Start(0)).unwrap();
    let metadata = GgufMetadata::read(&mut reader).unwrap();

    // Read the tensor bytes
    let mut tensors = HashMap::new();
    for (tensor_name, tensor_info) in metadata.tensor_infos.iter() {
        println!("{tensor_name}: {tensor_info:?}");
        let tensor_bytes = tensor_info
            .read_tensor_bytes(&mut reader, metadata.tensor_data_offset)
            .unwrap();
        tensors.insert(tensor_name.clone(), tensor_bytes);
    }

    println!("read {} tensors", tensors.len());

    // Write the model to a buffer
    let mut writer = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut writer);
        metadata
            .write(
                &mut cursor,
                tensors
                    .iter()
                    .map(|(name, bytes)| (&**name, bytes.as_ref())),
            )
            .unwrap();
    }
}

#[test]
fn test_round_trip_empty() {
    use std::collections::HashMap;
    // Reset the reader
    let metadata = GgufMetadata {
        tensor_data_offset: 32,
        tensor_infos: HashMap::default(),
        version: GgufVersion::V3,
        metadata: HashMap::default(),
    };

    // Write the model to a buffer
    let mut writer = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut writer);
        metadata.write(&mut cursor, []).unwrap();
    }

    println!("{writer:?}");

    // Read the model from the buffer
    let mut reader = std::io::Cursor::new(writer);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    assert_eq!(metadata.tensor_data_offset, 32);
    assert_eq!(metadata.tensor_infos.len(), 0);
    assert_eq!(metadata.version, GgufVersion::V3);
    assert_eq!(metadata.metadata.len(), 0);
}

#[cfg(test)]
#[tokio::test]
async fn test_load_tiny_llama() {
    use pretty_assertions::assert_eq;
    use std::io::Seek;

    let mut reader = tiny_llama().await;
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    reader.seek(std::io::SeekFrom::Start(0)).unwrap();
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let device = candle_core::Device::Cpu;
    for (name, candle_tensor) in candle_metadata.tensor_infos {
        let tensor = metadata.tensor_infos.get(&*name).unwrap();
        println!("{name}: {tensor:?}");
        let tensor_bytes = tensor
            .read_tensor_bytes(&mut reader, metadata.tensor_data_offset)
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
        match tensor.ty {
            GgmlType::Q4_0 => {
                let blocks: &[BlockQ4_0] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in blocks
                    .iter()
                    .zip(candle_tensor_data.chunks(Q4_0_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {dequantized:?}");
                            println!("candle: {candle_block:?}");
                            assert_eq!(dequantized, candle_block);
                        }
                    }
                }
            }
            GgmlType::Q5_0 => {
                let blocks: &[BlockQ5_0] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in blocks
                    .iter()
                    .zip(candle_tensor_data.chunks(Q5_0_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {dequantized:?}");
                            println!("candle: {candle_block:?}");
                            assert_eq!(dequantized, candle_block);
                        }
                    }
                }
            }
            GgmlType::Q8_0 => {
                let blocks: &[BlockQ8_0] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in blocks
                    .iter()
                    .zip(candle_tensor_data.chunks(Q8_0_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {dequantized:?}");
                            println!("candle: {candle_block:?}");
                            assert_eq!(dequantized, candle_block);
                        }
                    }
                }
            }
            GgmlType::Q4K => {
                let blocks: &[BlockQ4K] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in
                    blocks.iter().zip(candle_tensor_data.chunks(K_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {dequantized:?}");
                            println!("candle: {candle_block:?}");
                            assert_eq!(dequantized, candle_block);
                        }
                    }
                }
            }
            GgmlType::Q6K => {
                let blocks: &[BlockQ6K] = bytemuck::cast_slice(&tensor_bytes);
                for (block, candle_block) in
                    blocks.iter().zip(candle_tensor_data.chunks(K_BLOCK_SIZE))
                {
                    let dequantized = block.dequantize();
                    for (a, b) in dequantized.iter().zip(candle_block) {
                        if (a - b).abs() > 1e-6 {
                            println!("ours: {dequantized:?}");
                            println!("candle: {candle_block:?}");
                            assert_eq!(dequantized, candle_block);
                        }
                    }
                }
            }
            GgmlType::F32 => {
                let blocks: &[f32] = bytemuck::cast_slice(&tensor_bytes);
                for (a, b) in blocks.iter().zip(candle_tensor_data) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
            _ => todo!(),
        }
    }
}

#[test]
fn test_q8_0_quantize_roundtrip() {
    // Test that quantizing and dequantizing f32 values gives reasonable results
    let data: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = BlockQ8_0::quantize(&data);
    let dequant = block.dequantize();
    for (a, b) in data.iter().zip(&dequant) {
        // Quantization to 8-bit introduces ~1% error
        assert!((a - b).abs() < 0.02, "Mismatch: original={}, dequantized={}", a, b);
    }
}

#[test]
fn test_q8_k_quantize_roundtrip() {
    // Test that quantizing and dequantizing f32 values gives reasonable results
    let data: [f32; K_BLOCK_SIZE] = std::array::from_fn(|i| (i as f32 - 128.0) * 0.01);
    let block = BlockQ8K::quantize(&data);
    let dequant = block.dequantize();
    for (a, b) in data.iter().zip(&dequant) {
        // Quantization to 8-bit introduces ~1% error
        assert!((a - b).abs() < 0.02, "Mismatch: original={}, dequantized={}", a, b);
    }
}

#[test]
fn test_vec_dot_q8_0_matches_dequantize() {
    // Test that vec_dot gives the same result as dequantize-then-dot
    let weight_data: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.05);
    let activation_data: [f32; 32] = std::array::from_fn(|i| ((i as f32) * 0.1).sin());

    // Create weight block by quantizing the weight data
    let weight_block = BlockQ8_0::quantize(&weight_data);
    let activation_block = BlockQ8_0::quantize(&activation_data);

    // Compute vec_dot
    let vec_dot_result = weight_block.vec_dot(&activation_block);

    // Compute dequantize-then-dot
    let weight_dequant = weight_block.dequantize();
    let act_dequant = activation_block.dequantize();
    let dequant_dot: f32 = weight_dequant.iter().zip(&act_dequant).map(|(w, a)| w * a).sum();

    // Results should be very close (within quantization error)
    assert!(
        (vec_dot_result - dequant_dot).abs() < 0.5,
        "vec_dot={}, dequant_dot={}, diff={}",
        vec_dot_result, dequant_dot, (vec_dot_result - dequant_dot).abs()
    );
}

#[test]
fn test_vec_dot_q4_0_matches_dequantize() {
    // Create a Q4_0 block with known values
    let mut raw_bytes = Vec::new();
    let scale_f16 = half::f16::from_f32(0.1);
    raw_bytes.extend_from_slice(&scale_f16.to_le_bytes());
    // Pack 32 4-bit values: indices 0-15 in low nibble, 16-31 in high nibble
    for i in 0..16 {
        let low_val = (i % 16) as u8;  // indices 0-15
        let high_val = ((i + 8) % 16) as u8; // indices 16-31
        let packed = low_val | (high_val << 4);
        raw_bytes.push(packed);
    }
    let weight_block: BlockQ4_0 = *bytemuck::from_bytes(&raw_bytes);

    // Create activation block
    let activation_data: [f32; 32] = std::array::from_fn(|i| ((i as f32) * 0.2).cos());
    let activation_block = BlockQ8_0::quantize(&activation_data);

    // Compute vec_dot
    let vec_dot_result = weight_block.vec_dot(&activation_block);

    // Compute dequantize-then-dot
    let weight_dequant = weight_block.dequantize();
    let act_dequant = activation_block.dequantize();
    let dequant_dot: f32 = weight_dequant.iter().zip(&act_dequant).map(|(w, a)| w * a).sum();

    // Results should be close (within quantization error)
    assert!(
        (vec_dot_result - dequant_dot).abs() < 1.0,
        "vec_dot={}, dequant_dot={}, diff={}",
        vec_dot_result, dequant_dot, (vec_dot_result - dequant_dot).abs()
    );
}
