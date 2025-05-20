//! Support for [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files

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
        if tensor_elems % block_size != 0 {
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
                let value_type = arr[0].value_type();
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
    type AsBytes: AsRef<[u8]> + Copy;
    type Dequantized: AsRef<[f32]> + Copy;

    fn finite(&self) -> bool;
    fn into_wgsl_bytes(self) -> Self::Bytes;
    fn from_wgsl_bytes(bytes: Self::Bytes) -> Self;

    fn dequantize(&self) -> Self::Dequantized;
}

const Q4_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ4_0Wgsl {
    pub(crate) scale: half::f16,
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
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; Q4_0_BLOCK_SIZE];

    fn finite(&self) -> bool {
        self.scale.is_finite()
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
}

const Q5_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ5_0Wgsl {
    pub(crate) scale: half::f16,
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
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; Q5_0_BLOCK_SIZE];

    fn finite(&self) -> bool {
        self.scale.is_finite()
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
}

const Q8_0_BLOCK_SIZE: usize = 32;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct BlockQ8_0Wgsl {
    pub(crate) scale: half::f16,
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
}

impl GgufBlock for BlockQ8_0 {
    const BLOCK_SIZE: usize = Q8_0_BLOCK_SIZE;

    type Bytes = [u8; std::mem::size_of::<BlockQ8_0Wgsl>()];
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; Q8_0_BLOCK_SIZE];

    fn finite(&self) -> bool {
        self.scale.is_finite()
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
}

const K_BLOCK_SIZE: usize = 256;

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4KWgsl {
    scale: half::f16,
    min: half::f16,
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
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; K_BLOCK_SIZE];

    fn finite(&self) -> bool {
        self.scale.is_finite() && self.min.is_finite()
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
}

#[derive(AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct BlockQ6KWgsl {
    pub(crate) data_low_bits: [u32; (K_BLOCK_SIZE / 2) / 4],
    pub(crate) data_high_bits: [u32; (K_BLOCK_SIZE / 4) / 4],
    pub(crate) scales: [u32; (K_BLOCK_SIZE / 16) / 4],
    pub(crate) scale: half::f16,
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
    type AsBytes = [u8; std::mem::size_of::<Self>()];
    type Dequantized = [f32; K_BLOCK_SIZE];

    fn finite(&self) -> bool {
        self.scale.is_finite()
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
        println!("{}: {:?}", tensor_name, tensor_info);
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

    // Read the data again and assert everything is the same
    let mut reader = std::io::Cursor::new(writer);
    let new_metadata = GgufMetadata::read(&mut reader).unwrap();
    assert_eq!(new_metadata, metadata);

    // Read the tensor bytes
    let mut new_tensors = HashMap::new();
    for (tensor_name, tensor_info) in new_metadata.tensor_infos.iter() {
        println!("{}: {:?}", tensor_name, tensor_info);
        let tensor_bytes = tensor_info
            .read_tensor_bytes(&mut reader, new_metadata.tensor_data_offset)
            .unwrap();
        new_tensors.insert(tensor_name.clone(), tensor_bytes);
    }

    // Assert all the tensors are the same
    for (name, tensor) in tensors {
        let new_tensor = new_tensors.get(&name).unwrap();
        assert_eq!(&tensor, new_tensor);
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

    println!("{:?}", writer);

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
