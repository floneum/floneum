use std::fmt::Display;

use crate::DataTypeEnum;

#[derive(Clone, Debug)]
pub enum KernelGlobalSpace {
    Workgroup,
}

impl Display for KernelGlobalSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelGlobalSpace::Workgroup => write!(f, "workgroup"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelGlobal {
    id: u32,
    space: KernelGlobalSpace,
    ty: KernelGlobalType,
}

impl KernelGlobal {
    pub fn new(id: u32, space: KernelGlobalSpace, ty: KernelGlobalType) -> Self {
        Self { id, space, ty }
    }

    pub fn global_definition(&self) -> String {
        match &self.ty {
            KernelGlobalType::Array(array) => {
                let dtype = array.datatype.wgsl_type();
                let size = &array.size;
                let space = &self.space;
                format!("var<{space}> {self}: array<{dtype}, {size}>;\n")
            }
            KernelGlobalType::Vector(vector) => {
                let dtype = vector.datatype.wgsl_type();
                let size = &vector.size;
                let space = &self.space;
                format!("var<{space}> {self}: vec{size}<{dtype}>;\n")
            }
            KernelGlobalType::Matrix(matrix) => {
                let dtype = matrix.datatype.wgsl_type();
                let size = &matrix.size;
                let space = &self.space;
                format!(
                    "var<{space}> {self}: mat{}x{}<{dtype}>;\n",
                    size[0], size[1]
                )
            }
            KernelGlobalType::Value(ty) => {
                let space = &self.space;
                format!("var<{space}> {self}: {ty};\n")
            }
        }
    }
}

impl Display for KernelGlobal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g_{}", self.id)
    }
}

#[derive(Clone, Debug)]
pub enum KernelGlobalType {
    Array(ArrayType),
    Vector(VectorType),
    Matrix(MatrixType),
    Value(DataTypeEnum),
}

impl From<DataTypeEnum> for KernelGlobalType {
    fn from(value: DataTypeEnum) -> Self {
        KernelGlobalType::Value(value)
    }
}

impl From<ArrayType> for KernelGlobalType {
    fn from(value: ArrayType) -> Self {
        KernelGlobalType::Array(value)
    }
}

impl From<VectorType> for KernelGlobalType {
    fn from(value: VectorType) -> Self {
        KernelGlobalType::Vector(value)
    }
}

impl From<MatrixType> for KernelGlobalType {
    fn from(value: MatrixType) -> Self {
        KernelGlobalType::Matrix(value)
    }
}

impl KernelGlobalType {
    pub fn wgsl_type(&self) -> String {
        match self {
            KernelGlobalType::Array(array) => {
                let dtype = array.datatype.wgsl_type();
                let size = &array.size;
                format!("array<{dtype}, {size}>")
            }
            KernelGlobalType::Vector(vector) => {
                let dtype = vector.datatype.wgsl_type();
                let size = &vector.size;
                format!("vec{size}<{dtype}>")
            }
            KernelGlobalType::Matrix(matrix) => {
                let dtype = matrix.datatype.wgsl_type();
                let size = &matrix.size;
                format!("mat{}x{}<{dtype}>", size[0], size[1])
            }
            KernelGlobalType::Value(ty) => ty.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArrayType {
    size: String,
    datatype: Box<KernelGlobalType>,
}

impl ArrayType {
    pub fn new(size: String, datatype: impl Into<KernelGlobalType>) -> Self {
        Self {
            size,
            datatype: Box::new(datatype.into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorType {
    size: String,
    datatype: Box<KernelGlobalType>,
}

impl VectorType {
    pub fn new(size: String, datatype: impl Into<KernelGlobalType>) -> Self {
        Self {
            size,
            datatype: Box::new(datatype.into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatrixType {
    size: [String; 2],
    datatype: Box<KernelGlobalType>,
}

impl MatrixType {
    pub fn new(size: [String; 2], datatype: impl Into<KernelGlobalType>) -> Self {
        Self {
            size,
            datatype: Box::new(datatype.into()),
        }
    }
}
