use std::fmt::Display;
use std::fmt::Write;

use crate::DataTypeEnum;

#[derive(Clone)]
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

#[derive(Clone)]
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
                let dtype = &array.datatype;
                let size = &array.size;
                let space = &self.space;
                format!("var<{space}> {self}: array<{dtype}, {size}>;\n")
            }
        }
    }
}

impl Display for KernelGlobal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g_{}", self.id)
    }
}

#[derive(Clone)]
pub enum KernelGlobalType {
    Array(ArrayType),
}

#[derive(Clone)]
pub struct ArrayType {
    size: String,
    datatype: DataTypeEnum,
}

impl ArrayType {
    pub fn new(size: String, datatype: DataTypeEnum) -> Self {
        Self { size, datatype }
    }
}
