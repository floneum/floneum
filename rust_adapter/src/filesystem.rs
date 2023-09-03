use std::path::PathBuf;

use crate::{IntoPrimitiveValue, PrimitiveValue};

/// A wrapper around a file path.
pub struct File(PathBuf);

impl std::ops::Deref for File {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PathBuf> for File {
    fn from(path: PathBuf) -> Self {
        Self(path)
    }
}

impl From<String> for File {
    fn from(path: String) -> Self {
        Self(PathBuf::from(path))
    }
}

impl IntoPrimitiveValue for File {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::File(self.0.display().to_string())
    }
}

/// A wrapper around a folder path.
pub struct Folder(PathBuf);

impl std::ops::Deref for Folder {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PathBuf> for Folder {
    fn from(path: PathBuf) -> Self {
        Self(path)
    }
}

impl From<String> for Folder {
    fn from(path: String) -> Self {
        Self(PathBuf::from(path))
    }
}

impl IntoPrimitiveValue for Folder {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Folder(self.0.display().to_string())
    }
}
