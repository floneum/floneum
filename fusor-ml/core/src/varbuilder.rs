use std::{fmt::Debug, sync::Arc};

use crate::{Device, QMatrix};
use fusor_gguf::{GgufMetadata, GgufReadError};

trait ReadAndSeek: std::io::Read + std::io::Seek {}

impl<T: std::io::Read + std::io::Seek + ?Sized> ReadAndSeek for T {}

pub struct VarBuilder<'a> {
    reader: &'a mut dyn ReadAndSeek,
    metadata: Arc<GgufMetadata>,
    path: Vec<String>,
}

impl<'a> Debug for VarBuilder<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilder")
            .field("path", &self.path)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl<'a> VarBuilder<'a> {
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        reader: &'a mut R,
    ) -> Result<Self, GgufReadError> {
        let metadata = GgufMetadata::read(&mut *reader)?.into();
        let path = Default::default();
        Ok(Self {
            reader,
            metadata,
            path,
        })
    }

    pub fn pp<'b, S: ToString>(&'b mut self, s: S) -> VarBuilder<'b> {
        let mut new_path = self.path.clone();
        new_path.push(s.to_string());
        VarBuilder {
            reader: &mut *self.reader,
            metadata: self.metadata.clone(),
            path: new_path,
        }
    }

    fn format_path(&self, name: &str) -> String {
        let mut full_path = self.path.join(".");
        if !full_path.is_empty() {
            full_path.push('.');
        }
        full_path.push_str(name);
        full_path
    }

    pub fn get(&mut self, key: &str, device: &Device) -> crate::Result<QMatrix> {
        let full_path = self.format_path(key);
        let q_matrix_metadata = self.metadata.tensor_infos.get(&*full_path).ok_or_else(|| {
            crate::Error::VarBuilder(format!("Key '{}' not found in GGUF metadata", full_path))
        })?;

        let q_matrix = QMatrix::read(
            &device,
            q_matrix_metadata,
            &mut self.reader,
            self.metadata.tensor_data_offset,
        )?;

        Ok(q_matrix)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let full_path = self.format_path(key);
        self.metadata.tensor_infos.contains_key(&*full_path)
    }

    pub fn list_all_keys(&self) -> Vec<String> {
        self.metadata
            .tensor_infos
            .keys()
            .map(|k| k.to_string())
            .collect()
    }
}
