use convert_case::{Case, Casing};
use std::path::PathBuf;

use tokio::{fs::File, io::AsyncReadExt};

use crate::context::document::{Document, IntoDocument};

use super::FsDocumentError;

/// A text document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct TextDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for TextDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "txt" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for TextDocument {
    type Error = FsDocumentError;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let stem = self.path.file_stem();
        let title = stem
            .unwrap()
            .to_string_lossy()
            .to_string()
            .to_case(Case::Title);
        let file = File::open(self.path).await?;
        let mut text = String::new();
        tokio::io::BufReader::new(file)
            .read_to_string(&mut text)
            .await?;
        Ok(Document::from_parts(title, text))
    }
}
