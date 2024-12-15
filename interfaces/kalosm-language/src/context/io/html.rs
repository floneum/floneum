use std::path::PathBuf;

use tokio::{fs::File, io::AsyncReadExt};

use crate::context::{
    document::{Document, IntoDocument},
    page::extract_article,
    ExtractDocumentError,
};

use super::FsDocumentError;

/// An html document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct HtmlDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for HtmlDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "html" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for HtmlDocument {
    type Error = FsDocumentError<ExtractDocumentError>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let file = File::open(self.path).await?;
        let mut html = String::new();
        tokio::io::BufReader::new(file)
            .read_to_string(&mut html)
            .await?;
        extract_article(&html).map_err(|err| FsDocumentError::Decode(err))
    }
}
