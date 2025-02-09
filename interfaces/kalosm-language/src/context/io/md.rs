use std::path::PathBuf;

use tokio::{fs::File, io::AsyncReadExt};

use crate::context::{
    document::{Document, IntoDocument},
    page::extract_article,
    ExtractDocumentError,
};

use super::FsDocumentError;

/// A markdown document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct MdDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for MdDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "md" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

impl IntoDocument for MdDocument {
    type Error = FsDocumentError<ExtractDocumentError>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let file = File::open(self.path).await?;
        let mut md = String::new();
        tokio::io::BufReader::new(file)
            .read_to_string(&mut md)
            .await?;
        let parser = pulldown_cmark::Parser::new(&md);

        let mut html_output = String::new();
        pulldown_cmark::html::push_html(&mut html_output, parser);
        extract_article(&html_output).map_err(FsDocumentError::Decode)
    }
}
