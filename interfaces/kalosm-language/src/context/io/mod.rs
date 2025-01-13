use crate::context::document::Document;
use crate::context::document::IntoDocument;
use crate::context::document::IntoDocuments;
use ::pdf::PdfError;
use std::path::PathBuf;
use tokio::task::JoinSet;
mod docx;
pub use docx::*;
mod html;
pub use html::*;
mod md;
pub use md::*;
mod pdf;
pub use self::pdf::*;
mod txt;
pub use txt::*;

use super::ExtractDocumentError;

/// An error that can occur when reading a document from the file system.
#[derive(Debug, thiserror::Error)]
pub enum FsDocumentError<E = std::convert::Infallible> {
    /// An error reading the file
    #[error("Failed to read document: {0}")]
    Read(#[from] std::io::Error),
    /// An error decoding the file
    #[error("Failed to decode document: {0}")]
    Decode(E),
    /// Wrong file type
    #[error("Wrong file type")]
    WrongFileType,
}

impl<E> FsDocumentError<E> {
    fn map_decode<F, E2>(self, f: F) -> FsDocumentError<E2>
    where
        F: FnOnce(E) -> E2,
    {
        match self {
            FsDocumentError::Read(err) => FsDocumentError::Read(err),
            FsDocumentError::Decode(err) => FsDocumentError::Decode(f(err)),
            FsDocumentError::WrongFileType => FsDocumentError::WrongFileType,
        }
    }
}

/// An error that can occur when decoding a text file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TextFileDecodeError {
    /// An error extracting the document from the text file
    #[error("Failed to extract document from text file: {0}")]
    Extract(#[from] ExtractDocumentError),
    /// An error decoding the pdf file
    #[error("Failed to decode pdf file: {0}")]
    Pdf(#[from] PdfError),
    /// An error reading the docx file
    #[error("Failed to read docx file: {0}")]
    Docx(#[from] docx_rs::ReaderError),
}

/// A document that can be read from the file system.
///
/// # Example
/// ```rust, no_run
/// use kalosm_language::prelude::*;
/// use std::path::PathBuf;
///
/// #[tokio::main]
/// async fn main() {
///     let document = FsDocument::try_from(PathBuf::from("./documents"))
///         .unwrap()
///         .into_document()
///         .await
///         .unwrap();
///     println!("document: {:?}", document);
/// }
/// ```
#[derive(Debug, Clone)]
pub enum FsDocument {
    /// A docx document.
    Docx(DocxDocument),
    /// An html document.
    Html(HtmlDocument),
    /// A markdown document.
    Md(MdDocument),
    /// A pdf document.
    Pdf(PdfDocument),
    /// A text document.
    Txt(TextDocument),
}

impl TryFrom<PathBuf> for FsDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("docx") => Ok(Self::Docx(DocxDocument::try_from(path)?)),
            Some("html") => Ok(Self::Html(HtmlDocument::try_from(path)?)),
            Some("md") => Ok(Self::Md(MdDocument::try_from(path)?)),
            Some("pdf") => Ok(Self::Pdf(PdfDocument::try_from(path)?)),
            Some("txt") => Ok(Self::Txt(TextDocument::try_from(path)?)),
            _ => Err(FsDocumentError::WrongFileType),
        }
    }
}

#[async_trait::async_trait]
impl IntoDocument for FsDocument {
    type Error = FsDocumentError<TextFileDecodeError>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        match self {
            Self::Docx(docx) => docx
                .into_document()
                .await
                .map_err(|err| err.map_decode(TextFileDecodeError::Docx)),
            Self::Html(html) => html
                .into_document()
                .await
                .map_err(|err| err.map_decode(TextFileDecodeError::Extract)),
            Self::Md(md) => md
                .into_document()
                .await
                .map_err(|err| err.map_decode(TextFileDecodeError::Extract)),
            Self::Pdf(pdf) => pdf
                .into_document()
                .await
                .map_err(|err| err.map_decode(TextFileDecodeError::Pdf)),
            Self::Txt(txt) => txt
                .into_document()
                .await
                .map_err(|err| err.map_decode(|_| unreachable!())),
        }
    }
}

/// A folder full of documents.
///
/// # Example
/// ```rust, no_run
/// # use kalosm::language::*;
/// # use std::io::Write;
/// # use std::path::PathBuf;
/// #[tokio::main]
/// async fn main() {
///     // You can load a whole folder full of documents with the DocumentFolder source
///     let folder = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();
///     // Grab all the documents out of the folder
///     let documents = folder.into_documents().await.unwrap();
///
///     // Then chunk the documents into sentences and use those chunks however you need
///     let model = Bert::new().await.unwrap();
///     let chunked = SemanticChunker::new()
///         .chunk_batch(&documents, &model)
///         .await
///         .unwrap();
///     println!("{:?}", chunked);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DocumentFolder {
    path: PathBuf,
}

/// The path to a document folder was not a directory
#[derive(Debug, thiserror::Error)]
#[error("The path to a document folder was not a directory")]
pub struct DocumentFolderNotDirectoryError;

impl TryFrom<PathBuf> for DocumentFolder {
    type Error = DocumentFolderNotDirectoryError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_dir() {
            return Err(DocumentFolderNotDirectoryError);
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocuments for DocumentFolder {
    type Error = FsDocumentError<TextFileDecodeError>;

    async fn into_documents(self) -> Result<Vec<Document>, Self::Error> {
        let mut set = JoinSet::new();
        self.start_into_documents(&mut set).await?;
        let mut documents = Vec::new();
        while let Some(join) = set.join_next().await {
            let Ok(join) = join else {
                continue;
            };
            documents.push(join?);
        }
        Ok(documents)
    }
}

impl DocumentFolder {
    /// Try to create a new document folder from a path.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm_language::prelude::*;
    ///
    /// let folder = DocumentFolder::new("./documents").unwrap();
    /// ```
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, DocumentFolderNotDirectoryError> {
        Self::try_from(path.into())
    }

    fn start_into_documents<'a>(
        &'a self,
        set: &'a mut JoinSet<Result<Document, FsDocumentError<TextFileDecodeError>>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), std::io::Error>> + Send + Sync + 'a>,
    > {
        Box::pin(async move {
            let mut read_dir = tokio::fs::read_dir(&self.path).await?;
            while let Some(entry) = read_dir.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(folder) = DocumentFolder::try_from(path) {
                        folder.start_into_documents(set).await?;
                    }
                } else if let Ok(document) = FsDocument::try_from(path) {
                    set.spawn(document.into_document());
                }
            }
            Ok(())
        })
    }
}
