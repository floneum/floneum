use crate::context::document::Document;
use crate::context::document::IntoDocument;
use crate::context::document::IntoDocuments;
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

/// A document that can be read from the file system.
///
/// # Example
/// ```rust, no_run
/// use kalosm_language::prelude::*;
/// use std::path::PathBuf;
///
/// #[tokio::main]
/// async fn main() {
///     let document = FsDocument::try_from(PathBuf::from("./documents")).unwrap().into_document().await.unwrap();
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
    type Error = anyhow::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(anyhow::anyhow!("Path is not a file"));
        }
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("docx") => Ok(Self::Docx(DocxDocument::try_from(path)?)),
            Some("html") => Ok(Self::Html(HtmlDocument::try_from(path)?)),
            Some("md") => Ok(Self::Md(MdDocument::try_from(path)?)),
            Some("pdf") => Ok(Self::Pdf(PdfDocument::try_from(path)?)),
            Some("txt") => Ok(Self::Txt(TextDocument::try_from(path)?)),
            _ => Err(anyhow::anyhow!("Path is not a supported file type")),
        }
    }
}

#[async_trait::async_trait]
impl IntoDocument for FsDocument {
    async fn into_document(self) -> anyhow::Result<Document> {
        match self {
            Self::Docx(docx) => docx.into_document().await,
            Self::Html(html) => html.into_document().await,
            Self::Md(md) => md.into_document().await,
            Self::Pdf(pdf) => pdf.into_document().await,
            Self::Txt(txt) => txt.into_document().await,
        }
    }
}

/// A folder full of documents.
///
/// # Example
/// ```rust, no_run
/// use kalosm_language::prelude::*;
/// use std::io::Write;
/// use std::path::PathBuf;
///
/// #[tokio::main]
/// async fn main() {
///     let documents = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();
///
///     let mut database = DocumentDatabase::new(
///         Bert::new_for_search().unwrap(),
///         ChunkStrategy::Sentence {
///             sentence_count: 1,
///             overlap: 0,
///         },
///     );
///     database.extend(documents).await.unwrap();
///
///     loop {
///         print!("Query: ");
///         std::io::stdout().flush().unwrap();
///         let mut user_question = String::new();
///         std::io::stdin().read_line(&mut user_question).unwrap();
///
///         println!(
///             "{:?}",
///             database
///                 .search(&user_question, 5)
///                 .await
///                 .iter()
///                 .collect::<Vec<_>>()
///         );
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DocumentFolder {
    path: PathBuf,
}

impl TryFrom<PathBuf> for DocumentFolder {
    type Error = anyhow::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_dir() {
            return Err(anyhow::anyhow!("Path is not a directory"));
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocuments for DocumentFolder {
    async fn into_documents(self) -> anyhow::Result<Vec<Document>> {
        let mut set = JoinSet::new();
        self.start_into_documents(&mut set).await?;
        let mut documents = Vec::new();
        while let Some(join) = set.join_next().await {
            documents.push(join??);
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
    pub fn new(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        Self::try_from(path.into())
    }

    fn start_into_documents<'a>(
        &'a self,
        set: &'a mut JoinSet<anyhow::Result<Document>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + Sync + 'a>>
    {
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
