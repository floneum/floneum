use crate::context::document::Document;
use crate::context::document::IntoDocument;
use crate::index::IntoDocuments;
use std::path::PathBuf;
use tokio::task::JoinSet;
mod docx;
pub use docx::*;
mod html;
pub use html::*;
mod md;
pub use md::*;
mod pdf;
pub use pdf::*;
mod txt;
pub use txt::*;

#[derive(Debug, Clone)]
pub enum FsDocument {
    Docx(DocxDocument),
    Html(HtmlDocument),
    Md(MdDocument),
    Pdf(PdfDocument),
    Txt(TextDocument),
}

impl TryFrom<PathBuf> for FsDocument {
    type Error = anyhow::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(anyhow::anyhow!("Path is not a file"));
        }
        match path.extension().unwrap().to_str().unwrap() {
            "docx" => Ok(Self::Docx(DocxDocument::try_from(path)?)),
            "html" => Ok(Self::Html(HtmlDocument::try_from(path)?)),
            "md" => Ok(Self::Md(MdDocument::try_from(path)?)),
            "pdf" => Ok(Self::Pdf(PdfDocument::try_from(path)?)),
            "txt" => Ok(Self::Txt(TextDocument::try_from(path)?)),
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
    #[async_recursion::async_recursion]
    async fn start_into_documents(
        &self,
        set: &mut JoinSet<anyhow::Result<Document>>,
    ) -> anyhow::Result<()> {
        let mut read_dir = tokio::fs::read_dir(&self.path).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(folder) = DocumentFolder::try_from(path) {
                    folder.start_into_documents(set).await?;
                }
            } else {
                if let Ok(document) = FsDocument::try_from(path) {
                    set.spawn(document.into_document());
                }
            }
        }
        Ok(())
    }
}
