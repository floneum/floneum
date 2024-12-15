use crate::context::document::Document;
use crate::context::document::IntoDocument;
use itertools::Itertools;
use pdf::PdfError;
use std::fmt::Write;
use std::path::PathBuf;

use pdf::file::FileOptions;

use super::FsDocumentError;

/// A pdf document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct PdfDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for PdfDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "pdf" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for PdfDocument {
    type Error = FsDocumentError<PdfError>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let file = FileOptions::cached()
            .open(self.path)
            .map_err(FsDocumentError::Decode)?;
        let resolver = file.resolver();
        let mut title = String::new();
        let mut text = String::new();

        if let Some(info) = &file.trailer.info_dict {
            if let Some(pdf_title) = info.title.as_ref().map(|p| p.to_string_lossy()) {
                title = pdf_title;
            }
        }

        for page in file.pages().flatten() {
            if let Ok(flow) = pdf_text::run(&file, &page, &resolver) {
                for run in flow.runs {
                    for line in run.lines {
                        let _ = text.write_fmt(format_args!(
                            "{}",
                            line.words.iter().map(|w| &w.text).format(" ")
                        ));
                        text += "\n\n";
                    }
                }
            }
        }

        Ok(Document::from_parts(title, text))
    }
}
