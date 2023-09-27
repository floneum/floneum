use crate::context::document::Document;
use crate::context::document::IntoDocument;
use itertools::Itertools;
use std::fmt::Write;
use std::path::PathBuf;

use pdf::file::FileOptions;

#[derive(Debug, Clone)]
pub struct PdfDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for PdfDocument {
    type Error = anyhow::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(anyhow::anyhow!("Path is not a file"));
        }
        if path.extension().unwrap() != "pdf" {
            return Err(anyhow::anyhow!("Path is not a pdf file"));
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for PdfDocument {
    async fn into_document(self) -> anyhow::Result<Document> {
        let file = FileOptions::cached().open(&self.path).unwrap();
        let resolver = file.resolver();
        let mut title = String::new();
        let mut text = String::new();

        if let Some(info) = &file.trailer.info_dict {
            if let Some(pdf_title) = info.title.as_ref().map(|p| p.to_string_lossy()) {
                title = pdf_title;
            }
        }

        for (_, page) in file.pages().enumerate() {
            if let Ok(page) = page {
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
        }

        Ok(Document::from_parts(title, text))
    }
}
