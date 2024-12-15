use docx_rs::{Document as DocxFile, FromXML};
use std::path::PathBuf;

use std::fs::File;

use crate::context::document::{Document, IntoDocument};

use super::FsDocumentError;

/// A docx document that can be read from the file system.
#[derive(Debug, Clone)]
pub struct DocxDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for DocxDocument {
    type Error = FsDocumentError;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
        }
        if path.extension().unwrap() != "docx" {
            return Err(FsDocumentError::WrongFileType);
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for DocxDocument {
    type Error = FsDocumentError<docx_rs::ReaderError>;

    async fn into_document(self) -> Result<Document, Self::Error> {
        let file = File::open(self.path)?;
        let reader = std::io::BufReader::new(file);
        let docx = DocxFile::from_xml(reader).map_err(|err| FsDocumentError::Decode(err))?;
        let mut text = String::new();
        for section in docx.children {
            match section {
                docx_rs::DocumentChild::Paragraph(paragraph) => {
                    for child in paragraph.children {
                        match child {
                            docx_rs::ParagraphChild::Run(run) => {
                                for child in run.children {
                                    match child {
                                        docx_rs::RunChild::Text(text_child) => {
                                            text += &text_child.text;
                                        }
                                        docx_rs::RunChild::Sym(_) => {}
                                        docx_rs::RunChild::DeleteText(_) => {}
                                        docx_rs::RunChild::Tab(_) => {}
                                        docx_rs::RunChild::Break(_) => {}
                                        docx_rs::RunChild::Drawing(_) => {}
                                        docx_rs::RunChild::Shape(_) => {}
                                        docx_rs::RunChild::CommentStart(_) => {}
                                        docx_rs::RunChild::CommentEnd(_) => {}
                                        docx_rs::RunChild::FieldChar(_) => {}
                                        docx_rs::RunChild::InstrText(_) => {}
                                        docx_rs::RunChild::DeleteInstrText(_) => {}
                                        docx_rs::RunChild::InstrTextString(_) => {}
                                    }
                                }
                            }
                            docx_rs::ParagraphChild::Insert(_) => {}
                            docx_rs::ParagraphChild::Delete(_) => {}
                            docx_rs::ParagraphChild::BookmarkStart(_) => {}
                            docx_rs::ParagraphChild::Hyperlink(_) => {}
                            docx_rs::ParagraphChild::BookmarkEnd(_) => {}
                            docx_rs::ParagraphChild::CommentStart(_) => {}
                            docx_rs::ParagraphChild::CommentEnd(_) => {}
                            docx_rs::ParagraphChild::StructuredDataTag(_) => {}
                        }
                    }
                }
                docx_rs::DocumentChild::Table(_) => {}
                docx_rs::DocumentChild::BookmarkStart(_) => {}
                docx_rs::DocumentChild::BookmarkEnd(_) => {}
                docx_rs::DocumentChild::CommentStart(_) => {}
                docx_rs::DocumentChild::CommentEnd(_) => {}
                docx_rs::DocumentChild::StructuredDataTag(_) => {}
                docx_rs::DocumentChild::TableOfContents(_) => {}
            }
        }
        Ok(Document::from_parts("", text))
    }
}
