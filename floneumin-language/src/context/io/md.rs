use std::path::PathBuf;

use tokio::{fs::File, io::AsyncReadExt};

use crate::context::{
    document::{Document, IntoDocument},
    page::extract_article,
};

#[derive(Debug, Clone)]
pub struct MdDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for MdDocument {
    type Error = anyhow::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(anyhow::anyhow!("Path is not a file"));
        }
        if path.extension().unwrap() != "md" {
            return Err(anyhow::anyhow!("Path is not a md file"));
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for MdDocument {
    async fn into_document(self) -> anyhow::Result<Document> {
        let file = File::open(self.path).await?;
        let mut md = String::new();
        tokio::io::BufReader::new(file)
            .read_to_string(&mut md)
            .await?;
        let parser = pulldown_cmark::Parser::new(&md);

        let mut html_output = String::new();
        pulldown_cmark::html::push_html(&mut html_output, parser);
        extract_article(&html_output)
    }
}
