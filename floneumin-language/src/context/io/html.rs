use std::path::PathBuf;

use tokio::{fs::File, io::AsyncReadExt};

use crate::context::{
    document::{Document, IntoDocument},
    page::extract_article,
};

#[derive(Debug, Clone)]
pub struct HtmlDocument {
    path: PathBuf,
}

impl TryFrom<PathBuf> for HtmlDocument {
    type Error = anyhow::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if !path.is_file() {
            return Err(anyhow::anyhow!("Path is not a file"));
        }
        if path.extension().unwrap() != "html" {
            return Err(anyhow::anyhow!("Path is not a html file"));
        }
        Ok(Self { path })
    }
}

#[async_trait::async_trait]
impl IntoDocument for HtmlDocument {
    async fn into_document(self) -> anyhow::Result<Document> {
        let file = File::open(self.path).await?;
        let mut html = String::new();
        tokio::io::BufReader::new(file)
            .read_to_string(&mut html)
            .await?;
        extract_article(&html)
    }
}
