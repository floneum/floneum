use super::{document::Document, IntoDocument};
use url::Url;

mod browse;
pub use browse::*;
mod crawl;
pub use crawl::*;
mod node;
pub use node::*;
#[allow(clippy::module_inception)]
mod page;
pub use page::*;

/// An error that can occur when extracting a document from a URL.
#[derive(Debug, thiserror::Error)]
pub enum ExtractDocumentError {
    /// An error occurred when fetching the HTML.
    #[error("Failed to fetch HTML: {0}")]
    FetchHtml(#[from] reqwest::Error),
    /// An error occurred when extracting the article.
    #[error("Failed to extract article: {0}")]
    ExtractArticle(#[from] readability::error::Error),
    /// Failed to parse the URL.
    #[error("Failed to parse URL: {0}")]
    ParseUrl(#[from] url::ParseError),
}

pub(crate) async fn get_article(url: Url) -> Result<Document, ExtractDocumentError> {
    let html = reqwest::get(url.clone()).await?.text().await?;
    extract_article(&html)
}

pub(crate) fn extract_article(html: &str) -> Result<Document, ExtractDocumentError> {
    let cleaned =
        readability::extractor::extract(&mut html.as_bytes(), &Url::parse("https://example.com")?)
            .unwrap();
    Ok(Document::from_parts(cleaned.title, cleaned.text))
}

#[async_trait::async_trait]
impl IntoDocument for Url {
    type Error = ExtractDocumentError;

    async fn into_document(self) -> Result<Document, Self::Error> {
        super::page::get_article(self).await
    }
}
