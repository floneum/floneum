use super::document::Document;
pub use crate::context::page::crawl::CrawlFeedback;
pub use crate::context::page::crawl::CrawlingCallback;
pub use node::*;
pub use page::*;
use url::Url;

pub mod browse;
mod crawl;
mod node;
mod page;

pub async fn get_article(url: Url) -> Result<Document, anyhow::Error> {
    let html = reqwest::get(url.clone()).await?.text().await?;
    extract_article(&html)
}

pub fn extract_article(html: &str) -> anyhow::Result<Document> {
    let cleaned =
        readability::extractor::extract(&mut html.as_bytes(), &Url::parse("https://example.com")?)
            .unwrap();
    Ok(Document::from_parts(cleaned.title, cleaned.text))
}
