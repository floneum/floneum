use super::document::Document;
pub use crate::context::page::crawl::CrawlFeedback;
pub use crate::context::page::crawl::CrawlingCallback;
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

pub(crate) async fn get_article(url: Url) -> Result<Document, anyhow::Error> {
    let html = reqwest::get(url.clone()).await?.text().await?;
    extract_article(&html)
}

pub(crate) fn extract_article(html: &str) -> anyhow::Result<Document> {
    let cleaned =
        readability::extractor::extract(&mut html.as_bytes(), &Url::parse("https://example.com")?)
            .unwrap();
    Ok(Document::from_parts(cleaned.title, cleaned.text))
}
