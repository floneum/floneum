use url::Url;

use super::document::Document;

pub mod browse;

pub async fn get_article(url: &str) -> Result<Document, anyhow::Error> {
    let base_url = Url::parse(&url).unwrap();
    let html = reqwest::get(url).await?.text().await?;
    let cleaned = readability::extractor::extract(&mut html.as_bytes(), &base_url).unwrap();
    Ok(Document::new(cleaned.title, cleaned.content))
}
