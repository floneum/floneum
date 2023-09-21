use url::Url;

use super::document::Document;

pub mod browse;

pub async fn get_article(url: Url) -> Result<Document, anyhow::Error> {
    let html = reqwest::get(url.clone()).await?.text().await?;
    let cleaned = readability::extractor::extract(&mut html.as_bytes(), &url).unwrap();
    Ok(Document::from_parts(cleaned.title, cleaned.text))
}
