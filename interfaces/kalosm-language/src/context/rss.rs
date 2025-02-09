use rss::Channel;
use url::Url;

use super::document::{Document, IntoDocuments};

/// An error that can occur when interacting with an RSS feed.
#[derive(Debug, thiserror::Error)]
pub enum RssFeedError {
    /// An error occurred when fetching the RSS feed.
    #[error("Failed to fetch RSS feed: {0}")]
    FetchFeed(#[from] reqwest::Error),
    /// An error parsing the RSS feed.
    #[error("Failed to parse RSS feed: {0}")]
    ParseFeed(#[from] rss::Error),
}

/// A RSS feed that can be used to add documents to a search index.
///
/// # Example
/// ```rust, no_run
/// use kalosm_language::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let feed = RssFeed::new(
///         url::Url::parse("https://www.nytimes.com/services/xml/rss/nyt/HomePage.xml").unwrap(),
///     );
///     let documents = feed.read_top_n(5).await.unwrap();
///     println!("Documents: {:?}", documents);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RssFeed(Url);

impl From<Url> for RssFeed {
    fn from(url: Url) -> Self {
        Self::new(url)
    }
}

impl IntoDocuments for RssFeed {
    type Error = RssFeedError;

    async fn into_documents(self) -> Result<Vec<Document>, Self::Error> {
        self.read_top_n(usize::MAX).await
    }
}

impl RssFeed {
    /// Create a new RSS feed from the given URL.
    pub fn new(url: Url) -> Self {
        Self(url)
    }

    /// Get the URL of the RSS feed.
    pub fn url(&self) -> &Url {
        &self.0
    }

    /// Read the top N documents from the RSS feed.
    pub async fn read_top_n(&self, top_n: usize) -> Result<Vec<Document>, RssFeedError> {
        let xml = reqwest::get(self.0.clone()).await?.text().await?;
        let channel = Channel::read_from(xml.as_bytes())?;
        let mut documents = Vec::new();
        for item in channel.items().iter().take(top_n) {
            let mut message = String::new();
            if let Some(title) = item.title() {
                message.push_str(&format!("### {}\n", title));
            }
            let (source_url, content) = if let Some(content) = item.content() {
                (None, content.to_string())
            } else if let Some(source_url) = item.link() {
                (
                    Some(source_url),
                    reqwest::get(source_url).await?.text().await?,
                )
            } else {
                (None, String::new())
            };

            let url = match source_url {
                Some(url) => Url::parse(url).unwrap(),
                None => self.0.clone(),
            };

            if let Ok(article) =
                readability::extractor::extract(&mut std::io::Cursor::new(&content), &url)
            {
                documents.push(Document::from_parts(article.title, article.text));
            }
        }
        Ok(documents)
    }
}
