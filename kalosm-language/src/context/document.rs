use url::Url;

/// A document is a piece of text with a title.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    title: String,
    body: String,
}

impl Document {
    /// Create a new document from a source.
    pub async fn new<T: IntoDocument>(source: T) -> anyhow::Result<Self> {
        source.into_document().await
    }

    /// Create a new document from the raw parts.
    pub fn from_parts(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
        }
    }

    /// Get the title of the document.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get the body of the document.
    pub fn body(&self) -> &str {
        &self.body
    }
}

/// A trait for types that can be converted into a document.
#[async_trait::async_trait]
pub trait IntoDocument {
    /// Convert the type into a document.
    async fn into_document(self) -> anyhow::Result<Document>;
}

#[async_trait::async_trait]
impl IntoDocument for String {
    async fn into_document(self) -> anyhow::Result<Document> {
        Ok(Document::from_parts("", self))
    }
}

#[async_trait::async_trait]
impl IntoDocument for &str {
    async fn into_document(self) -> anyhow::Result<Document> {
        Ok(Document::from_parts("", self.to_string()))
    }
}

#[async_trait::async_trait]
impl IntoDocument for Document {
    async fn into_document(self) -> anyhow::Result<Document> {
        Ok(self)
    }
}

#[async_trait::async_trait]
impl IntoDocument for Url {
    async fn into_document(self) -> anyhow::Result<Document> {
        super::page::get_article(self).await
    }
}
