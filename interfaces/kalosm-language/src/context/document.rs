use url::Url;
pub use whatlang::Lang;

/// A document is a piece of text with a title.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    title: String,
    body: String,
    summary: Option<String>,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
    updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl Document {
    /// Create a new document from a source.
    pub async fn new<T: IntoDocument>(source: T) -> anyhow::Result<Self> {
        source.into_document().await
    }

    /// Get the language of the document.
    pub fn language(&self) -> Option<whatlang::Lang> {
        whatlang::detect_lang(&self.body)
    }

    /// Create a new document from the raw parts.
    pub fn from_parts(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            summary: None,
            created_at: None,
            updated_at: None,
        }
    }

    /// Set the summary of the document.
    pub fn set_summary(&mut self, summary: impl Into<String>) {
        self.summary = Some(summary.into());
    }

    /// Set the created at time of the document.
    pub fn set_created_at(&mut self, created_at: chrono::DateTime<chrono::Utc>) {
        self.created_at = Some(created_at);
    }

    /// Set the updated at time of the document.
    pub fn set_updated_at(&mut self, updated_at: chrono::DateTime<chrono::Utc>) {
        self.updated_at = Some(updated_at);
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

/// A document that can be added to a search index.
#[async_trait::async_trait]
pub trait IntoDocuments {
    /// Convert the document into a [`Document`]
    async fn into_documents(self) -> anyhow::Result<Vec<Document>>;
}

#[async_trait::async_trait]
impl<T: IntoDocument + Send + Sync, I> IntoDocuments for I
where
    I: IntoIterator<Item = T> + Send + Sync,
    <I as IntoIterator>::IntoIter: Send + Sync,
{
    async fn into_documents(self) -> anyhow::Result<Vec<Document>> {
        let mut documents = Vec::new();
        for document in self {
            documents.push(document.into_document().await?);
        }
        Ok(documents)
    }
}
