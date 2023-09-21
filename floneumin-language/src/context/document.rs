use url::Url;

#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    title: String,
    body: String,
}

impl Document {
    pub async fn new<T: IntoDocument>(source: T) -> anyhow::Result<Self> {
        source.into_document().await
    }

    pub fn from_parts(title: String, body: String) -> Self {
        Self { title, body }
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn body(&self) -> &str {
        &self.body
    }
}

#[async_trait::async_trait]
pub trait IntoDocument {
    async fn into_document(self) -> anyhow::Result<Document>;
}

#[async_trait::async_trait]
impl IntoDocument for String {
    async fn into_document(self) -> anyhow::Result<Document> {
        Ok(Document::from_parts(Default::default(), self))
    }
}

#[async_trait::async_trait]
impl IntoDocument for &str {
    async fn into_document(self) -> anyhow::Result<Document> {
        Ok(Document::from_parts(Default::default(), self.to_string()))
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
