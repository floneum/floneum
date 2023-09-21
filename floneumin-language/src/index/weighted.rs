use crate::index::DocumentSnippetRef;
use crate::index::IntoDocument;
use crate::index::IntoDocuments;
use crate::index::SearchIndex;

pub struct WeightedIndex<First, Second> {
    weights: Vec<f64>,
    first: First,
    second: Second,
}

#[async_trait::async_trait]
impl<First: SearchIndex + Send + Sync, Second: SearchIndex + Send + Sync> SearchIndex for WeightedIndex<First, Second> {
    async fn extend(&mut self, document: impl IntoDocuments + Send + Sync) -> anyhow::Result<()> {
        let documents = document.into_documents().await?;
        self.first.extend(documents.clone()).await?;
        self.second.extend(documents).await?;
        Ok(())
    }
    
    async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()> {
        let document = document.into_document().await?;
        self.first.add(document.clone()).await?;
        self.second.add(document).await?;
        Ok(())
    }

    async fn search(&self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef> {
        let mut first = self.first.search(query, top_n).await;
        let mut second = self.second.search(query, top_n).await;
        let mut result = Vec::new();
        while !first.is_empty() && !second.is_empty() {
            let first_score = first.last().unwrap().score();
            let second_score = second.last().unwrap().score();
            if first_score > second_score {
                result.push(first.pop().unwrap());
            } else {
                result.push(second.pop().unwrap());
            }
        }
        result
    }
}
