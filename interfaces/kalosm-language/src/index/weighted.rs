use crate::index::DocumentSnippetRef;
use crate::index::IntoDocuments;
use crate::index::SearchIndex;
use crate::IntoDocument;

/// A weighted index that combines two indexes with a weight. This allows you to combine two indexes with different strategies like the [`crate::index::keyword::FuzzySearchIndex`] and a [`crate::index::vector::DocumentDatabase`].
pub struct WeightedIndex<First, Second> {
    first: First,
    first_weight: f32,
    second: Second,
    second_weight: f32,
}

#[async_trait::async_trait]
impl<First: SearchIndex + Send + Sync, Second: SearchIndex + Send + Sync> SearchIndex
    for WeightedIndex<First, Second>
{
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

    async fn search(&mut self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef> {
        let mut first = self.first.search(query, top_n).await;
        let mut second = self.second.search(query, top_n).await;
        let mut result = Vec::new();
        while !first.is_empty() && !second.is_empty() {
            let first_score = first.last().unwrap().score() * self.first_weight;
            let second_score = second.last().unwrap().score() * self.second_weight;
            if first_score > second_score {
                result.push(first.pop().unwrap());
            } else {
                result.push(second.pop().unwrap());
            }
        }
        result
    }
}
