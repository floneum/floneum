use floneumin_language_model::{Embedder, VectorSpace};

use crate::tool::Tool;
use crate::{DocumentDatabase, SearchIndex};

/// A tool that can search the web
pub struct DocumentSearchTool<S: VectorSpace + Send + Sync + 'static, M: Embedder<S>> {
    database: DocumentDatabase<S, M>,
    top_n: usize,
}

impl<S: VectorSpace + Send + Sync + 'static, M: Embedder<S>> DocumentSearchTool<S, M> {
    /// Create a new web search tool
    pub fn new(database: DocumentDatabase<S, M>, top_n: usize) -> Self {
        Self { database, top_n }
    }
}

#[async_trait::async_trait]
impl<S: VectorSpace + Send + Sync + 'static, M: Embedder<S>> Tool for DocumentSearchTool<S, M> {
    fn name(&self) -> String {
        "Local Search".to_string()
    }

    fn input_prompt(&self) -> String {
        "Search query: ".to_string()
    }

    fn description(&self) -> String {
        "Search local documents for a query.\nUse tool with:\nAction: Local Search\nSearch query: the search query\nExample:\n\nQuestion: What is Floneum?\nThought: I don't remember what Floneum is. I should search for it.\nAction: Local Search\nAction Input: What is Floneum?\nObservation: Floneum is a visual editor for AI workflows.\nThought: I now know that Floneum is a visual editor for AI workflows.\nFinal Answer: Floneum is a visual editor for AI workflows.".to_string()
    }

    async fn run(&mut self, query: &str) -> String {
        let documents = self.database.search(query, self.top_n).await;
        let mut text = String::new();
        for document in documents {
            for word in document.body().split(' ').take(300) {
                text.push_str(word);
                text.push(' ');
            }
            text.push('\n');
        }
        text
    }
}
