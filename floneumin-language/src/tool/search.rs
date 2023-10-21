use crate::context::SearchQuery;
use crate::index::IntoDocuments;
use crate::tool::Tool;

/// A tool that can search the web
pub struct WebSearchTool {
    top_n: usize,
}

impl WebSearchTool {
    /// Create a new web search tool
    pub fn new(top_n: usize) -> Self {
        Self { top_n }
    }
}

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> String {
        "Web Search".to_string()
    }

    fn input_prompt(&self) -> String {
        "Search query: ".to_string()
    }

    fn description(&self) -> String {
        "Search the web for a query.\nUse tool with:\nAction: Web Search\nSearch query: the search query\nExample:\n\nQuestion: What is Floneum?\nThought: I don't remember what Floneum is. I should search the web for it.\nAction: Web Search\nAction Input: What is Floneum?\nObservation: Floneum is a visual editor for AI workflows.\nThought: I now know that Floneum is a visual editor for AI workflows.\nFinal Answer: Floneum is a visual editor for AI workflows.".to_string()
    }

    async fn run(&mut self, query: &str) -> String {
        let api_key = std::env::var("SERPER_API_KEY").unwrap();
        let search_query = SearchQuery::new(query, &api_key, self.top_n);
        let documents = search_query.into_documents().await.unwrap();
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
