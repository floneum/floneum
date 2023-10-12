use crate::context::SearchQuery;
use crate::floneumin_sample::StructureParser;
use crate::index::IntoDocuments;
use crate::tool::Tool;

/// A tool that can search the web
pub struct WebSearchTool;

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> String {
        "Web Search".to_string()
    }

    fn description(&self) -> String {
        "Search the web for a query.\nUse tool with:\nAction: Web Search\nAction Input: the search query\nExample:\n\nQuestion: What is Floneum?\nThought: I don't remember what Floneum is. I should search the web for it.\nAction: Web Search\nAction Input: What is Floneum?\nObservation: Floneum is a visual editor for AI workflows.\nThought: I now know that Floneum is a visual editor for AI workflows.\nFinal Answer: Floneum is a visual editor for AI workflows.".to_string()
    }

    fn constraints(&self) -> StructureParser {
        StructureParser::Then {
            first: Box::new(StructureParser::Literal("Search Query".to_string())),
            second: Box::new(StructureParser::String {
                min_len: 1,
                max_len: 100,
            }),
        }
    }

    async fn run(&self, args: Vec<String>) -> String {
        let query = args.join(" ");
        let api_key = std::env::var("GOOGLE_API_KEY").unwrap();
        let search_query = SearchQuery::new(query, api_key);
        let documents = search_query.into_documents().await.unwrap();
        let mut text = String::new();
        for document in documents {
            text.push_str(document.body());
            text.push('\n');
        }
        text
    }
}
