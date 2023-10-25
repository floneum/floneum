#![allow(missing_docs)]

use rand::seq::SliceRandom;
use url::Url;

use super::{document::Document, page::get_article};
use crate::index::IntoDocuments;

/// A search query that can be used to search for documents on the web.
pub struct SearchQuery<'a> {
    query: &'a str,
    api_key: &'a str,
    top: usize,
}

impl<'a> SearchQuery<'a> {
    /// Create a new search query.
    pub fn new(query: &'a str, api_key: &'a str, top_n: usize) -> Self {
        Self {
            query,
            api_key,
            top: top_n,
        }
    }
}

#[async_trait::async_trait]
impl IntoDocuments for SearchQuery<'_> {
    async fn into_documents(self) -> anyhow::Result<Vec<Document>> {
        let mut search_results = search(self.api_key, self.query).await?;

        let mut documents = vec![];
        search_results.organic.shuffle(&mut rand::thread_rng());

        for result in search_results.organic.into_iter().take(self.top) {
            if let Some(link) = &result.link {
                documents.push(get_article(Url::parse(link)?).await?);
            }
        }

        Ok(documents)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SearchResult {
    pub knowledge_graph: Option<KnowledgeGraph>,
    #[serde(default)]
    pub organic: Vec<Organic>,
    #[serde(default)]
    pub people_also_ask: Vec<PeopleAlsoAsk>,
    #[serde(default)]
    pub related_searches: Vec<RelatedSearches>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct KnowledgeGraph {
    pub title: String,
    pub type_: String,
    pub website: String,
    pub image_url: String,
    pub description: String,
    pub description_source: String,
    pub description_link: String,
    #[serde(default)]
    pub attributes: Vec<Attributes>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Attributes {
    pub key: String,
    pub value: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Organic {
    pub title: Option<String>,
    pub link: Option<String>,
    #[serde(default)]
    pub snippet: String,
    #[serde(default)]
    pub sitelinks: Vec<Sitelinks>,
    pub position: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Sitelinks {
    pub title: String,
    pub link: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PeopleAlsoAsk {
    pub question: String,
    pub snippet: String,
    pub title: String,
    pub link: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct RelatedSearches {
    pub query: String,
}

pub async fn search(api_key: &str, query: &str) -> Result<SearchResult, reqwest::Error> {
    let url = Url::parse("https://google.serper.dev/search").unwrap();
    let client = reqwest::Client::new();
    let res = client
        .post(url)
        .header("X-API-KEY", api_key)
        .json(&serde_json::json!({
            "q": query
        }))
        .send()
        .await
        .unwrap();
    res.json().await
}

pub fn prompt_search_query(question: &str) -> String {
    let date = chrono::Local::now().format("%Y-%m-%d");

    format!("My question is: {question}. Based on the conversation history, give me an appropriate query to answer my question for google search. You should not say more than query. You should not say any words except the query. For the context, today is {date}.")
}

#[tokio::test]
async fn search_result() {
    if let Some(key) = option_env!("SERPER_API_KEY") {
        let result = search(key, "apple inc").await.unwrap();
        println!("{:#?}", result);
    }
}
