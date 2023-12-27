// use std::borrow::Cow;
// use tantivy::collector::TopDocs;
// use tantivy::query::QueryParser;
// use tantivy::schema::*;
// use tantivy::{doc, Index};

// use crate::prelude::IntoDocument;


// /// A fuzzy keyword search index that can be used to search for documents.
// ///
// /// # Example
// /// ```rust
// /// use kalosm_language::*;
// ///
// /// #[tokio::main]
// /// async fn main() {
// ///     let mut fuzzy = SearchIndex::default();
// ///     fuzzy.add(Document::from_parts("title", "This is a test of fuzzy search")).await.unwrap();
// ///
// ///     let results = fuzzy.search("fuzzy", 1).await;
// ///     assert_eq!(results.len(), 1);
// ///     assert_eq!(results[0].body(), "This is a test of fuzzy search");
// /// }
// /// ```
// pub struct FuzzySearchIndex {
//     title: Field,
//     body: Field,
//     index: Index,
// }

// impl Default for FuzzySearchIndex {
//     fn default() -> Self {
//         let mut schema_builder = Schema::builder();
//         let title = schema_builder.add_text_field("title", TEXT | STORED);
//         let body = schema_builder.add_text_field("body", TEXT | STORED);
//         let schema = schema_builder.build();
//         let index = Index::create_in_ram(schema.clone());
//         Self { title, body, index }
//     }
// }

// #[async_trait::async_trait]
// impl super::SearchIndex for FuzzySearchIndex {
//     async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()> {
//         let document = document.into_document().await?;
//         let mut index_writer = self.index.writer(50_000_000).unwrap();
//         index_writer
//             .add_document(doc!(
//                 self.title => document.title(),
//                 self.body => document.body()
//             ))
//             .unwrap();
//         index_writer.commit().unwrap();
//         Ok(())
//     }

//     async fn search(&mut self, query_str: &str, top_n: usize) -> Vec<super::DocumentSnippetRef> {
//         let reader = self.index.reader().unwrap();

//         let searcher = reader.searcher();

//         let query_parser = QueryParser::for_index(&self.index, vec![self.title, self.body]);
//         let query = match query_parser.parse_query(query_str.trim()) {
//             Ok(query) => query,
//             Err(err) => {
//                 tracing::error!("Error parsing query: {:?}", err);
//                 return Vec::new();
//             }
//         };

//         let top_docs = match searcher.search(&query, &TopDocs::with_limit(top_n)) {
//             Ok(top_docs) => top_docs,
//             Err(err) => {
//                 tracing::error!("Error searching query: {:?}", err);
//                 return Vec::new();
//             }
//         };

//         let mut results = Vec::new();
//         for (score, doc_address) in top_docs {
//             let retrieved_doc = searcher.doc(doc_address).unwrap();
//             let body: Cow<str> = retrieved_doc
//                 .get_first(self.body)
//                 .map(|v| v.as_text().unwrap().to_string().into())
//                 .unwrap_or_default();

//             results.push(super::DocumentSnippetRef {
//                 score,
//                 title: retrieved_doc
//                     .get_first(self.title)
//                     .map(|v| v.as_text().unwrap().to_string().into())
//                     .unwrap_or_default(),
//                 byte_range: 0..body.len(),
//                 body,
//             });
//         }
//         results
//     }
// }

// #[test]
// fn testing() -> tantivy::Result<()> {
//     let mut schema_builder = Schema::builder();
//     let title = schema_builder.add_text_field("title", TEXT | STORED);
//     let body = schema_builder.add_text_field("body", TEXT);
//     let schema = schema_builder.build();
//     let index = Index::create_in_ram(schema.clone());
//     {
//         {
//             let mut index_writer = index.writer(15_000_000)?;
//             index_writer.add_document(doc!(
//                 title => "The Name of the Wind",
//             ))?;
//             index_writer.commit().unwrap();
//         }
//         {
//             let mut index_writer = index.writer(15_000_000)?;
//             index_writer.add_document(doc!(
//                 title => "The Diary of Muadib",
//             ))?;
//             index_writer.commit().unwrap();
//         }
//         {
//             let mut index_writer = index.writer(15_000_000)?;
//             index_writer.add_document(doc!(
//                 title => "A Dairy Cow",
//             ))?;
//             index_writer.commit().unwrap();
//         }
//         {
//             let mut index_writer = index.writer(15_000_000)?;
//             index_writer.add_document(doc!(
//                 title => "The Diary of a Young Girl",
//             ))?;
//             index_writer.commit().unwrap();
//         }
//     }
//     let reader = index.reader()?;
//     let searcher = reader.searcher();

//     {
//         let query_parser = QueryParser::for_index(&index, vec![title, body]);
//         let query = match query_parser.parse_query("dairy") {
//             Ok(query) => query,
//             Err(err) => {
//                 tracing::error!("Error parsing query: {:?}", err);
//                 return Ok(());
//             }
//         };

//         let top_docs = match searcher.search(&query, &TopDocs::with_limit(2)) {
//             Ok(top_docs) => top_docs,
//             Err(err) => {
//                 tracing::error!("Error searching query: {:?}", err);
//                 return Ok(());
//             }
//         };
//         println!("{:?}", top_docs);
//     }

//     Ok(())
// }
