use tantivy::collector::{Count, TopDocs};
use tantivy::query::FuzzyTermQuery;
use tantivy::schema::*;
use tantivy::{doc, Index};

pub struct FuzzySearchIndex {
    title: Field,
    body: Field,
    index: Index,
}

impl Default for FuzzySearchIndex {
    fn default() -> Self {
        let mut schema_builder = Schema::builder();
        let title = schema_builder.add_text_field("title", TEXT | STORED);
        let body = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        Self { title, body, index }
    }
}

#[async_trait::async_trait]
impl super::SearchIndex for FuzzySearchIndex {
    async fn add(&mut self, document: crate::context::document::Document) {
        let mut index_writer = self.index.writer(50_000_000).unwrap();
        index_writer
            .add_document(doc!(
                self.title => document.title(),
                self.body => document.body()
            ))
            .unwrap();
        index_writer.commit().unwrap();
    }

    async fn search(&self, query: &str, top_n: usize) -> Vec<super::DocumentSnippetRef> {
        let reader = self.index.reader().unwrap();
        let searcher = reader.searcher();
        let term = Term::from_field_text(self.title, query);
        let query = FuzzyTermQuery::new(term, 2, true);
        let (top_docs, _) = searcher
            .search(&query, &(TopDocs::with_limit(top_n), Count))
            .unwrap();
        let mut results = Vec::new();
        for (_, doc_address) in top_docs {
            let retrieved_doc = searcher.doc(doc_address).unwrap();
            results.push(super::DocumentSnippetRef {
                title: retrieved_doc
                    .get_first(self.title)
                    .unwrap()
                    .as_text()
                    .unwrap()
                    .to_string()
                    .into(),
                body: retrieved_doc
                    .get_first(self.body)
                    .unwrap()
                    .as_text()
                    .unwrap()
                    .to_string()
                    .into(),
                byte_range: 0..0,
            });
        }
        results
    }
}
