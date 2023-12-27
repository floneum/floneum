// use super::{DocumentId, DocumentSnippet, DocumentSnippetRef, SearchIndex, VectorDB};

// /// A document database that stores documents in a [`VectorDB`] in [`Chunk`]s.
// ///
// /// The documents can be searched using the [`SearchIndex`] trait. This database will search based on each chunks embedding to find documents with a similar meaning.
// pub struct DocumentDatabase<S: VectorSpace + Send + Sync + 'static, M: Embedder<S>> {
//     embedder: M,
//     documents: Slab<Document>,
//     database: VectorDB<DocumentSnippet, S>,
// }

// impl<S: VectorSpace + Send + Sync + 'static, M: Embedder<S>> std::fmt::Debug
//     for DocumentDatabase<S, M>
// {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("DocumentDatabase")
//             .field("documents", &self.documents)
//             .field("database", &self.database)
//             .field("strategy", &self.strategy)
//             .finish()
//     }
// }

// #[async_trait::async_trait]
// impl<M: Embedder<S> + Send + Sync + 'static, S: VectorSpace + Sync + Send + 'static> SearchIndex
//     for DocumentDatabase<S, M>
// {
//     async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()> {
//         let document = document.into_document().await?;
//         let embedded = EmbeddedDocument::<S>::new(&mut self.embedder, document, self.strategy)
//             .await
//             .unwrap();
//         let id = self.documents.insert(embedded.raw);
//         for chunk in embedded.chunks {
//             let snippet = DocumentSnippet {
//                 document_id: DocumentId(id),
//                 byte_range: chunk.byte_range,
//             };
//             self.database.add_embedding(chunk.embedding, snippet);
//         }

//         Ok(())
//     }

//     async fn extend(&mut self, documents: impl IntoDocuments + Send + Sync) -> anyhow::Result<()> {
//         let documents = documents.into_documents().await?;
//         let embedded =
//             EmbeddedDocument::<S>::batch_new(&mut self.embedder, documents, self.strategy)
//                 .await
//                 .unwrap();
//         let mut embeddings = Vec::new();
//         let mut values = Vec::new();
//         for embedded in embedded {
//             let id = self.documents.insert(embedded.raw);
//             for chunk in embedded.chunks {
//                 let snippet = DocumentSnippet {
//                     document_id: DocumentId(id),
//                     byte_range: chunk.byte_range,
//                 };
//                 embeddings.push(chunk.embedding);
//                 values.push(snippet);
//             }
//         }
//         self.database.add_embeddings(embeddings, values);

//         Ok(())
//     }

//     async fn search(&mut self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef> {
//         let embedding = self.embedder.embed(query).await.unwrap();
//         self.search_iter(embedding, top_n).collect()
//     }
// }

// impl<M: Embedder<S>, S: VectorSpace + Sync + Send + 'static> DocumentDatabase<S, M> {
//     /// Create a new document database.
//     pub fn new(embedder: M, chunk_strategy: ChunkStrategy) -> Self {
//         Self {
//             documents: Slab::new(),
//             database: VectorDB::new(Vec::new(), Vec::new()),
//             strategy: chunk_strategy,
//             embedder,
//         }
//     }

//     /// Add a document to the database.
//     pub fn add(&mut self, document: Document) -> DocumentId {
//         DocumentId(self.documents.insert(document))
//     }

//     /// Link a document to an embedding.
//     pub fn link_document(&mut self, embedding: Embedding<S>, snippet: DocumentSnippet) {
//         self.database.add_embedding(embedding, snippet);
//     }

//     /// Find the closest documents to a given embedding.
//     pub fn search_iter(
//         &self,
//         embedding: Embedding<S>,
//         n: usize,
//     ) -> impl Iterator<Item = DocumentSnippetRef<'_>> {
//         self.database
//             .get_closest(embedding, n)
//             .into_iter()
//             .map(|(score, snippet)| {
//                 let document = &self.documents[snippet.document_id.0];
//                 DocumentSnippetRef {
//                     score,
//                     title: document.title().into(),
//                     body: document.body().into(),
//                     byte_range: snippet.byte_range.clone(),
//                 }
//             })
//     }

//     /// Find the documents within a given distance of an embedding.
//     pub fn get_within_iter(
//         &self,
//         embedding: Embedding<S>,
//         distance: f32,
//     ) -> impl Iterator<Item = DocumentSnippetRef<'_>> {
//         self.database
//             .get_within(embedding, distance)
//             .into_iter()
//             .map(|(score, snippet)| {
//                 let document = &self.documents[snippet.document_id.0];
//                 DocumentSnippetRef {
//                     score,
//                     title: document.title().into(),
//                     body: document.body().into(),
//                     byte_range: snippet.byte_range.clone(),
//                 }
//             })
//     }
// }
