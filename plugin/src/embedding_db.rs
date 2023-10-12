use crate::host::State;
use crate::plugins::main;

use crate::plugins::main::types::{Embedding, EmbeddingDb};

use floneumin::floneumin_language::Document;

use floneumin::floneumin_language::VectorDB;

use wasmtime::component::__internal::async_trait;

#[async_trait]
impl main::types::HostEmbeddingDb for State {
    async fn new(
        &mut self,
        embeddings: Vec<Embedding>,
        documents: Vec<String>,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingDb>> {
        let embeddings = embeddings.into_iter().map(|x| x.vector.into()).collect();
        let documents = documents
            .into_iter()
            .map(|x| Document::from_parts(String::new(), x))
            .collect();
        let db = VectorDB::new(embeddings, documents);
        let idx = self.embedding_dbs.insert(db);
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn add_embedding(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        embedding: Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        self.embedding_dbs[self_.rep() as usize].add_embedding(
            embedding.vector.into(),
            Document::from_parts(String::new(), document),
        );
        Ok(())
    }

    async fn find_closest_documents(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        let documents = self.embedding_dbs[self_.rep() as usize]
            .get_closest(search.vector.into(), count as usize);
        Ok(documents
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    async fn find_documents_within(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        distance: f32,
    ) -> wasmtime::Result<Vec<String>> {
        Ok(self.embedding_dbs[self_.rep() as usize]
            .get_within(search.vector.into(), distance)
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<EmbeddingDb>) -> wasmtime::Result<()> {
        self.embedding_dbs.remove(rep.rep() as usize);
        Ok(())
    }
}
