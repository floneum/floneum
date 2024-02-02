use crate::host::State;
use crate::plugins::main;

use crate::plugins::main::types::{Embedding, EmbeddingDb};

use kalosm::language::{Document, UnknownVectorSpace, VectorDB};
use wasmtime::component::__internal::async_trait;

#[async_trait]
impl main::types::HostEmbeddingDb for State {
    async fn new(
        &mut self,
        embeddings: Vec<Embedding>,
        documents: Vec<String>,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingDb>> {
        let documents = documents
            .into_iter()
            .map(|x| Document::from_parts(String::new(), x));
        let mut db = VectorDBWithDocuments::new()?;

        for (embedding, document) in embeddings.into_iter().zip(documents.into_iter()) {
            db.add_embedding(embedding, document)?;
        }

        let idx = self.embedding_dbs.insert(db);
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn add_embedding(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        embedding: Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        self.embedding_dbs[self_.rep() as usize]
            .add_embedding(embedding, Document::from_parts(String::new(), document))?;
        Ok(())
    }

    async fn find_closest_documents(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        let documents =
            self.embedding_dbs[self_.rep() as usize].get_closest(search, count as usize);
        Ok(documents?
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<EmbeddingDb>) -> wasmtime::Result<()> {
        self.embedding_dbs.remove(rep.rep() as usize);
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct VectorDBWithDocuments {
    db: VectorDB<UnknownVectorSpace>,
    documents: Vec<Option<Document>>,
}

impl VectorDBWithDocuments {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            db: VectorDB::new()?,
            documents: Vec::new(),
        })
    }

    pub fn add_embedding(
        &mut self,
        embedding: Embedding,
        document: Document,
    ) -> anyhow::Result<()> {
        let id = self.db.add_embedding(embedding.vector.into())?;
        if id.0 as usize >= self.documents.len() {
            self.documents.resize(id.0 as usize + 1, None);
        }
        self.documents[id.0 as usize] = Some(document);
        Ok(())
    }

    pub fn get_closest(
        &self,
        embedding: Embedding,
        count: usize,
    ) -> anyhow::Result<Vec<(f32, &Document)>> {
        let results = self.db.get_closest(embedding.vector.into(), count)?;
        Ok(results
            .into_iter()
            .filter_map(|result| {
                let id = result.value;
                let distance = result.distance;
                let document = self.documents[id.0 as usize].as_ref()?;
                Some((distance, document))
            })
            .collect())
    }
}
