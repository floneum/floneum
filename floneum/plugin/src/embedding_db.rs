use std::ops::Deref;

use crate::host::State;

use crate::plugins::main::types::{Embedding, EmbeddingDb};

use kalosm::language::{Document, UnknownVectorSpace, VectorDB};
use once_cell::sync::Lazy;

impl State {
    pub(crate) fn impl_create_embedding_db(
        &mut self,
        embeddings: Vec<Embedding>,
        documents: Vec<String>,
    ) -> anyhow::Result<EmbeddingDb> {
        let documents = documents
            .into_iter()
            .map(|x| Document::from_parts(String::new(), x));
        let mut db = VectorDBWithDocuments::new();

        for (embedding, document) in embeddings.into_iter().zip(documents.into_iter()) {
            db.add_embedding(embedding, document)?;
        }

        let idx = self.embedding_dbs.insert(db);
        Ok(EmbeddingDb {
            id: idx as u64,
            owned: true,
        })
    }

    pub(crate) async fn impl_add_embedding(
        &mut self,
        self_: EmbeddingDb,
        embedding: Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        self.embedding_dbs[self_.id as usize]
            .add_embedding(embedding, Document::from_parts(String::new(), document))?;
        Ok(())
    }

    pub(crate) async fn impl_find_closest_documents(
        &mut self,
        self_: EmbeddingDb,
        search: Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        let documents = self.embedding_dbs[self_.id as usize].get_closest(search, count as usize);
        Ok(documents?
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    pub(crate) fn impl_drop_embedding_db(&mut self, rep: EmbeddingDb) -> wasmtime::Result<()> {
        self.embedding_dbs.remove(rep.id as usize);
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct VectorDBWithDocuments {
    db: Lazy<anyhow::Result<VectorDB<UnknownVectorSpace>>>,
    documents: Vec<Option<Document>>,
}

impl VectorDBWithDocuments {
    pub fn new() -> Self {
        Self {
            db: Lazy::new(VectorDB::new),
            documents: Vec::new(),
        }
    }

    pub fn add_embedding(
        &mut self,
        embedding: Embedding,
        document: Document,
    ) -> anyhow::Result<()> {
        let id = self
            .db
            .deref()
            .as_ref()?
            .add_embedding(embedding.vector.into())?;
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
        let results = self
            .db
            .deref()
            .as_ref()?
            .get_closest(embedding.vector.into(), count)?;
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
