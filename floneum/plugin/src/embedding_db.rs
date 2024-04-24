use std::ops::Deref;
use std::sync::Arc;

use crate::plugins::main::types::{Embedding, EmbeddingDbResource};
use crate::resource::ResourceStorage;

use kalosm::language::{Document, UnknownVectorSpace, VectorDB};
use once_cell::sync::Lazy;

impl ResourceStorage {
    pub(crate) fn impl_create_embedding_db(
        &self,
        embeddings: Vec<Embedding>,
        documents: Vec<String>,
    ) -> anyhow::Result<EmbeddingDbResource> {
        let documents = documents
            .into_iter()
            .map(|x| Document::from_parts(String::new(), x));
        let mut db = VectorDBWithDocuments::new();

        for (embedding, document) in embeddings.into_iter().zip(documents.into_iter()) {
            db.add_embedding(embedding, document)?;
        }

        let idx = self.insert(db);
        Ok(EmbeddingDbResource {
            id: idx.index() as u64,
            owned: true,
        })
    }

    pub(crate) async fn impl_add_embedding(
        &self,
        self_: EmbeddingDbResource,
        embedding: Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        let index = self_.into();
        self.get_mut(index)
            .ok_or(anyhow::anyhow!(
                "DB not found; It may have been already dropped"
            ))?
            .add_embedding(embedding, Document::from_parts(String::new(), document))?;
        Ok(())
    }

    pub(crate) async fn impl_find_closest_documents(
        &self,
        self_: EmbeddingDbResource,
        search: Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        let index = self_.into();
        let db = self.get(index).ok_or(anyhow::anyhow!(
            "DB not found; It may have been already dropped"
        ))?;
        let documents = db.get_closest(search, count as usize)?;
        Ok(documents
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    pub(crate) fn impl_drop_embedding_db(&self, rep: EmbeddingDbResource) -> wasmtime::Result<()> {
        let index = rep.into();
        self.drop_key(index);
        Ok(())
    }
}

pub(crate) struct VectorDBWithDocuments {
    db: Lazy<Result<VectorDB<UnknownVectorSpace>, Arc<heed::Error>>>,
    documents: Vec<Option<Document>>,
}

impl Default for VectorDBWithDocuments {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorDBWithDocuments {
    pub fn new() -> Self {
        Self {
            db: Lazy::new(|| VectorDB::new().map_err(Arc::new)),
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
            .as_ref()
            .map_err(Clone::clone)?
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
            .as_ref()
            .map_err(Clone::clone)?
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
