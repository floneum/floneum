#![allow(unused_macros)]

pub use crate::exports::plugins::main::definitions::{
    Definition, Definitions, Input, IoDefinition, Output, PrimitiveValue, PrimitiveValueType,
    ValueType,
};
use crate::plugins::main::imports::*;
pub use crate::plugins::main::imports::{get_request, Header};
pub use crate::plugins::main::types::{EmbeddingDbId, GptNeoXType, LlamaType, ModelType, MptType};
use crate::{plugins, IntoPrimitiveValue};
pub use floneum_rust_macro::export_plugin;
pub use plugins::main::types::Embedding;

pub struct VectorDatabase {
    pub id: EmbeddingDbId,
    drop: bool,
}

impl VectorDatabase {
    pub fn new(embeddings: &[plugins::main::types::Embedding], documents: &[String]) -> Self {
        let id = create_embedding_db(embeddings, documents);

        VectorDatabase { id, drop: true }
    }

    pub fn add_embedding(&self, embedding: &plugins::main::types::Embedding, document: &str) {
        add_embedding(self.id, embedding, document);
    }

    pub fn find_closest_documents(
        &self,
        embedding: &plugins::main::types::Embedding,
        count: usize,
    ) -> Vec<String> {
        find_closest_documents(self.id, embedding, count as u32)
    }

    pub fn find_documents_within(
        &self,
        embedding: &plugins::main::types::Embedding,
        distance: f32,
    ) -> Vec<String> {
        find_documents_within(self.id, embedding, distance)
    }

    pub fn from_id(id: EmbeddingDbId) -> Self {
        VectorDatabase { id, drop: false }
    }

    pub fn leak(self) -> EmbeddingDbId {
        let id = self.id;
        std::mem::forget(self);
        id
    }

    pub fn manually_drop(self) {
        remove_embedding_db(self.id);
    }
}

impl Drop for VectorDatabase {
    fn drop(&mut self) {
        if self.drop {
            log::trace!("Dropping vector database {}", self.id.id);
            remove_embedding_db(self.id);
        }
    }
}

impl crate::IntoPrimitiveValue for VectorDatabase {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Database(self.id)
    }
}

impl IntoPrimitiveValue for EmbeddingDbId {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Database(self)
    }
}
