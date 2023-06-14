#![allow(unused_macros)]

use crate::plugins::main::imports::*;

pub use crate::exports::plugins::main::definitions::{
    Definition, Definitions, IoDefinition, PrimitiveValue, PrimitiveValueType, Value, ValueType,
};
pub use crate::plugins::main::imports::{print, GptNeoXType, LlamaType, ModelType, MptType};
pub use plugins::main::types::Embedding;
use plugins::main::types::{JsonKvPair, JsonStructureEither, JsonStructureMap,NumberParameters,UnsignedRange};

wit_bindgen::generate!({path: "../wit", macro_export});

pub struct VectorDatabase {
    id: EmbeddingDbId,
}

impl VectorDatabase {
    pub fn new(embeddings: &[&plugins::main::types::Embedding], documents: &[&str]) -> Self {
        let id = create_embedding_db(embeddings, documents);

        VectorDatabase { id }
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
}

impl Drop for VectorDatabase {
    fn drop(&mut self) {
        let id = self.id;
        remove_embedding_db(id);
    }
}

pub struct ModelInstance {
    id: ModelId,
}

impl ModelInstance {
    pub fn new(ty: ModelType) -> Self {
        let id = load_model(ty);

        ModelInstance { id }
    }

    pub fn infer(&self, input: &str, max_tokens: Option<u32>, stop_on: Option<&str>) -> String {
        infer(self.id, input, max_tokens, stop_on)
    }

    pub fn infer_structured(
        &self,
        input: &str,
        max_tokens: Option<u32>,
        structure: Structured,
    ) -> String {
        infer_structured(self.id, input, max_tokens, structure.id)
    }

    pub fn get_embedding(&self, text: &str) -> Embedding {
        get_embedding(self.id, text)
    }
}

impl Drop for ModelInstance {
    fn drop(&mut self) {
        let id = self.id;
        unload_model(id);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Structured {
    id: StructureId,
}

impl Structured {
    pub fn sequence_of(item: Structured) -> Self {
        let inner = JsonStructure::Sequence(item.id);
        let id = create_json_structure(inner);
        Structured { id }
    }

    pub fn map_of(items: Vec<(String, Structured)>) -> Self {
        let items: Vec<_> = items
            .iter()
            .map(|(k, v)| JsonKvPair {
                key: &k,
                value: v.id,
            })
            .collect();
        let inner = JsonStructure::Map(JsonStructureMap { items: &items });
        let id = create_json_structure(inner);
        Structured { id }
    }

    pub fn float() -> Self {
        Self::ranged_float(f64::MIN, f64::MAX)
    }

    pub fn ranged_float(min: f64, max: f64) -> Self {
        Self::number(min, max, false)
    }

    pub fn int() -> Self {
        Self::ranged_int(f64::MIN, f64::MAX)
    }

    pub fn ranged_int(min: f64, max: f64) -> Self {
        Self::number(min, max, true)
    }

    pub fn number(min: f64, max: f64, int: bool) -> Self {
        let inner = JsonStructure::Num(
            NumberParameters{
                min,
                max,
                integer: int,
            }
        );
        let id = create_json_structure(inner);
        Structured { id }
    }

    pub fn str() -> Self {
        Self::ranged_str(0, u64::MAX)
    }

    pub fn ranged_str(min_len:u64, max_len:u64) -> Self {
        let inner = JsonStructure::Str(
            UnsignedRange{
                min: min_len ,
                max: max_len ,
            }
        );
        let id = create_json_structure(inner);
        Structured { id }
    }

    pub fn boolean() -> Self {
        let inner = JsonStructure::Boolean;
        let id = create_json_structure(inner);
        Structured { id }
    }

    pub fn null() -> Self {
        let null = JsonStructure::Null;
        let id = create_json_structure(null);
        Structured { id }
    }

    pub fn or_not(self) -> Self {
        let null = Self::null();
        Self::either(self, null)
    }

    pub fn either(first: Structured, second: Structured) -> Self {
        let inner = JsonStructure::Either(JsonStructureEither {
            first: first.id,
            second: second.id,
        });
        let id = create_json_structure(inner);
        Structured { id }
    }
}
