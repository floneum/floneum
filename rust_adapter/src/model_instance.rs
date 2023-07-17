pub use crate::exports::plugins::main::definitions::{
    Definition, Definitions, Input, IoDefinition, Output, PrimitiveValue, PrimitiveValueType,
    ValueType,
};
use crate::plugins::main::imports::*;
pub use crate::plugins::main::imports::{get_request, Header};
pub use crate::plugins::main::types::Embedding;
pub use crate::plugins::main::types::{EmbeddingDbId, GptNeoXType, LlamaType, ModelType, MptType};
use crate::Structured;
pub use floneum_rust_macro::export_plugin;

pub struct ModelInstance {
    pub(crate) id: ModelId,
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

impl crate::IntoPrimitiveValue for ModelInstance {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Model(self.id)
    }
}
