#![allow(unused_macros)]

pub use crate::exports::plugins::main::definitions::{
    Definition, Definitions, IoDefinition, PrimitiveValue, PrimitiveValueType, Value, ValueType,
};
use crate::plugins::main::imports::*;
pub use crate::plugins::main::imports::{
    EmbeddingDbId, GptNeoXType, LlamaType, ModelType, MptType,
};
pub use floneum_rust_macro::export_plugin;
pub use plugins::main::types::Embedding;
use plugins::main::types::{
    EitherStructure, NumberParameters, SequenceParameters, Structure, ThenStructure, UnsignedRange,
};
use std::ops::RangeInclusive;

wit_bindgen::generate!({path: "../wit", macro_export});

pub struct VectorDatabase {
    id: EmbeddingDbId,
    drop: bool,
}

impl VectorDatabase {
    pub fn new(embeddings: &[&plugins::main::types::Embedding], documents: &[&str]) -> Self {
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
            println!("Dropping vector database {}", self.id.id);
            remove_embedding_db(self.id);
        }
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
    pub fn literal(text: &str) -> Self {
        let inner = Structure::Literal(text);

        let id = create_structure(inner);
        Structured { id }
    }

    pub fn sequence_of(
        item: Structured,
        seperator: Structured,
        range: RangeInclusive<u64>,
    ) -> Self {
        let inner = Structure::Sequence(SequenceParameters {
            item: item.id,
            seperator: seperator.id,
            min_len: *range.start(),
            max_len: *range.end(),
        });
        let id = create_structure(inner);
        Structured { id }
    }

    pub fn float() -> Self {
        Self::ranged_float(f64::MIN..=f64::MAX)
    }

    pub fn ranged_float(range: RangeInclusive<f64>) -> Self {
        Self::number(range, false)
    }

    pub fn int() -> Self {
        Self::ranged_int(f64::MIN..=f64::MAX)
    }

    pub fn ranged_int(range: RangeInclusive<f64>) -> Self {
        Self::number(range, true)
    }

    pub fn number(range: RangeInclusive<f64>, int: bool) -> Self {
        let inner = Structure::Num(NumberParameters {
            min: *range.start(),
            max: *range.end(),
            integer: int,
        });
        let id = create_structure(inner);
        Structured { id }
    }

    pub fn str() -> Self {
        Self::ranged_str(0, u64::MAX)
    }

    pub fn ranged_str(min_len: u64, max_len: u64) -> Self {
        let inner = Structure::Str(UnsignedRange {
            min: min_len,
            max: max_len,
        });
        let id = create_structure(inner);
        Structured { id }
    }

    pub fn boolean() -> Self {
        Self::literal("true").or(Self::literal("false"))
    }

    pub fn null() -> Self {
        Self::literal("null")
    }

    pub fn or_not(self) -> Self {
        self.or(Self::null())
    }

    pub fn or(self, second: Structured) -> Self {
        let inner = Structure::Or(EitherStructure {
            first: self.id,
            second: second.id,
        });
        let id = create_structure(inner);
        Structured { id }
    }

    pub fn then(self, then: Structured) -> Self {
        let inner = Structure::Then(ThenStructure {
            first: self.id,
            second: then.id,
        });
        let id = create_structure(inner);
        Structured { id }
    }
}

trait IntoReturnValue {
    fn into_return_value(self) -> Value;
}

impl<T: IntoPrimitiveValue> IntoReturnValue for T {
    fn into_return_value(self) -> Value {
        Value::Single(self.into_primitive_value())
    }
}

impl IntoReturnValue for Vec<i64> {
    fn into_return_value(self) -> Value {
        Value::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

impl IntoReturnValue for Vec<String> {
    fn into_return_value(self) -> Value {
        Value::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

impl IntoReturnValue for Vec<ModelInstance> {
    fn into_return_value(self) -> Value {
        Value::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

impl IntoReturnValue for Vec<VectorDatabase> {
    fn into_return_value(self) -> Value {
        Value::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

impl IntoReturnValue for Vec<EmbeddingDbId> {
    fn into_return_value(self) -> Value {
        Value::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

impl IntoReturnValue for Vec<Embedding> {
    fn into_return_value(self) -> Value {
        Value::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

trait IntoPrimitiveValue {
    fn into_primitive_value(self) -> PrimitiveValue;
}

impl IntoPrimitiveValue for i64 {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Number(self)
    }
}

impl IntoPrimitiveValue for String {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Text(self)
    }
}

impl IntoPrimitiveValue for ModelInstance {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Model(self.id)
    }
}

impl IntoPrimitiveValue for VectorDatabase {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Database(self.id)
    }
}

impl IntoPrimitiveValue for EmbeddingDbId {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Database(self)
    }
}

impl IntoPrimitiveValue for Embedding {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Embedding(self)
    }
}

pub trait IntoReturnValues {
    fn into_return_values(self) -> Vec<Value>;
}

impl<T: IntoReturnValue> IntoReturnValues for T {
    fn into_return_values(self) -> Vec<Value> {
        vec![self.into_return_value()]
    }
}

macro_rules! impl_into_return_values {
    (
        $($var:ident : $ty:ident),*
    ) => {
        impl<$($ty: IntoReturnValue,)*> IntoReturnValues for ($($ty,)*) {
            fn into_return_values(self) -> Vec<Value> {
                let ($($var,)*) = self;
                vec![$($var.into_return_value(),)*]
            }
        }
    };
}

impl_into_return_values!();
impl_into_return_values!(a: A);
impl_into_return_values!(a: A, b: B);
impl_into_return_values!(a: A, b: B, c: C);
impl_into_return_values!(a: A, b: B, c: C, d: D);
impl_into_return_values!(a: A, b: B, c: C, d: D, e: E);
impl_into_return_values!(a: A, b: B, c: C, d: D, e: E, f: F);
impl_into_return_values!(a: A, b: B, c: C, d: D, e: E, f: F, g: G);
impl_into_return_values!(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H);
impl_into_return_values!(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H, i: I);
impl_into_return_values!(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H, i: I, j: J);
impl_into_return_values!(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
    k: K
);
