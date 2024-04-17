use crate::bindings::plugins::main::types::*;

pub trait IntoInputValue<T = ()> {
    fn into_input_value(self) -> Vec<PrimitiveValue>;
}

pub trait IntoReturnValue<T = ()> {
    fn into_return_value(self) -> Vec<PrimitiveValue>;
}

impl<T: IntoPrimitiveValue> IntoReturnValue for T {
    fn into_return_value(self) -> Vec<PrimitiveValue> {
        vec![self.into_primitive_value()]
    }
}

impl<T: IntoPrimitiveValue> IntoInputValue for T {
    fn into_input_value(self) -> Vec<PrimitiveValue> {
        vec![self.into_primitive_value()]
    }
}

#[doc(hidden)]
pub struct VecMarker;

impl<T: IntoPrimitiveValue> IntoReturnValue<VecMarker> for Vec<T> {
    fn into_return_value(self) -> Vec<PrimitiveValue> {
        self.into_iter().map(|x| x.into_primitive_value()).collect()
    }
}

impl<T: IntoPrimitiveValue> IntoInputValue for Vec<T> {
    fn into_input_value(self) -> Vec<PrimitiveValue> {
        self.into_iter().map(|x| x.into_primitive_value()).collect()
    }
}

trait IntoPrimitiveValue {
    fn into_primitive_value(self) -> PrimitiveValue;
}

impl IntoPrimitiveValue for PrimitiveValue {
    fn into_primitive_value(self) -> PrimitiveValue {
        self
    }
}

impl IntoPrimitiveValue for ModelType {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::ModelType(self)
    }
}

impl IntoPrimitiveValue for TextGenerationModel {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Model(self)
    }
}

impl IntoPrimitiveValue for EmbeddingModel {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::EmbeddingModel(self)
    }
}

impl IntoPrimitiveValue for EmbeddingModelType {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::EmbeddingModelType(self)
    }
}

impl IntoPrimitiveValue for Node {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Node(self)
    }
}

impl IntoPrimitiveValue for Page {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Page(self)
    }
}

impl IntoPrimitiveValue for bool {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Boolean(self)
    }
}

impl IntoPrimitiveValue for i64 {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Number(self)
    }
}

impl IntoPrimitiveValue for f64 {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Float(self)
    }
}

impl IntoPrimitiveValue for String {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Text(self)
    }
}

impl IntoPrimitiveValue for Embedding {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Embedding(self)
    }
}

impl IntoPrimitiveValue for EmbeddingDb {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Database(self)
    }
}

pub trait IntoReturnValues<T = ()> {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>>;
}

impl<T: IntoReturnValue<I>, I> IntoReturnValues<I> for T {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        vec![self.into_return_value()]
    }
}

#[doc(hidden)]
pub struct UnitMarker;

impl IntoReturnValues<UnitMarker> for () {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        vec![]
    }
}

impl<A: IntoReturnValue<A2>, A2> IntoReturnValues<(A2,)> for (A,) {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a,) = self;
        vec![a.into_return_value()]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2> IntoReturnValues<(A2, B2)> for (A, B) {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b) = self;
        vec![a.into_return_value(), b.into_return_value()]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2, C: IntoReturnValue<C2>, C2>
    IntoReturnValues<(A2, B2, C2)> for (A, B, C)
{
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b, c) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
        ]
    }
}

impl<
        A: IntoReturnValue<A2>,
        A2,
        B: IntoReturnValue<B2>,
        B2,
        C: IntoReturnValue<C2>,
        C2,
        D: IntoReturnValue<D2>,
        D2,
    > IntoReturnValues<(A2, B2, C2, D2)> for (A, B, C, D)
{
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b, c, d) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
            d.into_return_value(),
        ]
    }
}

impl<
        A: IntoReturnValue<A2>,
        A2,
        B: IntoReturnValue<B2>,
        B2,
        C: IntoReturnValue<C2>,
        C2,
        D: IntoReturnValue<D2>,
        D2,
        E: IntoReturnValue<E2>,
        E2,
    > IntoReturnValues<(A2, B2, C2, D2, E2)> for (A, B, C, D, E)
{
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b, c, d, e) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
            d.into_return_value(),
            e.into_return_value(),
        ]
    }
}

/// A wrapper around a file path.
pub struct File(std::path::PathBuf);

impl std::ops::Deref for File {
    type Target = std::path::PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<std::path::PathBuf> for File {
    fn from(path: std::path::PathBuf) -> Self {
        Self(path)
    }
}

impl From<String> for File {
    fn from(path: String) -> Self {
        Self(std::path::PathBuf::from(path))
    }
}

impl IntoPrimitiveValue for File {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::File(self.0.display().to_string())
    }
}

/// A wrapper around a folder path.
pub struct Folder(std::path::PathBuf);

impl std::ops::Deref for Folder {
    type Target = std::path::PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<std::path::PathBuf> for Folder {
    fn from(path: std::path::PathBuf) -> Self {
        Self(path)
    }
}

impl From<String> for Folder {
    fn from(path: String) -> Self {
        Self(std::path::PathBuf::from(path))
    }
}

impl IntoPrimitiveValue for Folder {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Folder(self.0.display().to_string())
    }
}
