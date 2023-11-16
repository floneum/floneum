#![allow(unused_macros)]

use once_cell::sync::Lazy;

use crate::exports::plugins::main::definitions::Guest;
pub use floneum_rust_macro::export_plugin;

#[doc(hidden)]
pub use inventory;

pub use plugins::main::types::*;
mod logging;
pub use logging::Logger;
mod state;
pub use state::*;
mod filesystem;
pub use filesystem::*;

wit_bindgen::generate!({
    path: "../wit",
    world: "both",
    exports: {
        world: ExportedDefinitions,
        "plugins:main/definitions": ExportedDefinitions,
    },
});

pub trait DynGuest {
    fn structure(&self) -> Definition;
    fn run(&self, inputs: Vec<Input>) -> Vec<Output>;
}

#[doc(hidden)]
pub struct LazyGuest(fn() -> Box<dyn DynGuest + Send + Sync>);

impl LazyGuest {
    pub const fn new(constructor: fn() -> Box<dyn DynGuest + Send + Sync>) -> Self {
        Self(constructor)
    }
}

inventory::collect!(LazyGuest);

static EXPORTED_DEFINITIONS: Lazy<Box<dyn DynGuest + Send + Sync>> = Lazy::new(|| {
    (inventory::iter::<LazyGuest>
        .into_iter()
        .next()
        .expect("no plugin exported")
        .0)()
});

struct ExportedDefinitions;

impl Guest for ExportedDefinitions {
    fn structure() -> Definition {
        EXPORTED_DEFINITIONS.structure()
    }

    fn run(inputs: Vec<Input>) -> Vec<Output> {
        EXPORTED_DEFINITIONS.run(inputs)
    }
}

pub trait IntoInputValue<T = ()> {
    fn into_input_value(self) -> Input;
}

pub trait IntoReturnValue<T = ()> {
    fn into_return_value(self) -> Output;
}

#[doc(hidden)]
pub struct OptionMarker;

impl<T: IntoReturnValue> IntoReturnValue<OptionMarker> for Option<T> {
    fn into_return_value(self) -> Output {
        match self {
            Some(x) => x.into_return_value(),
            None => Output::Halt,
        }
    }
}

impl<T: IntoPrimitiveValue> IntoReturnValue for T {
    fn into_return_value(self) -> Output {
        Output::Single(self.into_primitive_value())
    }
}

impl<T: IntoPrimitiveValue> IntoInputValue for T {
    fn into_input_value(self) -> Input {
        Input::Single(self.into_primitive_value())
    }
}

#[doc(hidden)]
pub struct VecMarker;

impl<T: IntoPrimitiveValue> IntoReturnValue<VecMarker> for Vec<T> {
    fn into_return_value(self) -> Output {
        Output::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
    }
}

impl<T: IntoPrimitiveValue> IntoInputValue for Vec<T> {
    fn into_input_value(self) -> Input {
        Input::Many(self.into_iter().map(|x| x.into_primitive_value()).collect())
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

impl IntoPrimitiveValue for Model {
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
    fn into_return_values(self) -> Vec<Output>;
}

impl<T: IntoReturnValue<I>, I> IntoReturnValues<I> for T {
    fn into_return_values(self) -> Vec<Output> {
        vec![self.into_return_value()]
    }
}

macro_rules! impl_into_return_values {
    (
        $($var:ident : $ty:ident $ty2:ident),*
    ) => {
        impl<$($ty: IntoReturnValue<$ty2>, $ty2, )*> IntoReturnValues<($($ty2,)*)> for ($($ty,)*) {
            fn into_return_values(self) -> Vec<Output> {
                let ($($var,)*) = self;
                vec![$($var.into_return_value(),)*]
            }
        }
    };
}

impl_into_return_values!();
impl_into_return_values!(a: A A2);
impl_into_return_values!(a: A A2, b: B B2);
impl_into_return_values!(a: A A2, b: B B2, c: C C2);
impl_into_return_values!(a: A A2, b: B B2, c: C C2, d: D D2);
impl_into_return_values!(a: A A2, b: B B2, c: C C2, d: D D2, e: E E2);
impl_into_return_values!(a: A A2, b: B B2, c: C C2, d: D D2, e: E E2, f: F F2);
impl_into_return_values!(a: A A2, b: B B2, c: C C2, d: D D2, e: E E2, f: F F2, g: G G2);
impl_into_return_values!(a: A A2, b: B B2, c: C C2, d: D D2, e: E E2, f: F F2, g: G G2, h: H H2);
