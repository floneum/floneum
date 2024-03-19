#[macro_export]
macro_rules! bindgen {
    ($name:ident) => {
        #[allow(clippy::all)]
        mod bindings {
            ::wit_bindgen::generate!({
                inline: "package plugins:main;

interface imports {
  use types.{embedding, model, model-type, embedding-db, node, page};
  
  store: func(key: list<u8>, value: list<u8>);

  load: func(key: list<u8>) -> list<u8>;

  unload: func(key: list<u8>);

  log-to-user: func(information: string);
}

interface types {
  record header {
    key: string,
    value: string,
  }

  get-request: func(url: string, headers: list<header>) -> string;

  enum browser-mode {
    headless,
    headfull,
  }

  resource page {
    constructor(mode: browser-mode, url: string);
    find-in-current-page: func(selector: string) -> node;
    screenshot-browser: func() -> list<u8>;
    html: func() -> string;
  }

  resource node {
    get-element-text: func() -> string;
    click-element: func();
    type-into-element: func(keys: string);
    get-element-outer-html: func() -> string;
    screenshot-element: func() -> list<u8>;
    find-child-of-element: func(selector: string) -> node;
  }

  resource embedding-db {
    constructor(embeddings: list<embedding>, documents: list<string>);
    add-embedding: func(embedding: embedding, documents: string);
    find-closest-documents: func(search: embedding, count: u32) -> list<string>;
  }

  resource model {
    constructor(ty: model-type);
    model-downloaded: static func(ty: model-type) -> bool;
    infer: func(input: string, max-tokens: option<u32>, stop-on: option<string>) -> string;
    infer-structured: func(input: string, regex: string) -> string;
  }

  resource embedding-model {
    constructor(ty: embedding-model-type);
    model-downloaded: static func(ty: embedding-model-type) -> bool;
    get-embedding: func(document: string) -> embedding;
  }

  record embedding {
    vector: list<float32>
  }

  variant primitive-value {
    model(model),
    embedding-model(embedding-model),
    model-type(model-type),
    embedding-model-type(embedding-model-type),
    database(embedding-db),
    number(s64),
    text(string),
    file(string),
    folder(string),
    embedding(embedding),
    boolean(bool),
    page(page),
    node(node)
  }

  variant borrowed-primitive-value {
    model(borrow<model>),
    embedding-model(borrow<embedding-model>),
    model-type(model-type),
    embedding-model-type(embedding-model-type),
    database(borrow<embedding-db>),
    number(s64),
    text(string),
    file(string),
    folder(string),
    embedding(embedding),
    boolean(bool),
    page(page),
    node(node)
  }

  variant value-type {
    single(primitive-value-type),
    many(primitive-value-type),
  }

  enum primitive-value-type {
    number,
    text,
    file,
    folder,
    embedding,
    database,
    model,
    embedding-model,
    model-type,
    embedding-model-type,
    boolean,
    page,
    node,
    any
  }

  record definition {
    name: string,
    description: string,
    inputs: list<io-definition>,
    outputs: list<io-definition>,
    examples: list<example>
  }

  record example {
    name: string,
    inputs: list<list<borrowed-primitive-value>>,
    outputs: list<list<primitive-value>>,
  }

  record io-definition {
    name: string,
    ty: value-type,
  }

  variant model-type {
    mistral-seven,
    mistral-seven-instruct,
    mistral-seven-instruct-two,
    zephyr-seven-alpha,
    zephyr-seven-beta,
    open-chat-seven,
    starling-seven-alpha,
    tiny-llama-chat,
    tiny-llama,
    llama-seven,
    llama-thirteen,
    llama-seventy,
    llama-seven-chat,
    llama-thirteen-chat,
    llama-seventy-chat,
    llama-seven-code,
    llama-thirteen-code,
    llama-thirty-four-code,
    solar-ten,
    solar-ten-instruct,
    phi-one,
    phi-one-point-five,
    phi-two,
    puffin-phi-two,
    dolphin-phi-two
  }
  variant embedding-model-type { bert }
}

interface definitions {
  use types.{definition, borrowed-primitive-value, primitive-value};

  structure: func() -> definition;

  run: func(inputs: list<list<borrowed-primitive-value>>) -> list<list<primitive-value>>;
}

world exports {
  import imports;
  import types;
}

world plugin-world {
  export definitions;
  import imports;
  import types;
}

world both {
  import imports;
  export definitions;
}
",
                world: "plugin-world",
            });
        }
        use bindings::*;

use plugins::main::types::*;
use exports::plugins::main::definitions::Guest;
use plugins::main::imports::log_to_user;

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

#[doc(hidden)]
pub struct UnitMarker;

impl IntoReturnValues<UnitMarker> for () {
    fn into_return_values(self) -> Vec<Output> {
        vec![]
    }
}

impl<A: IntoReturnValue<A2>, A2> IntoReturnValues<(A2,)> for (A,) {
    fn into_return_values(self) -> Vec<Output> {
        let (a,) = self;
        vec![a.into_return_value()]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2> IntoReturnValues<(A2, B2)>
    for (A, B)
{
    fn into_return_values(self) -> Vec<Output> {
        let (a, b) = self;
        vec![a.into_return_value(), b.into_return_value()]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2, C: IntoReturnValue<C2>, C2>
    IntoReturnValues<(A2, B2, C2)> for (A, B, C)
{
    fn into_return_values(self) -> Vec<Output> {
        let (a, b, c) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
        ]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2, C: IntoReturnValue<C2>, C2, D: IntoReturnValue<D2>, D2>
    IntoReturnValues<(A2, B2, C2, D2)> for (A, B, C, D)
{
    fn into_return_values(self) -> Vec<Output> {
        let (a, b, c, d) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
            d.into_return_value(),
        ]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2, C: IntoReturnValue<C2>, C2, D: IntoReturnValue<D2>, D2, E: IntoReturnValue<E2>, E2>
    IntoReturnValues<(A2, B2, C2, D2, E2)> for (A, B, C, D, E)
{
    fn into_return_values(self) -> Vec<Output> {
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

    };
}
