use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wasmtime::component::Resource;

use crate::plugins::main::{self, types::*};
use main::types::PrimitiveValueType;

#[derive(serde::Serialize, serde::Deserialize)]
enum MyValue {
    Single(MyPrimitiveValue),
    List(Vec<MyPrimitiveValue>),
    Unset,
}

impl From<&Input> for MyValue {
    fn from(value: &Input) -> Self {
        match value {
            Input::Single(value) => MyValue::Single(value.into()),
            Input::Many(values) => MyValue::List(values.iter().map(|v| v.into()).collect()),
        }
    }
}

impl From<MyValue> for Input {
    fn from(value: MyValue) -> Self {
        match value {
            MyValue::Single(value) => Input::Single(value.into()),
            MyValue::List(values) => Input::Many(values.into_iter().map(|v| v.into()).collect()),
            MyValue::Unset => Input::Many(Vec::new()),
        }
    }
}

impl Serialize for Input {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MyValue::from(self).serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for Input {
    fn deserialize<D: serde::Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        MyValue::deserialize(deserializer).map(|v| v.into())
    }
}

impl PartialEq for Input {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Input::Single(a), Input::Single(b)) => a == b,
            (Input::Many(a), Input::Many(b)) => a == b,
            _ => false,
        }
    }
}

impl From<&Output> for MyValue {
    fn from(output: &Output) -> Self {
        match output {
            Output::Single(value) => MyValue::Single(value.into()),
            Output::Many(values) => MyValue::List(values.iter().map(|v| v.into()).collect()),
            Output::Halt => MyValue::Unset,
        }
    }
}

impl From<MyValue> for Output {
    fn from(value: MyValue) -> Self {
        match value {
            MyValue::Single(value) => Output::Single(value.into()),
            MyValue::List(values) => Output::Many(values.into_iter().map(|v| v.into()).collect()),
            MyValue::Unset => Output::Halt,
        }
    }
}

impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MyValue::from(self).serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for Output {
    fn deserialize<D: serde::Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        MyValue::deserialize(deserializer).map(|v| v.into())
    }
}

impl PartialEq for Output {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Output::Single(a), Output::Single(b)) => a == b,
            (Output::Many(a), Output::Many(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialEq for PrimitiveValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PrimitiveValue::Number(a), PrimitiveValue::Number(b)) => a == b,
            (PrimitiveValue::Float(a), PrimitiveValue::Float(b)) => a == b,
            (PrimitiveValue::Text(a), PrimitiveValue::Text(b)) => a == b,
            (PrimitiveValue::File(a), PrimitiveValue::File(b)) => a == b,
            (PrimitiveValue::Folder(a), PrimitiveValue::Folder(b)) => a == b,
            (PrimitiveValue::Embedding(a), PrimitiveValue::Embedding(b)) => a.vector == b.vector,
            (PrimitiveValue::Database(a), PrimitiveValue::Database(b)) => a.rep() == b.rep(),
            (PrimitiveValue::Model(a), PrimitiveValue::Model(b)) => a.rep() == b.rep(),
            (PrimitiveValue::EmbeddingModel(a), PrimitiveValue::EmbeddingModel(b)) => {
                a.rep() == b.rep()
            }
            (PrimitiveValue::ModelType(a), PrimitiveValue::ModelType(b)) => a == b,
            (PrimitiveValue::Boolean(a), PrimitiveValue::Boolean(b)) => a == b,
            (PrimitiveValue::Page(a), PrimitiveValue::Page(b)) => a.rep() == b.rep(),
            (PrimitiveValue::Node(a), PrimitiveValue::Node(b)) => a.rep() == b.rep(),
            _ => false,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
enum MyPrimitiveValue {
    Number(i64),
    Float(f64),
    Text(String),
    File(String),
    Folder(String),
    Embedding(Vec<f32>),
    Model(u32),
    EmbeddingModel(u32),
    Database(u32),
    ModelType(MyModelType),
    EmbeddingModelType(MyEmbeddingModelType),
    Boolean(bool),
    Page(u32),
    Node(u32),
}

impl From<&PrimitiveValue> for MyPrimitiveValue {
    fn from(value: &PrimitiveValue) -> Self {
        match value {
            PrimitiveValue::Number(value) => MyPrimitiveValue::Number(*value),
            PrimitiveValue::Float(value) => MyPrimitiveValue::Float(*value),
            PrimitiveValue::Text(value) => MyPrimitiveValue::Text(value.clone()),
            PrimitiveValue::File(value) => MyPrimitiveValue::File(value.clone()),
            PrimitiveValue::Folder(value) => MyPrimitiveValue::Folder(value.clone()),
            PrimitiveValue::Embedding(value) => MyPrimitiveValue::Embedding(value.vector.clone()),
            PrimitiveValue::Model(value) => MyPrimitiveValue::Model(value.rep()),
            PrimitiveValue::EmbeddingModel(value) => MyPrimitiveValue::Model(value.rep()),
            PrimitiveValue::Database(value) => MyPrimitiveValue::Database(value.rep()),
            PrimitiveValue::ModelType(value) => MyPrimitiveValue::ModelType(value.into()),
            PrimitiveValue::EmbeddingModelType(value) => {
                MyPrimitiveValue::EmbeddingModelType(value.into())
            }
            PrimitiveValue::Boolean(value) => MyPrimitiveValue::Boolean(*value),
            PrimitiveValue::Page(value) => MyPrimitiveValue::Page(value.rep()),
            PrimitiveValue::Node(value) => MyPrimitiveValue::Node(value.rep()),
        }
    }
}

impl From<MyPrimitiveValue> for PrimitiveValue {
    fn from(value: MyPrimitiveValue) -> Self {
        match value {
            MyPrimitiveValue::Number(value) => PrimitiveValue::Number(value),
            MyPrimitiveValue::Float(value) => PrimitiveValue::Float(value),
            MyPrimitiveValue::Text(value) => PrimitiveValue::Text(value),
            MyPrimitiveValue::File(value) => PrimitiveValue::File(value),
            MyPrimitiveValue::Folder(value) => PrimitiveValue::Folder(value),
            MyPrimitiveValue::Embedding(value) => {
                PrimitiveValue::Embedding(Embedding { vector: value })
            }
            MyPrimitiveValue::Model(value) => PrimitiveValue::Model(Resource::new_own(value)),
            MyPrimitiveValue::EmbeddingModel(value) => {
                PrimitiveValue::EmbeddingModel(Resource::new_own(value))
            }
            MyPrimitiveValue::ModelType(value) => PrimitiveValue::ModelType(value.into()),
            MyPrimitiveValue::EmbeddingModelType(value) => {
                PrimitiveValue::EmbeddingModelType(value.into())
            }
            MyPrimitiveValue::Database(value) => PrimitiveValue::Database(Resource::new_own(value)),
            MyPrimitiveValue::Boolean(value) => PrimitiveValue::Boolean(value),
            MyPrimitiveValue::Page(value) => PrimitiveValue::Page(Resource::new_own(value)),
            MyPrimitiveValue::Node(value) => PrimitiveValue::Node(Resource::new_own(value)),
        }
    }
}

#[derive(Serialize, Deserialize)]
enum MyEmbeddingModelType {
    Bert,
}

impl From<&EmbeddingModelType> for MyEmbeddingModelType {
    fn from(value: &EmbeddingModelType) -> Self {
        match value {
            EmbeddingModelType::Bert => MyEmbeddingModelType::Bert,
        }
    }
}

impl From<MyEmbeddingModelType> for EmbeddingModelType {
    fn from(value: MyEmbeddingModelType) -> Self {
        match value {
            MyEmbeddingModelType::Bert => EmbeddingModelType::Bert,
        }
    }
}

impl PartialEq for EmbeddingModelType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (EmbeddingModelType::Bert, EmbeddingModelType::Bert)
        )
    }
}

#[derive(Serialize, Deserialize)]
enum MyModelType {
    MistralSeven,
    MistralSevenInstruct,
    MistralSevenInstructTwo,
    ZephyrSevenAlpha,
    ZephyrSevenBeta,
    OpenChatSeven,
    StarlingSevenAlpha,
    TinyLlamaChat,
    TinyLlama,
    LlamaSeven,
    LlamaThirteen,
    LlamaSeventy,
    LlamaSevenChat,
    LlamaThirteenChat,
    LlamaSeventyChat,
    LlamaSevenCode,
    LlamaThirteenCode,
    LlamaThirtyFourCode,
    SolarTen,
    SolarTenInstruct,
    PhiOne,
    PhiOnePointFive,
    PhiTwo,
    PuffinPhiTwo,
    DolphinPhiTwo,
}

impl From<&ModelType> for MyModelType {
    fn from(value: &ModelType) -> Self {
        match value {
            main::types::ModelType::MistralSeven => MyModelType::MistralSeven,
            main::types::ModelType::MistralSevenInstruct => MyModelType::MistralSevenInstruct,
            main::types::ModelType::MistralSevenInstructTwo => MyModelType::MistralSevenInstructTwo,
            main::types::ModelType::ZephyrSevenAlpha => MyModelType::ZephyrSevenAlpha,
            main::types::ModelType::ZephyrSevenBeta => MyModelType::ZephyrSevenBeta,
            main::types::ModelType::OpenChatSeven => MyModelType::OpenChatSeven,
            main::types::ModelType::StarlingSevenAlpha => MyModelType::StarlingSevenAlpha,
            main::types::ModelType::TinyLlamaChat => MyModelType::TinyLlamaChat,
            main::types::ModelType::TinyLlama => MyModelType::TinyLlama,
            main::types::ModelType::LlamaSeven => MyModelType::LlamaSeven,
            main::types::ModelType::LlamaThirteen => MyModelType::LlamaThirteen,
            main::types::ModelType::LlamaSeventy => MyModelType::LlamaSeventy,
            main::types::ModelType::LlamaSevenChat => MyModelType::LlamaSevenChat,
            main::types::ModelType::LlamaThirteenChat => MyModelType::LlamaThirteenChat,
            main::types::ModelType::LlamaSeventyChat => MyModelType::LlamaSeventyChat,
            main::types::ModelType::LlamaSevenCode => MyModelType::LlamaSevenCode,
            main::types::ModelType::LlamaThirteenCode => MyModelType::LlamaThirteenCode,
            main::types::ModelType::LlamaThirtyFourCode => MyModelType::LlamaThirtyFourCode,
            main::types::ModelType::SolarTen => MyModelType::SolarTen,
            main::types::ModelType::SolarTenInstruct => MyModelType::SolarTenInstruct,
            main::types::ModelType::PhiOne => MyModelType::PhiOne,
            main::types::ModelType::PhiOnePointFive => MyModelType::PhiOnePointFive,
            main::types::ModelType::PhiTwo => MyModelType::PhiTwo,
            main::types::ModelType::PuffinPhiTwo => MyModelType::PuffinPhiTwo,
            main::types::ModelType::DolphinPhiTwo => MyModelType::DolphinPhiTwo,
        }
    }
}

impl From<MyModelType> for ModelType {
    fn from(value: MyModelType) -> Self {
        match value {
            MyModelType::MistralSeven => main::types::ModelType::MistralSeven,
            MyModelType::MistralSevenInstruct => main::types::ModelType::MistralSevenInstruct,
            MyModelType::MistralSevenInstructTwo => main::types::ModelType::MistralSevenInstructTwo,
            MyModelType::ZephyrSevenAlpha => main::types::ModelType::ZephyrSevenAlpha,
            MyModelType::ZephyrSevenBeta => main::types::ModelType::ZephyrSevenBeta,
            MyModelType::OpenChatSeven => main::types::ModelType::OpenChatSeven,
            MyModelType::StarlingSevenAlpha => main::types::ModelType::StarlingSevenAlpha,
            MyModelType::TinyLlamaChat => main::types::ModelType::TinyLlamaChat,
            MyModelType::TinyLlama => main::types::ModelType::TinyLlama,
            MyModelType::LlamaSeven => main::types::ModelType::LlamaSeven,
            MyModelType::LlamaThirteen => main::types::ModelType::LlamaThirteen,
            MyModelType::LlamaSeventy => main::types::ModelType::LlamaSeventy,
            MyModelType::LlamaSevenChat => main::types::ModelType::LlamaSevenChat,
            MyModelType::LlamaThirteenChat => main::types::ModelType::LlamaThirteenChat,
            MyModelType::LlamaSeventyChat => main::types::ModelType::LlamaSeventyChat,
            MyModelType::LlamaSevenCode => main::types::ModelType::LlamaSevenCode,
            MyModelType::LlamaThirteenCode => main::types::ModelType::LlamaThirteenCode,
            MyModelType::LlamaThirtyFourCode => main::types::ModelType::LlamaThirtyFourCode,
            MyModelType::SolarTen => main::types::ModelType::SolarTen,
            MyModelType::SolarTenInstruct => main::types::ModelType::SolarTenInstruct,
            MyModelType::PhiOne => main::types::ModelType::PhiOne,
            MyModelType::PhiOnePointFive => main::types::ModelType::PhiOnePointFive,
            MyModelType::PhiTwo => main::types::ModelType::PhiTwo,
            MyModelType::PuffinPhiTwo => main::types::ModelType::PuffinPhiTwo,
            MyModelType::DolphinPhiTwo => main::types::ModelType::DolphinPhiTwo,
        }
    }
}

impl PartialEq for ModelType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (
                main::types::ModelType::MistralSeven,
                main::types::ModelType::MistralSeven
            ) | (
                main::types::ModelType::MistralSevenInstruct,
                main::types::ModelType::MistralSevenInstruct
            ) | (
                main::types::ModelType::MistralSevenInstructTwo,
                main::types::ModelType::MistralSevenInstructTwo
            ) | (
                main::types::ModelType::ZephyrSevenAlpha,
                main::types::ModelType::ZephyrSevenAlpha
            ) | (
                main::types::ModelType::ZephyrSevenBeta,
                main::types::ModelType::ZephyrSevenBeta
            ) | (
                main::types::ModelType::OpenChatSeven,
                main::types::ModelType::OpenChatSeven
            ) | (
                main::types::ModelType::StarlingSevenAlpha,
                main::types::ModelType::StarlingSevenAlpha
            ) | (
                main::types::ModelType::TinyLlamaChat,
                main::types::ModelType::TinyLlamaChat
            ) | (
                main::types::ModelType::TinyLlama,
                main::types::ModelType::TinyLlama
            ) | (
                main::types::ModelType::LlamaSeven,
                main::types::ModelType::LlamaSeven
            ) | (
                main::types::ModelType::LlamaThirteen,
                main::types::ModelType::LlamaThirteen
            ) | (
                main::types::ModelType::LlamaSeventy,
                main::types::ModelType::LlamaSeventy
            ) | (
                main::types::ModelType::LlamaSevenChat,
                main::types::ModelType::LlamaSevenChat
            ) | (
                main::types::ModelType::LlamaThirteenChat,
                main::types::ModelType::LlamaThirteenChat
            ) | (
                main::types::ModelType::LlamaSeventyChat,
                main::types::ModelType::LlamaSeventyChat
            ) | (
                main::types::ModelType::LlamaSevenCode,
                main::types::ModelType::LlamaSevenCode
            ) | (
                main::types::ModelType::LlamaThirteenCode,
                main::types::ModelType::LlamaThirteenCode
            ) | (
                main::types::ModelType::LlamaThirtyFourCode,
                main::types::ModelType::LlamaThirtyFourCode
            ) | (
                main::types::ModelType::SolarTen,
                main::types::ModelType::SolarTen
            ) | (
                main::types::ModelType::SolarTenInstruct,
                main::types::ModelType::SolarTenInstruct
            ) | (
                main::types::ModelType::PhiOne,
                main::types::ModelType::PhiOne
            ) | (
                main::types::ModelType::PhiOnePointFive,
                main::types::ModelType::PhiOnePointFive
            ) | (
                main::types::ModelType::PhiTwo,
                main::types::ModelType::PhiTwo
            ) | (
                main::types::ModelType::PuffinPhiTwo,
                main::types::ModelType::PuffinPhiTwo
            ) | (
                main::types::ModelType::DolphinPhiTwo,
                main::types::ModelType::DolphinPhiTwo
            )
        )
    }
}

impl PartialEq for ValueType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ValueType::Single(a), ValueType::Single(b)) => a == b,
            (ValueType::Many(a), ValueType::Many(b)) => a == b,
            _ => false,
        }
    }
}

impl ValueType {
    pub fn compatible(&self, other: &Self) -> bool {
        match (self, other) {
            (ValueType::Single(a), ValueType::Single(b)) => a.compatible(b),
            (ValueType::Many(a), ValueType::Many(b)) => a.compatible(b),
            (ValueType::Single(a), ValueType::Many(b)) => a.compatible(b),
            _ => false,
        }
    }

    pub fn create(&self) -> Input {
        match self {
            ValueType::Single(single) => Input::Single(single.create()),
            ValueType::Many(many) => Input::Many(vec![many.create()]),
        }
    }

    pub fn create_output(&self) -> Output {
        match self {
            ValueType::Single(single) => Output::Single(single.create()),
            ValueType::Many(many) => Output::Many(vec![many.create()]),
        }
    }
}

impl PrimitiveValueType {
    pub fn create(&self) -> PrimitiveValue {
        match self {
            PrimitiveValueType::Number => PrimitiveValue::Number(0),
            PrimitiveValueType::Float => PrimitiveValue::Float(0.),
            PrimitiveValueType::Text => PrimitiveValue::Text("".to_string()),
            PrimitiveValueType::File => PrimitiveValue::File("".to_string()),
            PrimitiveValueType::Folder => PrimitiveValue::Folder("".to_string()),
            PrimitiveValueType::Embedding => {
                PrimitiveValue::Embedding(Embedding { vector: vec![0.0] })
            }
            PrimitiveValueType::Database => PrimitiveValue::Database(Resource::new_own(0)),
            PrimitiveValueType::Model => PrimitiveValue::Model(Resource::new_own(0)),
            PrimitiveValueType::EmbeddingModel => {
                PrimitiveValue::EmbeddingModel(Resource::new_own(0))
            }
            PrimitiveValueType::ModelType => PrimitiveValue::ModelType(ModelType::LlamaSevenChat),
            PrimitiveValueType::EmbeddingModelType => {
                PrimitiveValue::EmbeddingModelType(EmbeddingModelType::Bert)
            }
            PrimitiveValueType::Boolean => PrimitiveValue::Boolean(false),
            PrimitiveValueType::Page => PrimitiveValue::Page(Resource::new_own(0)),
            PrimitiveValueType::Node => PrimitiveValue::Node(Resource::new_own(0)),
            PrimitiveValueType::Any => PrimitiveValue::Number(0),
        }
    }

    pub fn compatible(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (PrimitiveValueType::Number, PrimitiveValueType::Number)
                | (PrimitiveValueType::Float, PrimitiveValueType::Float)
                | (PrimitiveValueType::Text, PrimitiveValueType::Text)
                | (PrimitiveValueType::File, PrimitiveValueType::File)
                | (PrimitiveValueType::Folder, PrimitiveValueType::Folder)
                | (PrimitiveValueType::Embedding, PrimitiveValueType::Embedding)
                | (PrimitiveValueType::Database, PrimitiveValueType::Database)
                | (PrimitiveValueType::Model, PrimitiveValueType::Model)
                | (
                    PrimitiveValueType::EmbeddingModel,
                    PrimitiveValueType::EmbeddingModel
                )
                | (PrimitiveValueType::ModelType, PrimitiveValueType::ModelType)
                | (
                    PrimitiveValueType::EmbeddingModelType,
                    PrimitiveValueType::EmbeddingModelType
                )
                | (PrimitiveValueType::Boolean, PrimitiveValueType::Boolean)
                | (PrimitiveValueType::Page, PrimitiveValueType::Page)
                | (PrimitiveValueType::Node, PrimitiveValueType::Node)
                | (PrimitiveValueType::Any, _)
                | (_, PrimitiveValueType::Any)
        )
    }
}

#[derive(Serialize, Deserialize)]
struct MyIoDefinition {
    name: String,
    ty: MyValueType,
}

impl Serialize for IoDefinition {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let my_io_definition = MyIoDefinition {
            name: self.name.clone(),
            ty: self.ty.into(),
        };
        my_io_definition.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IoDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        let my_io_definition = MyIoDefinition::deserialize(deserializer)?;
        Ok(IoDefinition {
            name: my_io_definition.name,
            ty: my_io_definition.ty.into(),
        })
    }
}

#[derive(Serialize, Deserialize)]
enum MyValueType {
    Single(MyPrimitiveValueType),
    Many(MyPrimitiveValueType),
}

impl From<ValueType> for MyValueType {
    fn from(value: ValueType) -> Self {
        match value {
            ValueType::Single(value) => MyValueType::Single(value.into()),
            ValueType::Many(value) => MyValueType::Many(value.into()),
        }
    }
}

impl From<MyValueType> for ValueType {
    fn from(value: MyValueType) -> Self {
        match value {
            MyValueType::Single(value) => ValueType::Single(value.into()),
            MyValueType::Many(value) => ValueType::Many(value.into()),
        }
    }
}

impl Serialize for ValueType {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let my_value_type = MyValueType::from(*self);
        my_value_type.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ValueType {
    fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        let my_value_type = MyValueType::deserialize(deserializer)?;
        Ok(ValueType::from(my_value_type))
    }
}

#[derive(Serialize, Deserialize)]
enum MyPrimitiveValueType {
    Number,
    Float,
    Text,
    File,
    Folder,
    Embedding,
    Database,
    Model,
    EmbeddingModel,
    ModelType,
    EmbeddingModelType,
    Boolean,
    Page,
    Node,
    Any,
}

impl From<PrimitiveValueType> for MyPrimitiveValueType {
    fn from(value: PrimitiveValueType) -> Self {
        match value {
            PrimitiveValueType::Number => MyPrimitiveValueType::Number,
            PrimitiveValueType::Float => MyPrimitiveValueType::Float,
            PrimitiveValueType::Text => MyPrimitiveValueType::Text,
            PrimitiveValueType::File => MyPrimitiveValueType::File,
            PrimitiveValueType::Folder => MyPrimitiveValueType::Folder,
            PrimitiveValueType::Embedding => MyPrimitiveValueType::Embedding,
            PrimitiveValueType::Database => MyPrimitiveValueType::Database,
            PrimitiveValueType::Model => MyPrimitiveValueType::Model,
            PrimitiveValueType::EmbeddingModel => MyPrimitiveValueType::EmbeddingModel,
            PrimitiveValueType::ModelType => MyPrimitiveValueType::ModelType,
            PrimitiveValueType::EmbeddingModelType => MyPrimitiveValueType::EmbeddingModelType,
            PrimitiveValueType::Boolean => MyPrimitiveValueType::Boolean,
            PrimitiveValueType::Page => MyPrimitiveValueType::Page,
            PrimitiveValueType::Node => MyPrimitiveValueType::Node,
            PrimitiveValueType::Any => MyPrimitiveValueType::Any,
        }
    }
}

impl From<MyPrimitiveValueType> for PrimitiveValueType {
    fn from(value: MyPrimitiveValueType) -> Self {
        match value {
            MyPrimitiveValueType::Number => PrimitiveValueType::Number,
            MyPrimitiveValueType::Float => PrimitiveValueType::Float,
            MyPrimitiveValueType::Text => PrimitiveValueType::Text,
            MyPrimitiveValueType::File => PrimitiveValueType::File,
            MyPrimitiveValueType::Folder => PrimitiveValueType::Folder,
            MyPrimitiveValueType::Embedding => PrimitiveValueType::Embedding,
            MyPrimitiveValueType::Database => PrimitiveValueType::Database,
            MyPrimitiveValueType::Model => PrimitiveValueType::Model,
            MyPrimitiveValueType::EmbeddingModel => PrimitiveValueType::EmbeddingModel,
            MyPrimitiveValueType::ModelType => PrimitiveValueType::ModelType,
            MyPrimitiveValueType::EmbeddingModelType => PrimitiveValueType::EmbeddingModelType,
            MyPrimitiveValueType::Boolean => PrimitiveValueType::Boolean,
            MyPrimitiveValueType::Page => PrimitiveValueType::Page,
            MyPrimitiveValueType::Node => PrimitiveValueType::Node,
            MyPrimitiveValueType::Any => PrimitiveValueType::Any,
        }
    }
}

impl Serialize for PrimitiveValueType {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let my_primitive_value_type = MyPrimitiveValueType::from(*self);
        my_primitive_value_type.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PrimitiveValueType {
    fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        let my_primitive_value_type = MyPrimitiveValueType::deserialize(deserializer)?;
        Ok(PrimitiveValueType::from(my_primitive_value_type))
    }
}

impl PrimitiveValue {
    pub fn is_of_type(&self, ty: PrimitiveValueType) -> bool {
        matches!(
            (self, ty),
            (PrimitiveValue::Number(_), PrimitiveValueType::Number)
                | (PrimitiveValue::Float(_), PrimitiveValueType::Float)
                | (PrimitiveValue::Text(_), PrimitiveValueType::Text)
                | (PrimitiveValue::Embedding(_), PrimitiveValueType::Embedding)
                | (PrimitiveValue::Database(_), PrimitiveValueType::Database)
                | (PrimitiveValue::Model(_), PrimitiveValueType::Model)
                | (PrimitiveValue::ModelType(_), PrimitiveValueType::ModelType)
                | (PrimitiveValue::Boolean(_), PrimitiveValueType::Boolean)
                | (PrimitiveValue::Page(_), PrimitiveValueType::Page)
                | (PrimitiveValue::Node(_), PrimitiveValueType::Node)
        )
    }
}

impl Input {
    pub fn is_of_type(&self, ty: ValueType) -> bool {
        match (self, ty) {
            (Input::Single(value), ValueType::Single(ty)) => value.is_of_type(ty),
            (Input::Many(values), ValueType::Many(ty)) => {
                values.iter().all(|value| value.is_of_type(ty))
            }
            _ => false,
        }
    }
}

impl Output {
    pub fn is_of_type(&self, ty: ValueType) -> bool {
        match (self, ty) {
            (Output::Single(value), ValueType::Single(ty)) => value.is_of_type(ty),
            (Output::Many(values), ValueType::Many(ty)) => {
                values.iter().all(|value| value.is_of_type(ty))
            }
            _ => false,
        }
    }
}

impl Clone for Definition {
    fn clone(&self) -> Self {
        Definition {
            name: self.name.clone(),
            description: self.description.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            examples: self.examples.clone(),
        }
    }
}

impl Clone for Example {
    fn clone(&self) -> Self {
        Example {
            name: self.name.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

impl Clone for Input {
    fn clone(&self) -> Self {
        match self {
            Input::Single(value) => Input::Single(value.clone()),
            Input::Many(values) => Input::Many(values.clone()),
        }
    }
}

impl Clone for Output {
    fn clone(&self) -> Self {
        match self {
            Output::Single(value) => Output::Single(value.clone()),
            Output::Many(values) => Output::Many(values.clone()),
            Output::Halt => Output::Halt,
        }
    }
}

impl Clone for PrimitiveValue {
    fn clone(&self) -> Self {
        match self {
            PrimitiveValue::Number(value) => PrimitiveValue::Number(*value),
            PrimitiveValue::Float(value) => PrimitiveValue::Float(*value),
            PrimitiveValue::Text(value) => PrimitiveValue::Text(value.clone()),
            PrimitiveValue::File(value) => PrimitiveValue::File(value.clone()),
            PrimitiveValue::Folder(value) => PrimitiveValue::Folder(value.clone()),
            PrimitiveValue::Embedding(value) => PrimitiveValue::Embedding(value.clone()),
            PrimitiveValue::Database(value) => {
                PrimitiveValue::Database(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::Model(value) => {
                PrimitiveValue::Model(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::EmbeddingModel(value) => {
                PrimitiveValue::EmbeddingModel(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::ModelType(value) => PrimitiveValue::ModelType(*value),
            PrimitiveValue::EmbeddingModelType(value) => PrimitiveValue::EmbeddingModelType(*value),
            PrimitiveValue::Boolean(value) => PrimitiveValue::Boolean(*value),
            PrimitiveValue::Page(value) => PrimitiveValue::Page(Resource::new_borrow(value.rep())),
            PrimitiveValue::Node(value) => PrimitiveValue::Node(Resource::new_borrow(value.rep())),
        }
    }
}
