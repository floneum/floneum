use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wasmtime::component::Resource;

use crate::plugins::main::{self, types::*};
use main::types::PrimitiveValueType;

impl PartialEq for PrimitiveValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PrimitiveValue::Number(a), PrimitiveValue::Number(b)) => a == b,
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

impl PartialEq for BorrowedPrimitiveValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BorrowedPrimitiveValue::Number(a), BorrowedPrimitiveValue::Number(b)) => a == b,
            (BorrowedPrimitiveValue::Text(a), BorrowedPrimitiveValue::Text(b)) => a == b,
            (BorrowedPrimitiveValue::File(a), BorrowedPrimitiveValue::File(b)) => a == b,
            (BorrowedPrimitiveValue::Folder(a), BorrowedPrimitiveValue::Folder(b)) => a == b,
            (BorrowedPrimitiveValue::Embedding(a), BorrowedPrimitiveValue::Embedding(b)) => {
                a.vector == b.vector
            }
            (BorrowedPrimitiveValue::Database(a), BorrowedPrimitiveValue::Database(b)) => {
                a.rep() == b.rep()
            }
            (BorrowedPrimitiveValue::Model(a), BorrowedPrimitiveValue::Model(b)) => {
                a.rep() == b.rep()
            }
            (
                BorrowedPrimitiveValue::EmbeddingModel(a),
                BorrowedPrimitiveValue::EmbeddingModel(b),
            ) => a.rep() == b.rep(),
            (BorrowedPrimitiveValue::ModelType(a), BorrowedPrimitiveValue::ModelType(b)) => a == b,
            (BorrowedPrimitiveValue::Boolean(a), BorrowedPrimitiveValue::Boolean(b)) => a == b,
            (BorrowedPrimitiveValue::Page(a), BorrowedPrimitiveValue::Page(b)) => {
                a.rep() == b.rep()
            }
            (BorrowedPrimitiveValue::Node(a), BorrowedPrimitiveValue::Node(b)) => {
                a.rep() == b.rep()
            }
            _ => false,
        }
    }
}

impl Serialize for BorrowedPrimitiveValue {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let my_primitive_value = MyPrimitiveValue::from(self);
        my_primitive_value.serialize(serializer)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
enum MyPrimitiveValue {
    Number(i64),
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

impl From<&BorrowedPrimitiveValue> for MyPrimitiveValue {
    fn from(value: &BorrowedPrimitiveValue) -> Self {
        match value {
            BorrowedPrimitiveValue::Number(value) => MyPrimitiveValue::Number(*value),
            BorrowedPrimitiveValue::Text(value) => MyPrimitiveValue::Text(value.clone()),
            BorrowedPrimitiveValue::File(value) => MyPrimitiveValue::File(value.clone()),
            BorrowedPrimitiveValue::Folder(value) => MyPrimitiveValue::Folder(value.clone()),
            BorrowedPrimitiveValue::Embedding(value) => {
                MyPrimitiveValue::Embedding(value.vector.clone())
            }
            BorrowedPrimitiveValue::Model(value) => MyPrimitiveValue::Model(value.rep()),
            BorrowedPrimitiveValue::EmbeddingModel(value) => MyPrimitiveValue::Model(value.rep()),
            BorrowedPrimitiveValue::Database(value) => MyPrimitiveValue::Database(value.rep()),
            BorrowedPrimitiveValue::ModelType(value) => MyPrimitiveValue::ModelType(value.into()),
            BorrowedPrimitiveValue::EmbeddingModelType(value) => {
                MyPrimitiveValue::EmbeddingModelType(value.into())
            }
            BorrowedPrimitiveValue::Boolean(value) => MyPrimitiveValue::Boolean(*value),
            BorrowedPrimitiveValue::Page(value) => MyPrimitiveValue::Page(value.rep()),
            BorrowedPrimitiveValue::Node(value) => MyPrimitiveValue::Node(value.rep()),
        }
    }
}

impl From<MyPrimitiveValue> for BorrowedPrimitiveValue {
    fn from(value: MyPrimitiveValue) -> Self {
        match value {
            MyPrimitiveValue::Number(value) => BorrowedPrimitiveValue::Number(value),
            MyPrimitiveValue::Text(value) => BorrowedPrimitiveValue::Text(value),
            MyPrimitiveValue::File(value) => BorrowedPrimitiveValue::File(value),
            MyPrimitiveValue::Folder(value) => BorrowedPrimitiveValue::Folder(value),
            MyPrimitiveValue::Embedding(value) => {
                BorrowedPrimitiveValue::Embedding(Embedding { vector: value })
            }
            MyPrimitiveValue::Model(value) => {
                BorrowedPrimitiveValue::Model(Resource::new_own(value))
            }
            MyPrimitiveValue::EmbeddingModel(value) => {
                BorrowedPrimitiveValue::EmbeddingModel(Resource::new_own(value))
            }
            MyPrimitiveValue::ModelType(value) => BorrowedPrimitiveValue::ModelType(value.into()),
            MyPrimitiveValue::EmbeddingModelType(value) => {
                BorrowedPrimitiveValue::EmbeddingModelType(value.into())
            }
            MyPrimitiveValue::Database(value) => {
                BorrowedPrimitiveValue::Database(Resource::new_own(value))
            }
            MyPrimitiveValue::Boolean(value) => BorrowedPrimitiveValue::Boolean(value),
            MyPrimitiveValue::Page(value) => BorrowedPrimitiveValue::Page(Resource::new_own(value)),
            MyPrimitiveValue::Node(value) => BorrowedPrimitiveValue::Node(Resource::new_own(value)),
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
}

impl PrimitiveValueType {
    pub fn create(&self) -> PrimitiveValue {
        match self {
            PrimitiveValueType::Number => PrimitiveValue::Number(0),
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

    pub fn borrow(&self) -> BorrowedPrimitiveValue {
        match self {
            PrimitiveValue::Number(value) => BorrowedPrimitiveValue::Number(*value),
            PrimitiveValue::Text(value) => BorrowedPrimitiveValue::Text(value.clone()),
            PrimitiveValue::File(value) => BorrowedPrimitiveValue::File(value.clone()),
            PrimitiveValue::Folder(value) => BorrowedPrimitiveValue::Folder(value.clone()),
            PrimitiveValue::Embedding(value) => BorrowedPrimitiveValue::Embedding(value.clone()),
            PrimitiveValue::Database(value) => {
                BorrowedPrimitiveValue::Database(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::Model(value) => {
                BorrowedPrimitiveValue::Model(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::EmbeddingModel(value) => {
                BorrowedPrimitiveValue::EmbeddingModel(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::ModelType(value) => BorrowedPrimitiveValue::ModelType(*value),
            PrimitiveValue::EmbeddingModelType(value) => {
                BorrowedPrimitiveValue::EmbeddingModelType(*value)
            }
            PrimitiveValue::Boolean(value) => BorrowedPrimitiveValue::Boolean(*value),
            PrimitiveValue::Page(value) => {
                BorrowedPrimitiveValue::Page(Resource::new_borrow(value.rep()))
            }
            PrimitiveValue::Node(value) => {
                BorrowedPrimitiveValue::Node(Resource::new_borrow(value.rep()))
            }
        }
    }
}

impl BorrowedPrimitiveValue {
    pub fn is_of_type(&self, ty: PrimitiveValueType) -> bool {
        matches!(
            (self, ty),
            (
                BorrowedPrimitiveValue::Number(_),
                PrimitiveValueType::Number
            ) | (BorrowedPrimitiveValue::Text(_), PrimitiveValueType::Text)
                | (
                    BorrowedPrimitiveValue::Embedding(_),
                    PrimitiveValueType::Embedding
                )
                | (
                    BorrowedPrimitiveValue::Database(_),
                    PrimitiveValueType::Database
                )
                | (BorrowedPrimitiveValue::Model(_), PrimitiveValueType::Model)
                | (
                    BorrowedPrimitiveValue::ModelType(_),
                    PrimitiveValueType::ModelType
                )
                | (
                    BorrowedPrimitiveValue::Boolean(_),
                    PrimitiveValueType::Boolean
                )
                | (BorrowedPrimitiveValue::Page(_), PrimitiveValueType::Page)
                | (BorrowedPrimitiveValue::Node(_), PrimitiveValueType::Node)
        )
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

impl Clone for PrimitiveValue {
    fn clone(&self) -> Self {
        match self {
            PrimitiveValue::Number(value) => PrimitiveValue::Number(*value),
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

impl Clone for BorrowedPrimitiveValue {
    fn clone(&self) -> Self {
        match self {
            BorrowedPrimitiveValue::Number(value) => BorrowedPrimitiveValue::Number(*value),
            BorrowedPrimitiveValue::Text(value) => BorrowedPrimitiveValue::Text(value.clone()),
            BorrowedPrimitiveValue::File(value) => BorrowedPrimitiveValue::File(value.clone()),
            BorrowedPrimitiveValue::Folder(value) => BorrowedPrimitiveValue::Folder(value.clone()),
            BorrowedPrimitiveValue::Embedding(value) => {
                BorrowedPrimitiveValue::Embedding(value.clone())
            }
            BorrowedPrimitiveValue::Database(value) => {
                BorrowedPrimitiveValue::Database(Resource::new_borrow(value.rep()))
            }
            BorrowedPrimitiveValue::Model(value) => {
                BorrowedPrimitiveValue::Model(Resource::new_borrow(value.rep()))
            }
            BorrowedPrimitiveValue::EmbeddingModel(value) => {
                BorrowedPrimitiveValue::EmbeddingModel(Resource::new_borrow(value.rep()))
            }
            BorrowedPrimitiveValue::ModelType(value) => BorrowedPrimitiveValue::ModelType(*value),
            BorrowedPrimitiveValue::EmbeddingModelType(value) => {
                BorrowedPrimitiveValue::EmbeddingModelType(*value)
            }
            BorrowedPrimitiveValue::Boolean(value) => BorrowedPrimitiveValue::Boolean(*value),
            BorrowedPrimitiveValue::Page(value) => {
                BorrowedPrimitiveValue::Page(Resource::new_borrow(value.rep()))
            }
            BorrowedPrimitiveValue::Node(value) => {
                BorrowedPrimitiveValue::Node(Resource::new_borrow(value.rep()))
            }
        }
    }
}
