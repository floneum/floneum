use serde::{Deserialize, Serialize, Serializer};

use crate::exports::plugins::main::definitions::{
    Input, Output, PrimitiveValue, PrimitiveValueType, ValueType,
};
use crate::plugins::main::types::{
    Embedding, EmbeddingDbId, GptNeoXType, LlamaType, ModelType, MptType,
};
use crate::plugins::main::{
    imports::TabId,
    types::{ModelId, NodeId},
};

#[derive(serde::Serialize, serde::Deserialize)]
enum MyValueType {
    Single(MyPrimitiveValue),
    List(Vec<MyPrimitiveValue>),
    Unset,
}

impl From<Input> for MyValueType {
    fn from(value: Input) -> Self {
        match value {
            Input::Single(value) => MyValueType::Single(value.into()),
            Input::Many(values) => {
                MyValueType::List(values.into_iter().map(|v| v.into()).collect())
            }
        }
    }
}

impl From<MyValueType> for Input {
    fn from(value: MyValueType) -> Self {
        match value {
            MyValueType::Single(value) => Input::Single(value.into()),
            MyValueType::List(values) => {
                Input::Many(values.into_iter().map(|v| v.into()).collect())
            }
            MyValueType::Unset => Input::Many(Vec::new()),
        }
    }
}

impl Serialize for Input {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MyValueType::from(self.clone()).serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for Input {
    fn deserialize<D: serde::Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        MyValueType::deserialize(deserializer).map(|v| v.into())
    }
}

impl From<Output> for MyValueType {
    fn from(output: Output) -> Self {
        match output {
            Output::Single(value) => MyValueType::Single(value.into()),
            Output::Many(values) => {
                MyValueType::List(values.into_iter().map(|v| v.into()).collect())
            }
            Output::Halt => MyValueType::Unset,
        }
    }
}

impl From<MyValueType> for Output {
    fn from(value: MyValueType) -> Self {
        match value {
            MyValueType::Single(value) => Output::Single(value.into()),
            MyValueType::List(values) => {
                Output::Many(values.into_iter().map(|v| v.into()).collect())
            }
            MyValueType::Unset => Output::Halt,
        }
    }
}

impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MyValueType::from(self.clone()).serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for Output {
    fn deserialize<D: serde::Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        MyValueType::deserialize(deserializer).map(|v| v.into())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
enum MyPrimitiveValue {
    Number(i64),
    Text(String),
    Embedding(Vec<f32>),
    Model(u32),
    Database(u32),
    ModelType(MyModelType),
    Boolean(bool),
    Tab(u32),
    Node { id: u32, tab_id: u32 },
}

impl From<PrimitiveValue> for MyPrimitiveValue {
    fn from(value: PrimitiveValue) -> Self {
        match value {
            PrimitiveValue::Number(value) => MyPrimitiveValue::Number(value),
            PrimitiveValue::Text(value) => MyPrimitiveValue::Text(value),
            PrimitiveValue::Embedding(value) => MyPrimitiveValue::Embedding(value.vector),
            PrimitiveValue::Model(value) => MyPrimitiveValue::Model(value.id),
            PrimitiveValue::Database(value) => MyPrimitiveValue::Database(value.id),
            PrimitiveValue::ModelType(value) => MyPrimitiveValue::ModelType(value.into()),
            PrimitiveValue::Boolean(value) => MyPrimitiveValue::Boolean(value),
            PrimitiveValue::Tab(value) => MyPrimitiveValue::Tab(value.id),
            PrimitiveValue::Node(value) => MyPrimitiveValue::Node {
                id: value.id,
                tab_id: value.tab.id,
            },
        }
    }
}

impl From<MyPrimitiveValue> for PrimitiveValue {
    fn from(value: MyPrimitiveValue) -> Self {
        match value {
            MyPrimitiveValue::Number(value) => PrimitiveValue::Number(value),
            MyPrimitiveValue::Text(value) => PrimitiveValue::Text(value),
            MyPrimitiveValue::Embedding(value) => {
                PrimitiveValue::Embedding(Embedding { vector: value })
            }
            MyPrimitiveValue::Model(value) => PrimitiveValue::Model(ModelId { id: value }),
            MyPrimitiveValue::Database(value) => {
                PrimitiveValue::Database(EmbeddingDbId { id: value })
            }
            MyPrimitiveValue::ModelType(value) => PrimitiveValue::ModelType(value.into()),
            MyPrimitiveValue::Boolean(value) => PrimitiveValue::Boolean(value),
            MyPrimitiveValue::Tab(value) => PrimitiveValue::Tab(TabId { id: value }),
            MyPrimitiveValue::Node { id, tab_id } => PrimitiveValue::Node(NodeId {
                id,
                tab: TabId { id: tab_id },
            }),
        }
    }
}

#[derive(Serialize, Deserialize)]
enum MyModelType {
    Mpt(MyMptType),
    GptNeoX(MyGptNeoXType),
    Llama(MyLlamaType),
}

impl From<ModelType> for MyModelType {
    fn from(value: ModelType) -> Self {
        match value {
            ModelType::Mpt(value) => MyModelType::Mpt(value.into()),
            ModelType::GptNeoX(value) => MyModelType::GptNeoX(value.into()),
            ModelType::Llama(value) => MyModelType::Llama(value.into()),
        }
    }
}

impl From<MyModelType> for ModelType {
    fn from(value: MyModelType) -> Self {
        match value {
            MyModelType::Mpt(value) => ModelType::Mpt(value.into()),
            MyModelType::GptNeoX(value) => ModelType::GptNeoX(value.into()),
            MyModelType::Llama(value) => ModelType::Llama(value.into()),
        }
    }
}

#[derive(Serialize, Deserialize)]
enum MyMptType {
    Base,
    Story,
    Instruct,
    Chat,
}

impl From<MptType> for MyMptType {
    fn from(value: MptType) -> Self {
        match value {
            MptType::Base => MyMptType::Base,
            MptType::Story => MyMptType::Story,
            MptType::Instruct => MyMptType::Instruct,
            MptType::Chat => MyMptType::Chat,
        }
    }
}

impl From<MyMptType> for MptType {
    fn from(value: MyMptType) -> Self {
        match value {
            MyMptType::Base => MptType::Base,
            MyMptType::Story => MptType::Story,
            MyMptType::Instruct => MptType::Instruct,
            MyMptType::Chat => MptType::Chat,
        }
    }
}

#[derive(Serialize, Deserialize)]
enum MyGptNeoXType {
    LargePythia,
    TinyPythia,
    DollySevenB,
    Stablelm,
}

impl From<GptNeoXType> for MyGptNeoXType {
    fn from(value: GptNeoXType) -> Self {
        match value {
            GptNeoXType::LargePythia => MyGptNeoXType::LargePythia,
            GptNeoXType::TinyPythia => MyGptNeoXType::TinyPythia,
            GptNeoXType::DollySevenB => MyGptNeoXType::DollySevenB,
            GptNeoXType::Stablelm => MyGptNeoXType::Stablelm,
        }
    }
}

impl From<MyGptNeoXType> for GptNeoXType {
    fn from(value: MyGptNeoXType) -> Self {
        match value {
            MyGptNeoXType::LargePythia => GptNeoXType::LargePythia,
            MyGptNeoXType::TinyPythia => GptNeoXType::TinyPythia,
            MyGptNeoXType::DollySevenB => GptNeoXType::DollySevenB,
            MyGptNeoXType::Stablelm => GptNeoXType::Stablelm,
        }
    }
}

#[derive(Serialize, Deserialize)]
enum MyLlamaType {
    Vicuna,
    Guanaco,
    Wizardlm,
    Orca,
    LlamaSevenChat,
    LlamaThirteenChat,
}

impl From<LlamaType> for MyLlamaType {
    fn from(value: LlamaType) -> Self {
        match value {
            LlamaType::Vicuna => MyLlamaType::Vicuna,
            LlamaType::Guanaco => MyLlamaType::Guanaco,
            LlamaType::Wizardlm => MyLlamaType::Wizardlm,
            LlamaType::Orca => MyLlamaType::Orca,
            LlamaType::LlamaSevenChat => MyLlamaType::LlamaSevenChat,
            LlamaType::LlamaThirteenChat => MyLlamaType::LlamaThirteenChat,
        }
    }
}

impl From<MyLlamaType> for LlamaType {
    fn from(value: MyLlamaType) -> Self {
        match value {
            MyLlamaType::Vicuna => LlamaType::Vicuna,
            MyLlamaType::Guanaco => LlamaType::Guanaco,
            MyLlamaType::Wizardlm => LlamaType::Wizardlm,
            MyLlamaType::Orca => LlamaType::Orca,
            MyLlamaType::LlamaSevenChat => LlamaType::LlamaSevenChat,
            MyLlamaType::LlamaThirteenChat => LlamaType::LlamaThirteenChat,
        }
    }
}

impl ValueType {
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
            PrimitiveValueType::Text => PrimitiveValue::Text("".to_string()),
            PrimitiveValueType::Embedding => {
                PrimitiveValue::Embedding(Embedding { vector: vec![0.0] })
            }
            PrimitiveValueType::Database => PrimitiveValue::Database(EmbeddingDbId { id: 0 }),
            PrimitiveValueType::Model => PrimitiveValue::Model(ModelId { id: 0 }),
            PrimitiveValueType::ModelType => {
                PrimitiveValue::ModelType(ModelType::Llama(LlamaType::LlamaSevenChat))
            }
            PrimitiveValueType::Boolean => PrimitiveValue::Boolean(false),
            PrimitiveValueType::Tab => PrimitiveValue::Tab(TabId { id: 0 }),
            PrimitiveValueType::Node => PrimitiveValue::Node(NodeId {
                id: 0,
                tab: TabId { id: 0 },
            }),
            PrimitiveValueType::Any => PrimitiveValue::Number(0),
        }
    }
}
