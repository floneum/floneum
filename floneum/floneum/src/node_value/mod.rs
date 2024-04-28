mod show;
pub use show::*;
mod modify;
pub use modify::*;
mod structure;
use floneum_plugin::plugins::main::types::{
    EmbeddingModelType, ModelType, PrimitiveValueType, ValueType,
};
pub use structure::*;

pub trait Variants: Sized + 'static {
    const VARIANTS: &'static [Self];
}

impl Variants for ModelType {
    const VARIANTS: &'static [Self] = &[
        ModelType::MistralSeven,
        ModelType::MistralSevenInstruct,
        ModelType::MistralSevenInstructTwo,
        ModelType::ZephyrSevenAlpha,
        ModelType::ZephyrSevenBeta,
        ModelType::OpenChatSeven,
        ModelType::StarlingSevenAlpha,
        ModelType::TinyLlamaChat,
        ModelType::TinyLlama,
        ModelType::LlamaSeven,
        ModelType::LlamaThirteen,
        ModelType::LlamaSeventy,
        ModelType::LlamaSevenChat,
        ModelType::LlamaThirteenChat,
        ModelType::LlamaSeventyChat,
        ModelType::LlamaSevenCode,
        ModelType::LlamaThirteenCode,
        ModelType::LlamaThirtyFourCode,
        ModelType::SolarTen,
        ModelType::SolarTenInstruct,
        ModelType::PhiOne,
        ModelType::PhiOnePointFive,
        ModelType::PhiTwo,
        ModelType::PuffinPhiTwo,
        ModelType::DolphinPhiTwo,
    ];
}

impl Variants for EmbeddingModelType {
    const VARIANTS: &'static [Self] = &[EmbeddingModelType::Bert];
}

impl Variants for PrimitiveValueType {
    const VARIANTS: &'static [Self] = &[
        PrimitiveValueType::Text,
        PrimitiveValueType::Float,
        PrimitiveValueType::File,
        PrimitiveValueType::Folder,
        PrimitiveValueType::Number,
        PrimitiveValueType::Boolean,
        PrimitiveValueType::Embedding,
        PrimitiveValueType::Model,
        PrimitiveValueType::ModelType,
        PrimitiveValueType::Database,
        PrimitiveValueType::Page,
        PrimitiveValueType::Node,
        PrimitiveValueType::Any,
    ];
}

impl Variants for ValueType {
    const VARIANTS: &'static [Self] = &[
        ValueType::Single(PrimitiveValueType::Text),
        ValueType::Single(PrimitiveValueType::File),
        ValueType::Single(PrimitiveValueType::Folder),
        ValueType::Single(PrimitiveValueType::Number),
        ValueType::Single(PrimitiveValueType::Boolean),
        ValueType::Single(PrimitiveValueType::Embedding),
        ValueType::Single(PrimitiveValueType::Model),
        ValueType::Single(PrimitiveValueType::ModelType),
        ValueType::Single(PrimitiveValueType::Database),
        ValueType::Single(PrimitiveValueType::Page),
        ValueType::Single(PrimitiveValueType::Node),
        ValueType::Single(PrimitiveValueType::Any),
        ValueType::Many(PrimitiveValueType::Text),
        ValueType::Many(PrimitiveValueType::File),
        ValueType::Many(PrimitiveValueType::Folder),
        ValueType::Many(PrimitiveValueType::Number),
        ValueType::Many(PrimitiveValueType::Boolean),
        ValueType::Many(PrimitiveValueType::Embedding),
        ValueType::Many(PrimitiveValueType::Model),
        ValueType::Many(PrimitiveValueType::ModelType),
        ValueType::Many(PrimitiveValueType::Database),
        ValueType::Many(PrimitiveValueType::Page),
        ValueType::Many(PrimitiveValueType::Node),
        ValueType::Many(PrimitiveValueType::Any),
    ];
}

pub trait Named {
    fn name(&self) -> &'static str;
}

impl Named for ModelType {
    fn name(&self) -> &'static str {
        match self {
            ModelType::MistralSeven => "Mistral Seven",
            ModelType::MistralSevenInstruct => "Mistral Seven Instruct",
            ModelType::MistralSevenInstructTwo => "Mistral Seven Instruct Two",
            ModelType::ZephyrSevenAlpha => "Zephyr Seven Alpha",
            ModelType::ZephyrSevenBeta => "Zephyr Seven Beta",
            ModelType::OpenChatSeven => "Open Chat Seven",
            ModelType::StarlingSevenAlpha => "Starling Seven Alpha",
            ModelType::TinyLlamaChat => "Tiny Llama Chat",
            ModelType::TinyLlama => "Tiny Llama",
            ModelType::LlamaSeven => "Llama Seven",
            ModelType::LlamaThirteen => "Llama Thirteen",
            ModelType::LlamaSeventy => "Llama Seventy",
            ModelType::LlamaSevenChat => "Llama Seven Chat",
            ModelType::LlamaThirteenChat => "Llama Thirteen Chat",
            ModelType::LlamaSeventyChat => "Llama Seventy Chat",
            ModelType::LlamaSevenCode => "Llama Seven Code",
            ModelType::LlamaThirteenCode => "Llama Thirteen Code",
            ModelType::LlamaThirtyFourCode => "Llama Thirty Four Code",
            ModelType::SolarTen => "Solar Ten",
            ModelType::SolarTenInstruct => "Solar Ten Instruct",
            ModelType::PhiOne => "Phi One",
            ModelType::PhiOnePointFive => "Phi One Point Five",
            ModelType::PhiTwo => "Phi Two",
            ModelType::PuffinPhiTwo => "Puffin Phi Two",
            ModelType::DolphinPhiTwo => "Dolphin Phi Two",
        }
    }
}

impl Named for EmbeddingModelType {
    fn name(&self) -> &'static str {
        match self {
            EmbeddingModelType::Bert => "Bert",
        }
    }
}

fn model_type_from_str(s: &str) -> Option<ModelType> {
    match &*s.to_lowercase() {
        "mistral seven" => Some(ModelType::MistralSeven),
        "mistral seven instruct" => Some(ModelType::MistralSevenInstruct),
        "mistral seven instruct two" => Some(ModelType::MistralSevenInstructTwo),
        "zephyr seven alpha" => Some(ModelType::ZephyrSevenAlpha),
        "zephyr seven beta" => Some(ModelType::ZephyrSevenBeta),
        "open chat seven" => Some(ModelType::OpenChatSeven),
        "starling seven alpha" => Some(ModelType::StarlingSevenAlpha),
        "tiny llama chat" => Some(ModelType::TinyLlamaChat),
        "tiny llama" => Some(ModelType::TinyLlama),
        "llama seven" => Some(ModelType::LlamaSeven),
        "llama thirteen" => Some(ModelType::LlamaThirteen),
        "llama seventy" => Some(ModelType::LlamaSeventy),
        "llama seven chat" => Some(ModelType::LlamaSevenChat),
        "llama thirteen chat" => Some(ModelType::LlamaThirteenChat),
        "llama seventy chat" => Some(ModelType::LlamaSeventyChat),
        "llama seven code" => Some(ModelType::LlamaSevenCode),
        "llama thirteen code" => Some(ModelType::LlamaThirteenCode),
        "llama thirty four code" => Some(ModelType::LlamaThirtyFourCode),
        "solar ten" => Some(ModelType::SolarTen),
        "solar ten instruct" => Some(ModelType::SolarTenInstruct),
        "phi one" => Some(ModelType::PhiOne),
        "phi one point five" => Some(ModelType::PhiOnePointFive),
        "phi two" => Some(ModelType::PhiTwo),
        "puffin phi two" => Some(ModelType::PuffinPhiTwo),
        "dolphin phi two" => Some(ModelType::DolphinPhiTwo),
        _ => None,
    }
}

fn embedding_model_type_from_str(s: &str) -> Option<EmbeddingModelType> {
    match &*s.to_lowercase() {
        "bert" => Some(EmbeddingModelType::Bert),
        _ => None,
    }
}

pub trait Colored {
    fn color(&self) -> String;
}

impl Colored for ValueType {
    fn color(&self) -> String {
        match self {
            ValueType::Single(ty) => ty.color(),
            ValueType::Many(ty) => ty.color(),
        }
    }
}

impl Colored for PrimitiveValueType {
    fn color(&self) -> String {
        let index = Self::VARIANTS.iter().position(|v| v == self).unwrap();
        let hue_index = index;
        let saturation_scale = (index % 4) as f32 / 4.;
        let brightness_scale = (index % 8) as f32 / 8.;
        let hue = 360. * hue_index as f32 / Self::VARIANTS.len() as f32;
        let min_saturation = 50.;
        let max_saturation = 100.;
        let saturation = min_saturation + saturation_scale * (max_saturation - min_saturation);
        let min_brightness = 30.;
        let max_brightness = 60.;
        let brightness = min_brightness + brightness_scale * (max_brightness - min_brightness);
        format!("hsl({hue}, {saturation}%, {brightness}%)")
    }
}
