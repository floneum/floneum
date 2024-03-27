use crate::host::State;
use crate::plugins::main;
use crate::plugins::main::types::TextGenerationModel;

use kalosm::language::*;
use kalosm_common::ModelLoadingProgress;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

pub(crate) enum LazyTextGenerationModel {
    Uninitialized(main::types::ModelType),
    Initialized(ConcreteTextGenerationModel),
}

pub(crate) enum ConcreteTextGenerationModel {
    Llama(Llama),
    Phi(Phi),
}

impl LazyTextGenerationModel {
    async fn value(&mut self) -> anyhow::Result<&mut ConcreteTextGenerationModel> {
        if let LazyTextGenerationModel::Uninitialized(ty) = self {
            let model_type_as_id = *ty as usize;
            let progress = move |progress: ModelLoadingProgress| {
                if let Some(callbacks) = MODEL_DOWNLOAD_PROGRESS
                    .write()
                    .unwrap()
                    .get_mut(&model_type_as_id)
                {
                    for callback in callbacks {
                        callback(progress.clone());
                    }
                }
            };
            let builder = ty.llm_builder();
            match builder {
                LlmBuilder::Llama(builder) => {
                    let model = builder.build_with_loading_handler(progress).await?;
                    *self = LazyTextGenerationModel::Initialized(
                        ConcreteTextGenerationModel::Llama(model),
                    );
                }
                LlmBuilder::Phi(builder) => {
                    let model = builder.build_with_loading_handler(progress).await?;
                    *self = LazyTextGenerationModel::Initialized(ConcreteTextGenerationModel::Phi(
                        model,
                    ));
                }
            }
        }
        match self {
            LazyTextGenerationModel::Initialized(model) => Ok(model),
            _ => unreachable!(),
        }
    }
}

#[allow(clippy::type_complexity)]
static MODEL_DOWNLOAD_PROGRESS: Lazy<
    RwLock<HashMap<usize, Vec<Box<dyn FnMut(ModelLoadingProgress) + Send + Sync>>>>,
> = Lazy::new(Default::default);

pub fn listen_to_model_download_progresses<
    F: FnMut(ModelLoadingProgress) + Send + Sync + 'static,
>(
    model_type: main::types::ModelType,
    f: F,
) {
    let mut progress = MODEL_DOWNLOAD_PROGRESS.write().unwrap();
    let model_type_as_id = model_type as usize;
    progress
        .entry(model_type_as_id)
        .or_default()
        .push(Box::new(f));
}

enum LlmBuilder {
    Llama(LlamaBuilder),
    Phi(PhiBuilder),
}

impl From<PhiBuilder> for LlmBuilder {
    fn from(builder: PhiBuilder) -> Self {
        LlmBuilder::Phi(builder)
    }
}

impl From<LlamaBuilder> for LlmBuilder {
    fn from(builder: LlamaBuilder) -> Self {
        LlmBuilder::Llama(builder)
    }
}

impl LlmBuilder {
    fn requires_download(&self) -> bool {
        match self {
            LlmBuilder::Llama(builder) => builder.requires_download(),
            LlmBuilder::Phi(builder) => builder.requires_download(),
        }
    }
}

impl main::types::ModelType {
    fn llm_builder(&self) -> LlmBuilder {
        match self {
            main::types::ModelType::MistralSeven => Llama::builder()
                .with_source(LlamaSource::mistral_7b())
                .into(),
            main::types::ModelType::MistralSevenInstruct => Llama::builder()
                .with_source(LlamaSource::mistral_7b_instruct())
                .into(),
            main::types::ModelType::MistralSevenInstructTwo => Llama::builder()
                .with_source(LlamaSource::mistral_7b_instruct_2())
                .into(),
            main::types::ModelType::ZephyrSevenAlpha => Llama::builder()
                .with_source(LlamaSource::zephyr_7b_alpha())
                .into(),
            main::types::ModelType::ZephyrSevenBeta => Llama::builder()
                .with_source(LlamaSource::zephyr_7b_beta())
                .into(),
            main::types::ModelType::OpenChatSeven => Llama::builder()
                .with_source(LlamaSource::open_chat_7b())
                .into(),
            main::types::ModelType::StarlingSevenAlpha => Llama::builder()
                .with_source(LlamaSource::starling_7b_alpha())
                .into(),
            main::types::ModelType::TinyLlamaChat => Llama::builder()
                .with_source(LlamaSource::tiny_llama_1_1b_chat())
                .into(),
            main::types::ModelType::TinyLlama => Llama::builder()
                .with_source(LlamaSource::tiny_llama_1_1b())
                .into(),
            main::types::ModelType::LlamaSeven => {
                Llama::builder().with_source(LlamaSource::llama_7b()).into()
            }
            main::types::ModelType::LlamaThirteen => Llama::builder()
                .with_source(LlamaSource::llama_13b())
                .into(),
            main::types::ModelType::LlamaSeventy => Llama::builder()
                .with_source(LlamaSource::llama_70b())
                .into(),
            main::types::ModelType::LlamaSevenChat => Llama::builder()
                .with_source(LlamaSource::llama_7b_chat())
                .into(),
            main::types::ModelType::LlamaThirteenChat => Llama::builder()
                .with_source(LlamaSource::llama_13b_chat())
                .into(),
            main::types::ModelType::LlamaSeventyChat => Llama::builder()
                .with_source(LlamaSource::llama_70b_chat())
                .into(),
            main::types::ModelType::LlamaSevenCode => Llama::builder()
                .with_source(LlamaSource::llama_7b_code())
                .into(),
            main::types::ModelType::LlamaThirteenCode => Llama::builder()
                .with_source(LlamaSource::llama_13b_code())
                .into(),
            main::types::ModelType::LlamaThirtyFourCode => Llama::builder()
                .with_source(LlamaSource::llama_34b_code())
                .into(),
            main::types::ModelType::SolarTen => Llama::builder()
                .with_source(LlamaSource::solar_10_7b())
                .into(),
            main::types::ModelType::SolarTenInstruct => Llama::builder()
                .with_source(LlamaSource::solar_10_7b_instruct())
                .into(),
            main::types::ModelType::PhiOne => Phi::builder().with_source(PhiSource::v1()).into(),
            main::types::ModelType::PhiOnePointFive => {
                Phi::builder().with_source(PhiSource::v1_5()).into()
            }
            main::types::ModelType::PhiTwo => Phi::builder().with_source(PhiSource::v2()).into(),
            main::types::ModelType::PuffinPhiTwo => Phi::builder()
                .with_source(PhiSource::puffin_phi_v2())
                .into(),
            main::types::ModelType::DolphinPhiTwo => Phi::builder()
                .with_source(PhiSource::dolphin_phi_v2())
                .into(),
        }
    }
}

impl main::types::ModelType {
    /// Returns whether the model has been downloaded.
    pub fn model_downloaded_sync(&self) -> bool {
        !self.llm_builder().requires_download()
    }
}

impl State {
    pub(crate) fn impl_create_text_generation_model(
        &mut self,
        ty: main::types::ModelType,
    ) -> TextGenerationModel {
        let model = LazyTextGenerationModel::Uninitialized(ty);
        let idx = self.models.insert(model);

        TextGenerationModel {
            id: idx as u64,
            owned: true,
        }
    }

    pub(crate) async fn impl_text_generation_model_downloaded(
        &mut self,
        ty: main::types::ModelType,
    ) -> wasmtime::Result<bool> {
        Ok(ty.model_downloaded_sync())
    }

    pub(crate) async fn impl_infer(
        &mut self,
        self_: TextGenerationModel,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> wasmtime::Result<String> {
        let model = self.models[self_.id as usize].value().await?;
        match model {
            ConcreteTextGenerationModel::Llama(model) => Ok(model
                .generate_text(&input)
                .with_max_length(max_tokens.unwrap_or(u32::MAX))
                .with_stop_on(stop_on)
                .await?),
            ConcreteTextGenerationModel::Phi(model) => Ok(model
                .generate_text(&input)
                .with_max_length(max_tokens.unwrap_or(u32::MAX))
                .with_stop_on(stop_on)
                .await?),
        }
    }

    pub(crate) async fn impl_infer_structured(
        &mut self,
        self_: TextGenerationModel,
        input: String,
        regex: String,
    ) -> wasmtime::Result<String> {
        let structure = RegexParser::new(&regex)?;

        let model = self.models[self_.id as usize].value().await?;
        match model {
            ConcreteTextGenerationModel::Llama(model) => Ok(model
                .stream_structured_text(&input, structure)
                .await?
                .text()
                .await),
            ConcreteTextGenerationModel::Phi(model) => Ok(model
                .stream_structured_text(&input, structure)
                .await?
                .text()
                .await),
        }
    }

    pub(crate) fn impl_drop_text_generation_model(
        &mut self,
        model: TextGenerationModel,
    ) -> wasmtime::Result<()> {
        self.models.remove(model.id as usize);
        Ok(())
    }
}
