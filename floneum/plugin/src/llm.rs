use crate::host::State;
use crate::plugins::main;
use crate::plugins::main::types::{Embedding, EmbeddingModel, Model};

use kalosm::language::*;
use kalosm_common::ModelLoadingProgress;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;
use wasmtime::component::__internal::async_trait;

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

#[allow(clippy::type_complexity)]
static EMBEDDING_MODEL_DOWNLOAD_PROGRESS: Lazy<
    RwLock<HashMap<usize, Vec<Box<dyn FnMut(ModelLoadingProgress) + Send + Sync>>>>,
> = Lazy::new(Default::default);

pub fn listen_to_embedding_model_download_progresses<
    F: FnMut(ModelLoadingProgress) + Send + Sync + 'static,
>(
    model_type: main::types::EmbeddingModelType,
    f: F,
) {
    let mut progress = EMBEDDING_MODEL_DOWNLOAD_PROGRESS.write().unwrap();
    let model_type_as_id = model_type as usize;
    progress
        .entry(model_type_as_id)
        .or_default()
        .push(Box::new(f));
}

impl main::types::EmbeddingModelType {
    /// Returns whether the model has been downloaded.
    pub fn model_downloaded_sync(&self) -> bool {
        !Bert::builder().requires_download()
    }
}

#[async_trait]
impl main::types::HostEmbeddingModel for State {
    async fn new(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingModel>> {
        let model = match ty {
            main::types::EmbeddingModelType::Bert => Bert::builder()
                .build_with_loading_handler(move |progress: ModelLoadingProgress| {
                    if let Some(callbacks) = EMBEDDING_MODEL_DOWNLOAD_PROGRESS
                        .write()
                        .unwrap()
                        .get_mut(&(ty as usize))
                    {
                        for callback in callbacks {
                            callback(progress.clone());
                        }
                    }
                })
                .await?
                .into_any_embedder(),
        };
        let idx = self.embedders.insert(model);

        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn model_downloaded(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<bool> {
        Ok(ty.model_downloaded_sync())
    }

    async fn get_embedding(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingModel>,
        document: String,
    ) -> wasmtime::Result<Embedding> {
        Ok(main::types::Embedding {
            vector: self.embedders[self_.rep() as usize]
                .embed(&document)
                .await?
                .to_vec(),
        })
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<EmbeddingModel>) -> wasmtime::Result<()> {
        self.embedders.remove(rep.rep() as usize);
        Ok(())
    }
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

#[async_trait]
impl main::types::HostModel for State {
    async fn new(
        &mut self,
        ty: main::types::ModelType,
    ) -> wasmtime::Result<wasmtime::component::Resource<Model>> {
        let model_type_as_id = ty as usize;
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
        let model = match ty.llm_builder() {
            LlmBuilder::Llama(builder) => builder
                .build_with_loading_handler(progress)
                .await?
                .into_any_model(),
            LlmBuilder::Phi(builder) => builder
                .build_with_loading_handler(progress)
                .await?
                .into_any_model(),
        };
        let idx = self.models.insert(model);

        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn model_downloaded(&mut self, ty: main::types::ModelType) -> wasmtime::Result<bool> {
        Ok(ty.model_downloaded_sync())
    }

    async fn infer(
        &mut self,
        self_: wasmtime::component::Resource<Model>,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> wasmtime::Result<String> {
        Ok(self.models[self_.rep() as usize]
            .generate_text(&input)
            .with_max_length(max_tokens.unwrap_or(u32::MAX))
            .with_stop_on(stop_on)
            .await?)
    }

    async fn infer_structured(
        &mut self,
        self_: wasmtime::component::Resource<Model>,
        input: String,
        regex: String,
    ) -> wasmtime::Result<String> {
        let structure = RegexParser::new(&regex)?;
        let model = &mut self.models[self_.rep() as usize];

        Ok(model
            .stream_structured_text(&input, structure)
            .await?
            .text()
            .await)
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Model>) -> wasmtime::Result<()> {
        self.models.remove(rep.rep() as usize);
        Ok(())
    }
}
