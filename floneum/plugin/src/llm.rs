use crate::host::State;
use crate::plugins::main;

use crate::plugins::main::types::{Embedding, EmbeddingModel, Model};

use kalosm::language::*;

use wasmtime::component::__internal::async_trait;

#[async_trait]
impl main::types::HostEmbeddingModel for State {
    async fn new(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingModel>> {
        let model = match ty {
            main::types::EmbeddingModelType::Bert => {
                Bert::builder().build().await?.into_any_embedder()
            }
        };
        let idx = self.embedders.insert(model);

        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn model_downloaded(
        &mut self,
        _ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<bool> {
        Ok(false)
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

#[async_trait]
impl main::types::HostModel for State {
    async fn new(
        &mut self,
        ty: main::types::ModelType,
    ) -> wasmtime::Result<wasmtime::component::Resource<Model>> {
        let model = match ty {
            main::types::ModelType::MistralSeven => Llama::builder()
                .with_source(LlamaSource::mistral_7b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::MistralSevenInstruct => Llama::builder()
                .with_source(LlamaSource::mistral_7b_instruct())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::MistralSevenInstructTwo => Llama::builder()
                .with_source(LlamaSource::mistral_7b_instruct_2())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::ZephyrSevenAlpha => Llama::builder()
                .with_source(LlamaSource::zephyr_7b_alpha())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::ZephyrSevenBeta => Llama::builder()
                .with_source(LlamaSource::zephyr_7b_beta())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::OpenChatSeven => Llama::builder()
                .with_source(LlamaSource::open_chat_7b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::StarlingSevenAlpha => Llama::builder()
                .with_source(LlamaSource::starling_7b_alpha())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::TinyLlamaChat => Llama::builder()
                .with_source(LlamaSource::tiny_llama_1_1b_chat())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::TinyLlama => Llama::builder()
                .with_source(LlamaSource::tiny_llama_1_1b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaSeven => Llama::builder()
                .with_source(LlamaSource::llama_7b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaThirteen => Llama::builder()
                .with_source(LlamaSource::llama_13b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaSeventy => Llama::builder()
                .with_source(LlamaSource::llama_70b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaSevenChat => Llama::builder()
                .with_source(LlamaSource::llama_7b_chat())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaThirteenChat => Llama::builder()
                .with_source(LlamaSource::llama_13b_chat())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaSeventyChat => Llama::builder()
                .with_source(LlamaSource::llama_70b_chat())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaSevenCode => Llama::builder()
                .with_source(LlamaSource::llama_7b_code())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaThirteenCode => Llama::builder()
                .with_source(LlamaSource::llama_13b_code())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::LlamaThirtyFourCode => Llama::builder()
                .with_source(LlamaSource::llama_34b_code())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::SolarTen => Llama::builder()
                .with_source(LlamaSource::solar_10_7b())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::SolarTenInstruct => Llama::builder()
                .with_source(LlamaSource::solar_10_7b_instruct())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::PhiOne => Phi::builder()
                .with_source(PhiSource::v1())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::PhiOnePointFive => Phi::builder()
                .with_source(PhiSource::v1_5())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::PhiTwo => Phi::builder()
                .with_source(PhiSource::v2())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::PuffinPhiTwo => Phi::builder()
                .with_source(PhiSource::puffin_phi_v2())
                .build()
                .await?
                .into_any_model(),
            main::types::ModelType::DolphinPhiTwo => Phi::builder()
                .with_source(PhiSource::dolphin_phi_v2())
                .build()
                .await?
                .into_any_model(),
        };
        let idx = self.models.insert(model);

        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn model_downloaded(&mut self, _ty: main::types::ModelType) -> wasmtime::Result<bool> {
        // TODO: actually check if the model is downloaded
        Ok(false)
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
        let structure = RegexParser::new(&regex).map_err(|e| wasmtime::Error::msg(format!("{e:?}")))?;
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
