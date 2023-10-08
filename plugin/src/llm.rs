use crate::host::State;
use crate::plugins::main;
use crate::plugins::main::imports::{self};
use crate::plugins::main::types::{
    EitherStructure, Embedding, EmbeddingDb, EmbeddingModel, Model, Node, NumberParameters, Page,
    SequenceParameters, Structure, ThenStructure, UnsignedRange,
};
use floneumin::floneumin_language::context::document::Document;
use floneumin::floneumin_language::floneumin_sample::structured::StructuredSampler;
use floneumin::floneumin_language::floneumin_sample::structured_parser::StructureParser;
use floneumin::floneumin_language::local::{Bert, LocalSession, Mistral, Phi};
use floneumin::floneumin_language::model::{Model as _, *};
use floneumin::floneumin_language::vector_db::VectorDB;
use once_cell::sync::Lazy;
use reqwest::header::{HeaderMap, HeaderName};
use slab::Slab;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use wasmtime::component::Linker;
use wasmtime::component::__internal::async_trait;
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime_wasi::preview2::{self, DirPerms, FilePerms, WasiView};
use wasmtime_wasi::Dir;

#[async_trait]
impl main::types::HostEmbeddingModel for State {
    async fn new(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingModel>> {
        let model = match ty {
            main::types::EmbeddingModelType::Mpt(mpt) => match mpt {
                main::types::MptType::Base => LocalSession::<MptBaseSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::MptType::Story => LocalSession::<MptStorySpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::MptType::Instruct => LocalSession::<MptInstructSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::MptType::Chat => LocalSession::<MptChatSpace>::start()
                    .await
                    .into_any_embedder(),
            },
            main::types::EmbeddingModelType::GptNeoX(neo) => match neo {
                main::types::GptNeoXType::LargePythia => LocalSession::<LargePythiaSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::GptNeoXType::TinyPythia => LocalSession::<TinyPythiaSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::GptNeoXType::DollySevenB => LocalSession::<DollySevenBSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::GptNeoXType::Stablelm => LocalSession::<StableLmSpace>::start()
                    .await
                    .into_any_embedder(),
            },
            main::types::EmbeddingModelType::Llama(llama) => match llama {
                main::types::LlamaType::Vicuna => LocalSession::<VicunaSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::LlamaType::Guanaco => LocalSession::<GuanacoSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::LlamaType::Wizardlm => LocalSession::<WizardLmSpace>::start()
                    .await
                    .into_any_embedder(),
                main::types::LlamaType::Orca => {
                    LocalSession::<OrcaSpace>::start().await.into_any_embedder()
                }
                main::types::LlamaType::LlamaSevenChat => {
                    LocalSession::<LlamaSevenChatSpace>::start()
                        .await
                        .into_any_embedder()
                }
                main::types::LlamaType::LlamaThirteenChat => {
                    LocalSession::<LlamaThirteenChatSpace>::start()
                        .await
                        .into_any_embedder()
                }
            },
            main::types::EmbeddingModelType::Bert => {
                Bert::new(Default::default())?.into_any_embedder()
            }
        };
        let idx = self.embedders.insert(model);

        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn model_downloaded(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<bool> {
        Ok(match ty {
            main::types::EmbeddingModelType::Mpt(mpt) => match mpt {
                main::types::MptType::Base => !LocalSession::<MptBaseSpace>::requires_download(),
                main::types::MptType::Story => !LocalSession::<MptStorySpace>::requires_download(),
                main::types::MptType::Instruct => {
                    !LocalSession::<MptInstructSpace>::requires_download()
                }
                main::types::MptType::Chat => !LocalSession::<MptChatSpace>::requires_download(),
            },
            main::types::EmbeddingModelType::GptNeoX(neo) => match neo {
                main::types::GptNeoXType::LargePythia => {
                    !LocalSession::<LargePythiaSpace>::requires_download()
                }
                main::types::GptNeoXType::TinyPythia => {
                    !LocalSession::<TinyPythiaSpace>::requires_download()
                }
                main::types::GptNeoXType::DollySevenB => {
                    !LocalSession::<DollySevenBSpace>::requires_download()
                }
                main::types::GptNeoXType::Stablelm => {
                    !LocalSession::<StableLmSpace>::requires_download()
                }
            },
            main::types::EmbeddingModelType::Llama(llama) => match llama {
                main::types::LlamaType::Vicuna => !LocalSession::<VicunaSpace>::requires_download(),
                main::types::LlamaType::Guanaco => {
                    !LocalSession::<GuanacoSpace>::requires_download()
                }
                main::types::LlamaType::Wizardlm => {
                    !LocalSession::<WizardLmSpace>::requires_download()
                }
                main::types::LlamaType::Orca => !LocalSession::<OrcaSpace>::requires_download(),
                main::types::LlamaType::LlamaSevenChat => {
                    !LocalSession::<LlamaSevenChatSpace>::requires_download()
                }
                main::types::LlamaType::LlamaThirteenChat => {
                    !LocalSession::<LlamaThirteenChatSpace>::requires_download()
                }
            },
            main::types::EmbeddingModelType::Bert => !Bert::requires_download(),
        })
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
            main::types::ModelType::Mpt(mpt) => match mpt {
                main::types::MptType::Base => {
                    LocalSession::<MptBaseSpace>::start().await.into_any_model()
                }
                main::types::MptType::Story => LocalSession::<MptStorySpace>::start()
                    .await
                    .into_any_model(),
                main::types::MptType::Instruct => LocalSession::<MptInstructSpace>::start()
                    .await
                    .into_any_model(),
                main::types::MptType::Chat => {
                    LocalSession::<MptChatSpace>::start().await.into_any_model()
                }
            },
            main::types::ModelType::GptNeoX(neo) => match neo {
                main::types::GptNeoXType::LargePythia => LocalSession::<LargePythiaSpace>::start()
                    .await
                    .into_any_model(),
                main::types::GptNeoXType::TinyPythia => LocalSession::<TinyPythiaSpace>::start()
                    .await
                    .into_any_model(),
                main::types::GptNeoXType::DollySevenB => LocalSession::<DollySevenBSpace>::start()
                    .await
                    .into_any_model(),
                main::types::GptNeoXType::Stablelm => LocalSession::<StableLmSpace>::start()
                    .await
                    .into_any_model(),
            },
            main::types::ModelType::Llama(llama) => match llama {
                main::types::LlamaType::Vicuna => {
                    LocalSession::<VicunaSpace>::start().await.into_any_model()
                }
                main::types::LlamaType::Guanaco => {
                    LocalSession::<GuanacoSpace>::start().await.into_any_model()
                }
                main::types::LlamaType::Wizardlm => LocalSession::<WizardLmSpace>::start()
                    .await
                    .into_any_model(),
                main::types::LlamaType::Orca => {
                    LocalSession::<OrcaSpace>::start().await.into_any_model()
                }
                main::types::LlamaType::LlamaSevenChat => {
                    LocalSession::<LlamaSevenChatSpace>::start()
                        .await
                        .into_any_model()
                }
                main::types::LlamaType::LlamaThirteenChat => {
                    LocalSession::<LlamaThirteenChatSpace>::start()
                        .await
                        .into_any_model()
                }
            },
            main::types::ModelType::Phi => Phi::builder().build()?.into_any_model(),
            main::types::ModelType::Mistral => Mistral::builder().build()?.into_any_model(),
        };
        let idx = self.models.insert(model);

        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn model_downloaded(&mut self, ty: main::types::ModelType) -> wasmtime::Result<bool> {
        Ok(match ty {
            main::types::ModelType::Mpt(mpt) => match mpt {
                main::types::MptType::Base => !LocalSession::<MptBaseSpace>::requires_download(),
                main::types::MptType::Story => !LocalSession::<MptStorySpace>::requires_download(),
                main::types::MptType::Instruct => {
                    !LocalSession::<MptInstructSpace>::requires_download()
                }
                main::types::MptType::Chat => !LocalSession::<MptChatSpace>::requires_download(),
            },
            main::types::ModelType::GptNeoX(neo) => match neo {
                main::types::GptNeoXType::LargePythia => {
                    !LocalSession::<LargePythiaSpace>::requires_download()
                }
                main::types::GptNeoXType::TinyPythia => {
                    !LocalSession::<TinyPythiaSpace>::requires_download()
                }
                main::types::GptNeoXType::DollySevenB => {
                    !LocalSession::<DollySevenBSpace>::requires_download()
                }
                main::types::GptNeoXType::Stablelm => {
                    !LocalSession::<StableLmSpace>::requires_download()
                }
            },
            main::types::ModelType::Llama(llama) => match llama {
                main::types::LlamaType::Vicuna => !LocalSession::<VicunaSpace>::requires_download(),
                main::types::LlamaType::Guanaco => {
                    !LocalSession::<GuanacoSpace>::requires_download()
                }
                main::types::LlamaType::Wizardlm => {
                    !LocalSession::<WizardLmSpace>::requires_download()
                }
                main::types::LlamaType::Orca => !LocalSession::<OrcaSpace>::requires_download(),
                main::types::LlamaType::LlamaSevenChat => {
                    !LocalSession::<LlamaSevenChatSpace>::requires_download()
                }
                main::types::LlamaType::LlamaThirteenChat => {
                    !LocalSession::<LlamaThirteenChatSpace>::requires_download()
                }
            },
            main::types::ModelType::Phi => !Phi::requires_download(),
            main::types::ModelType::Mistral => !Mistral::requires_download(),
        })
    }

    async fn infer(
        &mut self,
        self_: wasmtime::component::Resource<Model>,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> wasmtime::Result<String> {
        let parameters = GenerationParameters::default()
            .with_max_length(max_tokens.unwrap_or(u32::MAX))
            .with_stop_on(stop_on);
        Ok(self.models[self_.rep() as usize]
            .generate_text(&input, parameters)
            .await?)
    }

    async fn infer_structured(
        &mut self,
        self_: wasmtime::component::Resource<Model>,
        input: String,
        max_tokens: Option<u32>,
        structure: wasmtime::component::Resource<Structure>,
    ) -> wasmtime::Result<String> {
        let decoded_structure = self.get_full_structured_parser(&structure).ok_or_else(|| {
            anyhow::Error::msg(
                "Structure is not a valid structure. This is a bug in the plugin host.",
            )
        })?;
        let model = &mut self.models[self_.rep() as usize];

        let structured = StructuredSampler::new(decoded_structure.clone(), 0, model.tokenizer());

        Ok(model
            .generate_text_with_sampler(&input, max_tokens, None, Arc::new(Mutex::new(structured)))
            .await?)
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Model>) -> wasmtime::Result<()> {
        self.models.remove(rep.rep() as usize);
        Ok(())
    }
}
