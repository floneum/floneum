use self::plugins::main;
use self::plugins::main::imports::{self, Header};
use self::plugins::main::types::{
    Embedding, EmbeddingDb, EmbeddingModel, Model, Node, Page, Structure, NumberParameters, UnsignedRange, EitherStructure, ThenStructure, SequenceParameters,
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

enum StructureType {
    Num(NumberParameters),
    Str(UnsignedRange),
    Literal(String),
    Or(EitherStructure),
    Then(ThenStructure),
    Sequence(SequenceParameters),
  }


pub(crate) static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    Exports::add_to_linker(l, |x| x).unwrap();
    preview2::command::add_to_linker(l).unwrap();
    // preview2::bindings::filesystem::types::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::filesystem::preopens::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::io::streams::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::environment::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::exit::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::stdin::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::stdout::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::stderr::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_input::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_output::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_stdin::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_stdout::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_stderr::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::clocks::monotonic_clock::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::clocks::timezone::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::clocks::wall_clock::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::random::insecure::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::random::insecure_seed::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::random::random::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::network::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::instance_network::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::tcp::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::tcp_create_socket::add_to_linker(&mut linker, |x| x).unwrap();

    linker
});
pub(crate) static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true).async_support(true);
    Engine::new(&config).unwrap()
});

pub struct State {
    pub(crate) logs: Arc<RwLock<Vec<String>>>,
    structures: Slab<StructureType>,
    models: Slab<DynModel>,
    embedders: Slab<DynEmbedder>,
    embedding_dbs: Slab<VectorDB<Document>>,
    plugin_state: HashMap<Vec<u8>, Vec<u8>>,
    table: preview2::Table,
    ctx: preview2::WasiCtx,
}

impl State {
    fn get_full_structured_parser(
        &self,
        structure: &wasmtime::component::Resource<Structure>,
    ) -> Option<StructureParser> {
        match self.structures.get(structure.rep() as usize)? {
            StructureType::Num(num) => Some(StructureParser::Num {
                min: num.min,
                max: num.max,
                integer: num.integer,
            }),
            StructureType::Str(str) => Some(StructureParser::String {
                min_len: str.min,
                max_len: str.max,
            }),
            StructureType::Literal(literal) => Some(StructureParser::Literal(literal.clone())),
            StructureType::Or(or) => Some(StructureParser::Either {
                first: Box::new(self.get_full_structured_parser(&or.first)?),
                second: Box::new(self.get_full_structured_parser(&or.second)?),
            }),
            StructureType::Then(then) => Some(StructureParser::Then {
                first: Box::new(self.get_full_structured_parser(&then.first)?),
                second: Box::new(self.get_full_structured_parser(&then.second)?),
            }),
            StructureType::Sequence(sequence) => Some(StructureParser::Sequence {
                item: Box::new(self.get_full_structured_parser(&sequence.item)?),
                separator: Box::new(self.get_full_structured_parser(&sequence.separator)?),
                min_len: sequence.min_len,
                max_len: sequence.max_len,
            }),
        }
    }
}

impl Default for State {
    fn default() -> Self {
        let sandbox = Path::new("./sandbox");
        std::fs::create_dir_all(sandbox).unwrap();
        let mut ctx = preview2::WasiCtxBuilder::new();
        let ctx_builder = ctx
            .inherit_stderr()
            .inherit_stdin()
            .inherit_stdio()
            .inherit_stdout()
            .preopened_dir(
                Dir::open_ambient_dir(sandbox, wasmtime_wasi::sync::ambient_authority()).unwrap(),
                DirPerms::all(),
                FilePerms::all(),
                ".",
            );
        let table = preview2::Table::new();
        let ctx = ctx_builder.build();
        State {
            plugin_state: Default::default(),
            structures: Default::default(),
            embedders: Default::default(),
            models: Default::default(),
            embedding_dbs: Default::default(),
            logs: Default::default(),
            table,
            ctx,
        }
    }
}

impl WasiView for State {
    fn table(&self) -> &preview2::Table {
        &self.table
    }

    fn table_mut(&mut self) -> &mut preview2::Table {
        &mut self.table
    }

    fn ctx(&self) -> &preview2::WasiCtx {
        &self.ctx
    }

    fn ctx_mut(&mut self) -> &mut preview2::WasiCtx {
        &mut self.ctx
    }
}

impl main::types::Host for State {}

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

    async fn model_downloaded(
        &mut self,
        ty: main::types::ModelType,
    ) -> wasmtime::Result<bool> {
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

#[async_trait]
impl main::types::HostStructure for State {
    async fn num(
        &mut self,
        num: main::types::NumberParameters,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Num(num));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn str(
        &mut self,
        str: main::types::UnsignedRange,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Str(str));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn literal(
        &mut self,
        literal: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Literal(literal));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn or(
        &mut self,
        or: main::types::EitherStructure,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Or(or));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn then(
        &mut self,
        then: main::types::ThenStructure,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Then(then));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn sequence(
        &mut self,
        sequence: main::types::SequenceParameters,
    ) -> wasmtime::Result<wasmtime::component::Resource<Structure>> {
        let idx = self.structures.insert(StructureType::Sequence(sequence));
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Structure>) -> wasmtime::Result<()> {
        self.structures.remove(rep.rep() as usize);
        Ok(())
    }
}

#[async_trait]
impl main::types::HostEmbeddingDb for State {
    async fn new(
        &mut self,
        embeddings: Vec<Embedding>,
        documents: Vec<String>,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingDb>> {
        let embeddings = embeddings.into_iter().map(|x| x.vector.into()).collect();
        let documents = documents
            .into_iter()
            .map(|x| Document::from_parts(String::new(), x))
            .collect();
        let db = VectorDB::new(embeddings, documents);
        let idx = self.embedding_dbs.insert(db);
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn add_embedding(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        embedding: Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        self.embedding_dbs[self_.rep() as usize].add_embedding(
            embedding.vector.into(),
            Document::from_parts(String::new(), document),
        );
        Ok(())
    }

    async fn find_closest_documents(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        let documents = self.embedding_dbs[self_.rep() as usize]
            .get_closest(search.vector.into(), count as usize);
        Ok(documents
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    async fn find_documents_within(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        distance: f32,
    ) -> wasmtime::Result<Vec<String>> {
        Ok(self.embedding_dbs[self_.rep() as usize]
            .get_within(search.vector.into(), distance)
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<EmbeddingDb>) -> wasmtime::Result<()> {
        self.embedding_dbs.remove(rep.rep() as usize);
        Ok(())
    }
}

#[async_trait]
impl main::types::HostNode for State {
    async fn get_element_text(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        todo!()
    }

    async fn click_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn type_into_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
        keys: String,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn get_element_outer_html(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        todo!()
    }

    async fn screenshot_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<Vec<u8>> {
        todo!()
    }

    async fn find_child_of_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
        query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        todo!()
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Node>) -> wasmtime::Result<()> {
        todo!()
    }
}

#[async_trait]
impl main::types::HostPage for State {
    async fn new(
        &mut self,
        mode: main::types::BrowserMode,
        url: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Page>> {
        todo!()
    }

    async fn find_in_current_page(
        &mut self,
        self_: wasmtime::component::Resource<Page>,
        query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        todo!()
    }

    async fn screenshot_browser(
        &mut self,
        self_: wasmtime::component::Resource<Page>,
    ) -> wasmtime::Result<Vec<u8>> {
        todo!()
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Page>) -> wasmtime::Result<()> {
        todo!()
    }
}

#[async_trait]
impl imports::Host for State {
    async fn get_request(
        &mut self,
        url: String,
        headers: Vec<Header>,
    ) -> std::result::Result<String, wasmtime::Error> {
        let client = reqwest::Client::new();
        let mut header_map = HeaderMap::new();

        header_map.append(
            reqwest::header::USER_AGENT,
            "Floneum/0.1.0 (Unknown; Unknown; Unknown; Unknown) Floneum/0.1.0 Floneum/0.1.0"
                .parse()
                .unwrap(),
        );

        for header in headers {
            header_map.append(HeaderName::try_from(&header.key)?, header.value.parse()?);
        }

        let response = client.get(&url).headers(header_map).send().await?;
        Ok(response.text().await?)
    }

    async fn log_to_user(&mut self, message: String) -> std::result::Result<(), wasmtime::Error> {
        let mut logs = self
            .logs
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock logs: {}", e)))?;
        if logs.len() >= 100 {
            logs.remove(0);
        }
        logs.push(message);
        Ok(())
    }

    async fn store(
        &mut self,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.plugin_state.insert(key, value);

        Ok(())
    }

    async fn load(&mut self, key: Vec<u8>) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(self.plugin_state.get(&key).cloned().unwrap_or_default())
    }

    async fn unload(&mut self, key: Vec<u8>) -> std::result::Result<(), wasmtime::Error> {
        self.plugin_state.remove(&key);
        Ok(())
    }
}

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
    world: "exports",
});
