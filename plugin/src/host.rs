use crate::plugins::main::imports::*;
use exports::plugins::main::definitions::*;
use floneumin::floneumin_language::context::page::browse::Browser;
use floneumin::floneumin_language::embedding::VectorSpace;
use floneumin::floneumin_language::floneumin_sample::structured_parser::StructureParser;
use floneumin::floneumin_language::model::ModelType;
use floneumin::floneumin_language::vector_db::VectorDB;
use floneumite::PackageIndexEntry;
use futures_util::Future;
use once_cell::sync::Lazy;
use plugins::main::types::Structure;
use pollster::FutureExt;
use reqwest::header::{HeaderMap, HeaderName};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use slab::Slab;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LockResult, RwLock, RwLockReadGuard};
use tokio::sync::broadcast;
use wasmtime::component::__internal::async_trait;
use wasmtime::component::{Component, Linker};
use wasmtime::Engine;
use wasmtime::Store;
use wasmtime::{Config, Error};
use wasmtime_wasi::preview2::{self, DirPerms, FilePerms, WasiView};
use wasmtime_wasi::Dir;
use wit_component::ComponentEncoder;

pub(crate) static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    PluginWorld::add_to_linker(l, |x| x).unwrap();
    preview2::command::add_to_linker(l)?;
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
static MULTI_PLUGIN_STATE: Lazy<RwLock<MultiPluginState>> = Lazy::new(Default::default);

struct UnknownVectorSpace;

impl VectorSpace for UnknownVectorSpace {}

#[derive(Default)]
struct MultiPluginState {
    vector_dbs: Slab<VectorDB<String, UnknownVectorSpace>>,
    browser: Browser,
}

impl MultiPluginState {
    fn vector_db_get(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> Option<&VectorDB<String, UnknownVectorSpace>> {
        self.vector_dbs.get(id.id as usize)
    }

    fn vector_db_get_mut(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> Option<&mut VectorDB<String, UnknownVectorSpace>> {
        self.vector_dbs.get_mut(id.id as usize)
    }

    pub fn create_db(
        &mut self,
        embedding: Vec<exports::plugins::main::definitions::Embedding>,
        documents: Vec<String>,
    ) -> exports::plugins::main::definitions::EmbeddingDbId {
        let idx = self.vector_dbs.insert(VectorDB::new(embedding, documents));

        exports::plugins::main::definitions::EmbeddingDbId { id: idx as u32 }
    }

    pub fn remove_embedding_db(&mut self, id: exports::plugins::main::definitions::EmbeddingDbId) {
        self.vector_dbs.remove(id.id as usize);
    }
}

pub struct State {
    sessions: InferenceSessions,
    structures: Slab<Structure>,
    logs: Arc<RwLock<Vec<String>>>,
    plugin_state: HashMap<Vec<u8>, Vec<u8>>,
    table: preview2::Table,
    ctx: preview2::WasiCtx,
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
        let mut table = preview2::Table::new();
        let ctx = ctx_builder.build(&mut table).unwrap();
        State {
            sessions: InferenceSessions::default(),
            structures: Slab::new(),
            plugin_state: Default::default(),
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

impl State {
    fn decode_structure(&self, id: StructureId) -> StructureParser {
        match &self.structures[id.id as usize] {
            Structure::Literal(text) => StructureParser::Literal(text.clone()),
            Structure::Str(range) => StructureParser::String {
                min_len: range.min,
                max_len: range.max,
            },
            Structure::Num(range) => StructureParser::Num {
                min: range.min,
                max: range.max,
                integer: range.integer,
            },
            Structure::Or(params) => StructureParser::Either {
                first: Box::new(self.decode_structure(params.first)),
                second: Box::new(self.decode_structure(params.second)),
            },
            Structure::Then(params) => StructureParser::Then {
                first: Box::new(self.decode_structure(params.first)),
                second: Box::new(self.decode_structure(params.second)),
            },
            Structure::Sequence(params) => StructureParser::Sequence {
                item: Box::new(self.decode_structure(params.item)),
                separator: Box::new(self.decode_structure(params.separator)),
                min_len: params.min_len,
                max_len: params.max_len,
            },
        }
    }

    pub fn get_closest(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: exports::plugins::main::definitions::Embedding,
        n: usize,
    ) -> Option<Vec<String>> {
        MULTI_PLUGIN_STATE
            .read()
            .unwrap()
            .vector_db_get(id)
            .map(|db| db.get_closest(embedding, n))
    }

    pub fn get_within(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: exports::plugins::main::definitions::Embedding,
        distance: f32,
    ) -> Option<Vec<String>> {
        MULTI_PLUGIN_STATE
            .read()
            .unwrap()
            .vector_db_get(id)
            .map(|db| db.get_within(embedding, distance))
    }
}

impl plugins::main::types::Host for State {}

#[async_trait]
impl Host for State {
    async fn get_request(
        &mut self,
        url: String,
        headers: Vec<Header>,
    ) -> std::result::Result<String, wasmtime::Error> {
        let client = reqwest::Client::new();
        let mut header_map = HeaderMap::new();

        header_map.append(
            reqwest::header::USER_AGENT,
            // Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0
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

    async fn remove_structure(
        &mut self,
        id: StructureId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.structures.remove(id.id as usize);
        Ok(())
    }

    async fn create_structure(
        &mut self,
        structure: Structure,
    ) -> std::result::Result<plugins::main::types::StructureId, wasmtime::Error> {
        let id = self.structures.insert(structure);
        Ok(plugins::main::types::StructureId { id: id as u32 })
    }

    async fn model_downloaded(
        &mut self,
        ty: exports::plugins::main::definitions::ModelType,
    ) -> std::result::Result<bool, wasmtime::Error> {
        let model_type: ModelType = ty.into();
        Ok(!model_type.requires_download())
    }

    async fn load_model(
        &mut self,
        ty: exports::plugins::main::definitions::ModelType,
    ) -> std::result::Result<exports::plugins::main::definitions::ModelId, wasmtime::Error> {
        Ok(self.sessions.create(ty))
    }

    async fn unload_model(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.sessions.remove(id);
        Ok(())
    }

    async fn get_embedding(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        text: String,
    ) -> std::result::Result<plugins::main::types::Embedding, wasmtime::Error> {
        Ok(self.sessions.get_embedding(id, &text))
    }

    async fn create_embedding_db(
        &mut self,
        embeddings: Vec<plugins::main::types::Embedding>,
        documents: Vec<String>,
    ) -> std::result::Result<exports::plugins::main::definitions::EmbeddingDbId, wasmtime::Error>
    {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .unwrap()
            .create_db(embeddings, documents))
    }

    async fn add_embedding(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: plugins::main::types::Embedding,
        document: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .unwrap()
            .vector_db_get_mut(id)
            .ok_or(wasmtime::Error::msg("Invalid embedding db id"))?
            .add_embedding(embedding, document);
        Ok(())
    }

    async fn remove_embedding_db(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE.write().unwrap().remove_embedding_db(id);
        Ok(())
    }

    async fn find_closest_documents(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        count: u32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        self.get_closest(id, search, count as usize)
            .ok_or(wasmtime::Error::msg("Invalid embedding db id"))
    }

    async fn find_documents_within(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        distance: f32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        self.get_within(id, search, distance)
            .ok_or(wasmtime::Error::msg("Invalid embedding db id"))
    }

    async fn infer(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.sessions.infer(id, input, max_tokens, stop_on))
    }

    async fn infer_structured(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        input: String,
        max_tokens: Option<u32>,
        structure: StructureId,
    ) -> std::result::Result<String, wasmtime::Error> {
        let structure = self.decode_structure(structure);
        Ok(self
            .sessions
            .infer_validate(id, input, max_tokens, structure))
    }

    async fn new_tab(
        &mut self,
        headless: bool,
    ) -> std::result::Result<crate::plugins::main::imports::TabId, wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .new_tab(headless)
    }

    async fn remove_tab(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .remove_tab(tab);
        Ok(())
    }

    async fn browse_to(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
        url: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .goto(tab, &url)?;
        Ok(())
    }

    async fn find_in_current_page(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
        query: String,
    ) -> std::result::Result<crate::plugins::main::imports::NodeId, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .find(tab, &query)?)
    }

    async fn get_element_text(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .get_text(id)?)
    }

    async fn click_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .click(id)?)
    }

    async fn type_into_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
        keys: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .send_keys(id, &keys)?)
    }

    async fn get_element_outer_html(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .outer_html(id)?)
    }

    async fn screenshot_browser(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
    ) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .screenshot(tab)?)
    }

    async fn screenshot_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .screenshot_of_id(id)?)
    }

    async fn find_child_of_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
        query: String,
    ) -> std::result::Result<crate::plugins::main::imports::NodeId, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .find_child(id, &query)?)
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
});
