// Ideas:
// https://github.com/1rgs/jsonformer
// https://github.com/bytecodealliance/wasmtime/issues/6074

use once_cell::sync::Lazy;
use plugins::main::types::Structure;
use pollster::FutureExt;
use reqwest::header::{HeaderMap, HeaderName};
use std::path::Path;
use std::sync::{Arc, RwLock};
use wasmtime::component::__internal::async_trait;
use wasmtime_wasi::preview2::{self, DirPerms, FilePerms, WasiView};
use wasmtime_wasi::Dir;

use crate::plugins::main::imports::*;
use exports::plugins::main::definitions::*;
use futures_util::Future;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use slab::Slab;
use structured_parser::StructureParser;
use tokio::sync::broadcast;
use wasmtime::component::{Component, Linker};
use wasmtime::Engine;
use wasmtime::Store;
use wasmtime::{Config, Error};
use wit_component::ComponentEncoder;

mod browse;
mod download;
mod embedding;
mod sessions;
mod structured;
mod structured_parser;
mod vector_db;

use crate::sessions::InferenceSessions;
use crate::{exports::plugins::main::definitions::ModelType, vector_db::VectorDB};

static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    PluginWorld::add_to_linker(l, |x| x).unwrap();
    preview2::wasi::clocks::wall_clock::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::clocks::monotonic_clock::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::clocks::timezone::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::filesystem::filesystem::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::poll::poll::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::io::streams::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::random::random::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::cli_base::exit::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::cli_base::environment::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::cli_base::preopens::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::cli_base::stdin::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::cli_base::stdout::add_to_linker(l, |t| t).unwrap();
    preview2::wasi::cli_base::stderr::add_to_linker(l, |t| t).unwrap();
    linker
});
static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true).async_support(true);
    Engine::new(&config).unwrap()
});
static MULTI_PLUGIN_STATE: Lazy<RwLock<MultiPluginState>> = Lazy::new(Default::default);

#[derive(Default)]
struct MultiPluginState {
    vector_dbs: Slab<VectorDB<String>>,
}

impl MultiPluginState {
    fn vector_db_get(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> &VectorDB<String> {
        self.vector_dbs.get(id.id as usize).unwrap()
    }

    fn vector_db_get_mut(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> &mut VectorDB<String> {
        self.vector_dbs.get_mut(id.id as usize).unwrap()
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
    browser: Option<browse::Browser>,
    table: preview2::Table,
    ctx: preview2::WasiCtx,
}

impl Default for State {
    fn default() -> Self {
        let sandbox = Path::new("./sandbox");
        std::fs::create_dir_all(sandbox).unwrap();
        let ctx_builder = preview2::WasiCtxBuilder::new()
            .inherit_stderr()
            .inherit_stdin()
            .inherit_stdio()
            .inherit_stdout()
            .push_preopened_dir(
                Dir::open_ambient_dir(sandbox, wasmtime_wasi::sync::ambient_authority()).unwrap(),
                DirPerms::all(),
                FilePerms::all(),
                ".",
            )
            .set_clocks(wasmtime_wasi::preview2::clocks::host::clocks_ctx());
        let mut table = preview2::Table::new();
        let ctx = ctx_builder.build(&mut table).unwrap();
        State {
            sessions: InferenceSessions::default(),
            structures: Slab::new(),
            browser: None,
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
    fn browser_mut(&mut self) -> Result<&mut browse::Browser, wasmtime::Error> {
        if !self.browser.is_some() {
            let browser = browse::Browser::new()?;
            self.browser = Some(browser);
        }
        Ok(self.browser.as_mut().unwrap())
    }

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
    ) -> Vec<String> {
        MULTI_PLUGIN_STATE
            .read()
            .unwrap()
            .vector_db_get(id)
            .get_closest(embedding, n)
    }

    pub fn get_within(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: exports::plugins::main::definitions::Embedding,
        distance: f32,
    ) -> Vec<String> {
        MULTI_PLUGIN_STATE
            .read()
            .unwrap()
            .vector_db_get(id)
            .get_within(embedding, distance)
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
        Ok(self.get_closest(id, search, count as usize))
    }

    async fn find_documents_within(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        distance: f32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        Ok(self.get_within(id, search, distance))
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
        self.browser_mut()?.new_tab(headless)
    }

    async fn remove_tab(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.browser_mut()?.remove_tab(tab);
        Ok(())
    }

    async fn browse_to(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
        url: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.browser_mut()?.goto(tab, &url)?;
        Ok(())
    }

    async fn find_in_current_page(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
        query: String,
    ) -> std::result::Result<crate::plugins::main::imports::NodeId, wasmtime::Error> {
        Ok(self.browser_mut()?.find(tab, &query)?)
    }

    async fn get_element_text(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.browser_mut()?.get_text(id)?)
    }

    async fn click_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(self.browser_mut()?.click(id)?)
    }

    async fn type_into_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
        keys: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(self.browser_mut()?.send_keys(id, &keys)?)
    }

    async fn get_element_outer_html(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.browser_mut()?.outer_html(id)?)
    }

    async fn screenshot_browser(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
    ) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(self.browser_mut()?.screenshot(tab)?)
    }

    async fn screenshot_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(self.browser_mut()?.screenshot_of_id(id)?)
    }

    async fn find_child_of_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
        query: String,
    ) -> std::result::Result<crate::plugins::main::imports::NodeId, wasmtime::Error> {
        Ok(self.browser_mut()?.find_child(id, &query)?)
    }
}

#[derive(Default)]
pub struct PluginEngine {}

impl PluginEngine {
    pub async fn load_plugin(&mut self, path: &Path) -> Plugin {
        println!("loading plugin");

        // we first read the bytes of the wasm module.
        let module = std::fs::read(path).unwrap();
        self.load_plugin_from_bytes(module).await
    }

    pub async fn load_plugin_from_bytes(&mut self, bytes: Vec<u8>) -> Plugin {
        let size = bytes.len();
        println!("loaded plugin ({:01} mb)", size as f64 / (1024. * 1024.));
        // then we transform module to compoennt.
        // remember to get wasi_snapshot_preview1.wasm first.
        let component = ComponentEncoder::default()
            .module(bytes.as_slice())
            .unwrap()
            .validate(true)
            .adapter(
                "wasi_snapshot_preview1",
                include_bytes!("../wasi_snapshot_preview1.wasm",),
            )
            .unwrap()
            .encode()
            .unwrap();
        let component = Component::from_binary(&ENGINE, &component).unwrap();

        // then we get the structure of the plugin.
        let mut store = Store::new(&ENGINE, State::default());
        let (world, _instance) = PluginWorld::instantiate_async(&mut store, &component, &*LINKER)
            .await
            .unwrap();
        let structure = world.interface0.call_structure(&mut store).await.unwrap();

        Plugin {
            bytes: Arc::new(bytes),
            component,
            metadata: structure,
        }
    }
}

pub struct Plugin {
    bytes: Arc<Vec<u8>>,
    metadata: Definition,
    component: Component,
}

impl Serialize for Plugin {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Just serialize the bytes
        serializer.serialize_bytes(&self.bytes)
    }
}

impl<'de> Deserialize<'de> for Plugin {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Just deserialize the bytes
        let bytes = Vec::<u8>::deserialize(deserializer)?;

        Ok(async move {
            let mut engine = PluginEngine::default();
            engine.load_plugin_from_bytes(bytes).await
        }
        .block_on())
    }
}

impl Plugin {
    pub async fn instance(&self) -> PluginInstance {
        // create the store of models
        let mut store = Store::new(&ENGINE, State::default());
        let (world, _instance) =
            PluginWorld::instantiate_async(&mut store, &self.component, &LINKER)
                .await
                .unwrap();

        let (input_sender, mut input_reciever) = broadcast::channel::<Vec<Input>>(100);
        let (output_sender, output_reciever) = broadcast::channel(100);

        tokio::spawn(async move {
            loop {
                let Ok(inputs) = input_reciever.recv().await else{break;};
                let borrowed = inputs.iter().collect::<Vec<_>>();
                let outputs = world.interface0.call_run(&mut store, &borrowed).await;
                if output_sender.send(Arc::new(outputs)).is_err() {
                    break;
                }
            }
        });

        PluginInstance {
            source_bytes: self.bytes.clone(),
            sender: input_sender,
            reciever: output_reciever,
            metadata: self.metadata.clone(),
        }
    }

    pub fn name(&self) -> String {
        self.metadata.name.clone()
    }

    pub fn description(&self) -> String {
        self.metadata.description.clone()
    }
}

pub struct PluginInstance {
    source_bytes: Arc<Vec<u8>>,
    metadata: Definition,
    sender: broadcast::Sender<Vec<Input>>,
    reciever: broadcast::Receiver<Arc<Result<Vec<Output>, wasmtime::Error>>>,
}

impl Serialize for PluginInstance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Just serialize the bytes
        serializer.serialize_bytes(&self.source_bytes)
    }
}

impl<'de> Deserialize<'de> for PluginInstance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Just deserialize the bytes
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Ok(async move {
            PluginEngine::default()
                .load_plugin_from_bytes(bytes)
                .await
                .instance()
                .await
        }
        .block_on())
    }
}

impl PluginInstance {
    pub fn run(
        &self,
        inputs: Vec<Input>,
    ) -> impl Future<Output = Option<Arc<Result<Vec<Output>, Error>>>> + 'static {
        let sender = self.sender.clone();
        let mut reciever = self.reciever.resubscribe();
        async move {
            let _ = sender.send(inputs);
            reciever.recv().await.ok()
        }
    }

    pub fn metadata(&self) -> &Definition {
        &self.metadata
    }
}

#[tokio::test]
async fn load_plugin() {
    // first build the plugin_demo
    // cargo build --release --target wasm32-unknown-unknown
    let command = std::process::Command::new("cargo")
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .current_dir("./plugins/format")
        .stdout(std::process::Stdio::inherit())
        .output()
        .unwrap();

    println!("{:?}", command);

    let path = "./target/wasm32-unknown-unknown/release/plugin_format.wasm";

    let mut engine = PluginEngine::default();

    let plugin = engine.load_plugin(&std::path::PathBuf::from(path)).await;

    let instance = plugin.instance().await;

    let inputs = vec![
        Input::Single(PrimitiveValue::Text("hello {}".to_string())),
        Input::Single(PrimitiveValue::Text("world".to_string())),
    ];
    let outputs = instance.run(inputs).await.unwrap();
    let outputs = outputs.as_deref().clone().unwrap();
    println!("{:?}", outputs);

    assert_eq!(outputs.len(), 1);
    let first = outputs.first().unwrap();
    match first {
        Output::Single(PrimitiveValue::Text(text)) => {
            assert_eq!(text, "hello world");
        }
        _ => panic!("unexpected text output"),
    }
}

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
});
