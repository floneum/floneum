// Ideas:
// https://github.com/1rgs/jsonformer
// https://github.com/bytecodealliance/wasmtime/issues/6074

use once_cell::sync::Lazy;
use std::path::Path;
use std::sync::Arc;

use crate::plugins::main::imports::*;
use exports::plugins::main::definitions::*;
use futures_util::Future;
use json::{Structure, StructureMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use slab::Slab;
use tokio::sync::broadcast;
use wasmtime::component::{Component, Linker};
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime::Store;
use wit_component::ComponentEncoder;

mod download;
mod embedding;
mod json;
mod sessions;
mod structured;
mod vector_db;

use crate::sessions::InferenceSessions;

wasmtime::component::bindgen!({path: "../wit"});

#[derive(Default)]
pub struct State {
    sessions: InferenceSessions,
    structures: Slab<JsonStructure>,
}

impl State {
    fn decode_structure(&self, id: StructureId) -> Structure {
        match &self.structures[id.id as usize] {
            JsonStructure::Sequence(id) => {
                Structure::Sequence(Box::new(self.decode_structure(*id)))
            }
            JsonStructure::Map(map) => {
                let mut new_map = std::collections::HashMap::new();
                for kv in map.items.iter() {
                    let key = &kv.key;
                    let value = &kv.value;
                    new_map.insert(key.clone(), self.decode_structure(*value));
                }
                Structure::Map(StructureMap(new_map))
            }
            JsonStructure::Str(range) => Structure::String(range.min, range.max),
            JsonStructure::Num(range) => Structure::Num {
                min: range.min,
                max: range.max,
                integer: range.integer,
            },
            JsonStructure::Boolean => Structure::Bool,
            JsonStructure::Null => Structure::Null,
            JsonStructure::Either(either) => {
                let id1 = &either.first;
                let id2: &plugins::main::types::StructureId = &either.second;
                Structure::Either(
                    Box::new(self.decode_structure(*id1)),
                    Box::new(self.decode_structure(*id2)),
                )
            }
        }
    }
}

impl plugins::main::types::Host for State {}

impl Host for State {
    fn remove_json_structure(
        &mut self,
        id: StructureId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.structures.remove(id.id as usize);
        Ok(())
    }

    fn create_json_structure(
        &mut self,
        json: JsonStructure,
    ) -> std::result::Result<plugins::main::types::StructureId, wasmtime::Error> {
        let id = self.structures.insert(json);
        Ok(plugins::main::types::StructureId { id: id as u32 })
    }

    fn load_model(
        &mut self,
        ty: ModelType,
    ) -> std::result::Result<exports::plugins::main::definitions::ModelId, wasmtime::Error> {
        Ok(self.sessions.create(ty))
    }

    fn unload_model(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.sessions.remove(id);
        Ok(())
    }

    fn get_embedding(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        text: String,
    ) -> std::result::Result<plugins::main::types::Embedding, wasmtime::Error> {
        Ok(self.sessions.get_embedding(id, &text))
    }

    fn create_embedding_db(
        &mut self,
        embeddings: Vec<plugins::main::types::Embedding>,
        documents: Vec<String>,
    ) -> std::result::Result<exports::plugins::main::definitions::EmbeddingDbId, wasmtime::Error>
    {
        Ok(self.sessions.create_db(embeddings, documents))
    }

    fn remove_embedding_db(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(self.sessions.remove_embedding_db(id))
    }

    fn find_closest_documents(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        count: u32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        Ok(self.sessions.get_closest(id, search, count as usize))
    }

    fn find_documents_within(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        distance: f32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        Ok(self.sessions.get_within(id, search, distance))
    }

    fn infer(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.sessions.infer(id, input, max_tokens, stop_on))
    }

    fn infer_structured(
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

    fn print(&mut self, str: String) -> std::result::Result<(), wasmtime::Error> {
        print!("{}", str);
        Ok(())
    }
}

static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&*ENGINE);
    PluginWorld::add_to_linker(&mut linker, |x| x).unwrap();
    linker
});
static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true);
    Engine::new(&config).unwrap()
});

pub struct PluginEngine;

impl Default for PluginEngine {
    fn default() -> Self {
        Self
    }
}

impl PluginEngine {
    pub fn load_plugin(&mut self, path: &Path) -> Plugin {
        println!("loading plugin");

        // we first read the bytes of the wasm module.
        let module = std::fs::read(path).unwrap();
        self.load_plugin_from_bytes(module)
    }

    pub fn load_plugin_from_bytes(&mut self, bytes: Vec<u8>) -> Plugin {
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
        let component = Component::from_binary(&*ENGINE, &component).unwrap();

        // then we get the structure of the plugin.
        let mut store = Store::new(&*ENGINE, State::default());
        let (world, _instance) =
            PluginWorld::instantiate(&mut store, &component, &*LINKER).unwrap();
        let structure = world.interface0.call_structure(&mut store).unwrap();

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
        // Just serialize the bytes.
        serializer.serialize_bytes(&self.bytes)
    }
}

impl<'de> Deserialize<'de> for Plugin {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Just deserialize the bytes.
        println!("deserializing plugin");
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let mut engine = PluginEngine::default();
        println!("deserialized plugin");
        Ok(engine.load_plugin_from_bytes(bytes))
    }
}

impl Plugin {
    pub fn instance(&self) -> PluginInstance {
        // create the store of models
        let mut store = Store::new(&*ENGINE, State::default());
        let (world, _instance) =
            PluginWorld::instantiate(&mut store, &self.component, &LINKER).unwrap();

        let (input_sender, mut input_reciever) = broadcast::channel::<Vec<Value>>(100);
        let (output_sender, output_reciever) = broadcast::channel(100);

        tokio::spawn(async move {
            loop {
                let Ok(inputs) = input_reciever.recv().await else{break;};
                let borrowed = inputs.iter().collect::<Vec<_>>();
                let outputs = world.interface0.call_run(&mut store, &borrowed).unwrap();
                if output_sender.send(outputs).is_err() {
                    break;
                }
            }
            println!("plugin instance dropped... exiting");
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
    sender: broadcast::Sender<Vec<Value>>,
    reciever: broadcast::Receiver<Vec<Value>>,
}

impl Serialize for PluginInstance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Just serialize the bytes.
        serializer.serialize_bytes(&self.source_bytes)
    }
}

impl<'de> Deserialize<'de> for PluginInstance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Just deserialize the bytes.
        println!("deserializing plugin instance");
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        println!("deserialized plugin instance");
        Ok(PluginEngine.load_plugin_from_bytes(bytes).instance())
    }
}

impl PluginInstance {
    pub fn run(&self, inputs: Vec<Value>) -> impl Future<Output = Vec<Value>> + 'static {
        let sender = self.sender.clone();
        let mut reciever = self.reciever.resubscribe();
        async move {
            sender.send(inputs).unwrap();
            reciever.recv().await.unwrap()
        }
    }

    pub fn metadata(&self) -> &Definition {
        &self.metadata
    }
}

#[test]
fn load_plugin() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    runtime.block_on(async move {
        // first build the plugin_demo
        // cargo build --release --target wasm32-unknown-unknown
        let command = std::process::Command::new("cargo")
            .args(&["build", "--release", "--target", "wasm32-unknown-unknown"])
            .current_dir("../plugins/format")
            .stdout(std::process::Stdio::inherit())
            .output()
            .unwrap();

        println!("{:?}", command);

        let path = "../target/wasm32-unknown-unknown/release/plugin_format.wasm";

        let mut engine = PluginEngine::default();

        let plugin = engine.load_plugin(&std::path::PathBuf::from(path));

        let instance = plugin.instance();

        let inputs = vec![
            Value::Single(PrimitiveValue::Text("hello {}".to_string())),
            Value::Single(PrimitiveValue::Text("world".to_string())),
        ];
        let outputs = instance.run(inputs).await;
        println!("{:?}", outputs);

        assert_eq!(outputs.len(), 1);
        let first = outputs.first().unwrap();
        match first {
            Value::Single(PrimitiveValue::Text(text)) => {
                assert_eq!(text, "hello world");
            }
            _ => panic!("unexpected text output"),
        }
    });
}
