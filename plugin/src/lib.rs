use std::path::Path;

use crate::plugins::main::imports::*;
use exports::plugins::main::definitions::*;
use futures_util::Future;
use tokio::sync::broadcast;
use wasmtime::component::{Component, Linker};
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime::Store;
use wit_component::ComponentEncoder;

mod embedding;
mod infer;

use crate::infer::InferenceSessions;

wasmtime::component::bindgen!(in "../wit");

#[derive(Default)]
pub struct State {
    sessions: InferenceSessions,
}

impl Host for State {
    fn load_model(&mut self, ty: ModelType) -> std::result::Result<ModelId, wasmtime::Error> {
        Ok(self.sessions.create(ty))
    }

    fn unload_model(&mut self, id: ModelId) -> std::result::Result<(), wasmtime::Error> {
        self.sessions.remove(id);
        Ok(())
    }

    fn infer(
        &mut self,
        id: ModelId,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.sessions.infer(id, input, max_tokens, stop_on))
    }

    fn print(&mut self, str: String) -> std::result::Result<(), wasmtime::Error> {
        print!("{}", str);
        Ok(())
    }
}

pub struct PluginEngine {
    engine: Engine,
    linker: Linker<State>,
}

impl Default for PluginEngine {
    fn default() -> Self {
        let mut config = Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config).unwrap();
        let mut linker = Linker::new(&engine);
        PluginWorld::add_to_linker(&mut linker, |x| x).unwrap();
        Self { engine, linker }
    }
}

impl PluginEngine {
    pub fn load_plugin(&mut self, path: &Path) -> Plugin {
        println!("loading plugin");

        // we first read the bytes of the wasm module.
        let module = std::fs::read(path).unwrap();
        let size = module.len();
        println!("loaded plugin ({:01} mb)", size as f64 / (1024. * 1024.));
        // then we transform module to compoennt.
        // remember to get wasi_snapshot_preview1.wasm first.
        let component = ComponentEncoder::default()
            .module(module.as_slice())
            .unwrap()
            .validate(true)
            .adapter(
                "wasi_snapshot_preview1",
                &std::fs::read(&std::path::Path::new(
                    "/Users/evanalmloff/Desktop/Github/ai/plugin/wasi_snapshot_preview1.wasm",
                ))
                .unwrap(),
            )
            .unwrap()
            .encode()
            .unwrap();
        let component = Component::from_binary(&self.engine, &component).unwrap();

        // then we get the structure of the plugin.
        let mut store = Store::new(&self.engine, State::default());
        let (world, _instance) =
            PluginWorld::instantiate(&mut store, &component, &self.linker).unwrap();
        let structure = world.interface0.call_structure(&mut store).unwrap();

        Plugin {
            component,
            metadata: structure,
        }
    }
}

pub struct Plugin {
    metadata: Definition,
    component: Component,
}

impl Plugin {
    pub fn instance(&self, engine: &PluginEngine) -> PluginInstance {
        // create the store of models
        let mut store = Store::new(&engine.engine, State::default());
        let (world, _instance) =
            PluginWorld::instantiate(&mut store, &self.component, &engine.linker).unwrap();

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
    metadata: Definition,
    sender: broadcast::Sender<Vec<Value>>,
    reciever: broadcast::Receiver<Vec<Value>>,
}

impl Default for PluginInstance {
    fn default() -> Self {
        panic!("PluginInstance cannot be created by default")
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
            .current_dir("../plugin_demo")
            .stdout(std::process::Stdio::inherit())
            .output()
            .unwrap();

        println!("{:?}", command);

        let path = "../target/wasm32-unknown-unknown/release/plugin_demo.wasm";

        let mut engine = PluginEngine::default();

        let plugin = engine.load_plugin(&std::path::PathBuf::from(path));

        let mut instance = plugin.instance(&engine);

        let inputs = vec![Value::Text("hello world".to_string())];
        let outputs = instance.run(inputs).await;
        println!("{:?}", outputs);
    });
}
