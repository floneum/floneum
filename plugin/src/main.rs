use crate::plugins::main::imports::*;
use wasmtime::component::{Component, Linker};
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime::Store;
use wit_component::{ComponentEncoder, StringEncoding};

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
        stop_on: String,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.sessions.infer(id, input, stop_on))
    }

    fn print(&mut self, str: String) -> std::result::Result<(), wasmtime::Error> {
        println!("{}", str);
        Ok(())
    }
}

fn main() {
    let mut config = Config::new();
    config.wasm_component_model(true);
    let engine = Engine::new(&config).unwrap();
    let mut store = Store::new(&engine, State::default());
    let mut linker = Linker::new(&engine);
    PluginWorld::add_to_linker(&mut linker, |x| x).unwrap();
    // we first read the bytes of the wasm module.
    let module =
        std::fs::read("../plugin_demo/target/wasm32-unknown-unknown/release/plugin_demo.wasm")
            .unwrap();
    // then we transform module to compoennt.
    // remember to get wasi_snapshot_preview1.wasm first.
    let component = ComponentEncoder::default()
        .module(module.as_slice())
        .unwrap()
        .validate(true)
        .adapter(
            "wasi_snapshot_preview1",
            &std::fs::read(&std::path::Path::new("./wasi_snapshot_preview1.wasm")).unwrap(),
        )
        .unwrap()
        .encode()
        .unwrap();
    let component = Component::from_binary(&engine, &component).unwrap();
    // after getting the component, we can instantiate a markdown instance.
    let (testing, _instance) = PluginWorld::instantiate(&mut store, &component, &linker).unwrap();
    // let res = markdown(&mut store, 1234).unwrap();
    panic!("res: {:?}", testing.call_start(&mut store));
}
