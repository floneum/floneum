// cargo build --target wasm32-wasi
// wasm-tools component new ./target/wasm32-wasi/debug/plugin.wasm \
// -o my-component.wasm --adapt ./wasi_snapshot_preview1.wasm


wit_bindgen::generate!();

struct MyHost;

impl HelloWorld for MyHost {
    fn greet() {
        println!("Hello, world!")
    }
}

export_hello_world!(MyHost);
