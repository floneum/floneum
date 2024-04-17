# Plugins

You can extend Floneum using [Web Assembly](https://webassembly.org/getting-started/developers-guide/).

Plugins can be created in any language that supports WASM. Currently, only Rust has a [launage-specific wrapper](https://github.com/floneum/floneum/tree/main/rust_adapter). If you would like to use another language, you can generate bindings using a tool like [wit-bindgen](https://github.com/bytecodealliance/wit-bindgen) with the [WASM interface types](https://github.com/WebAssembly/component-model) defined in the [plugin.wit](https://github.com/floneum/floneum/blob/main/wit/plugin.wit) file.
