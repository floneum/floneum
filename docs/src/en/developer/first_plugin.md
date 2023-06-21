# Your First plugin

Plugins can be created in any language that supports [WASM](https://webassembly.org). This example will use [rust](https://www.rust-lang.org/).

First, edit your cargo.toml to add the rust_adapter dependancy and change the crate type to a dynamicly linked (C-like) library
```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
rust_adapter = { path = "../../rust_adapter" }
```

Then create your plugin with the export plugin macro:
```rust
{{#include ../../../examples/add_plugin.rs:plugin}}
```

Next, build your plugin:
```sh
cargo build --target wasm32-unknown-unknown --release
```

> You can look at the default plugins [here](../../../plugins) to see how more complex plugins work

Finally, load your plugin by running the main Floneum application and typing the path to your `.wasm` file in the load plugin text box in the top left.
