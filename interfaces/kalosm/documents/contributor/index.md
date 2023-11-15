# Contributing

To start contributing to Floneum, we provide a list of [Good first issues](https://github.com/floneum/floneum/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

If you are interesting in contributing to Floneum, consider reaching out on [discord](https://discord.gg/dQdmhuB8q5). We are always happy to help new contributors get started!

## Project Architecture

Plugins:

[WASM](https://webassembly.org/):

Wasm is an assembly language originally built for the web. It has a few characteristics that make it appealing for Floneum plugins:

- Many languages can compile to it
- When you run WASM, it is sandboxed from you environment by default. You have to explicitly allow WASM plugins to access certain resources

[Wasmtime](https://github.com/bytecodealliance/wasmtime):

Wasmtime is a library Floneum uses to run WASM plugins

[WASI](https://wasi.dev/):

WASI is a set of interfaces for WASM programs. It is a set of resources you can allow your WASM plugin to access. Specifically, it allows controlled, file system, network, and time access. You can think of WASI as something like the "standard library" for WASM programs.


[WIT (wasm component model)](https://github.com/WebAssembly/component-model/blob/main/design/mvp/WIT.md):

The WASM component model allows the environment that runs WASM (Floneum) to declare a typed interface that Plugins (Nodes) can use. Because the interface is typed, each language can read the common environment declaration and create wrappers that work well with that language.

In Floneum, the interface is declared in the [plugin.wit](https://github.com/floneum/floneum/blob/master/wit/plugin.wit) file.

This file is used to create the language specific types in each plugin using [wit-bindgen](https://github.com/bytecodealliance/wit-bindgen). We also provide an extra level of wrapping with a rust macro and some special functions specific to rust in the [rust adapter](https://github.com/floneum/floneum/tree/master/rust_adapter) package

It is also used to declare the interface in wasmtime [here](https://github.com/floneum/floneum/blob/cd83ac7d3487826c54789619529db53125859923/plugin/src/lib.rs#L218). 

> Limitations:
> Currently, the Floneum bindings uses a lot of *-id types. These types represent resources that live in Floneum like a model instance. We also use ids to represent recursive types like the structure type used to constrain structured generation.
> As WIT matures, first party `resource` types will be implemented that make these id types unnecessary

UI:

The main UI for Floneum is written in [Diouxs](https://lib.rs/crates/dioxus). The code for the UI of floneum can be found in the [main package](https://github.com/floneum/floneum/tree/master/src)
