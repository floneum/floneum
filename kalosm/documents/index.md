# Introduction

Floneum is a graph editor for AI workflows with a focus on community made plugins, local AI and safety.

![Demo Screenshot](https://github.com/Demonthos/floneum/assets/66571940/c60d621d-72b9-423c-b1d5-57cdb737e449)

## Features

- Visual interface: You can use Floneum without any knowledge of programming. The visual graph editor makes it easy to combine community made plugins with local AI models
- Instantly run local large language models: Floneum does not require any external dependencies or even a GPU to run. It uses [llm](https://github.com/rustformers/llm) to run large language models locally. Because of this, you can run Floneum with your data without worrying about privacy
- Plugins: By combining large language models with plugins, you can improve their performance and make models work better for your specific use case. All plugins run in an isolated environment so you don't need to trust any plugins you load. Plugins can only interact with their environment in a safe way
- Multi-language plugins: Plugins can use in any language that supports web assembly. In addition to the API that can be accessed in any language, Floneum has a rust wrapper with ergonomic macros that make it simple to create plugins
- Controlled text generation: Plugins can control the output of the large language models with a process similar to jsonformer or guidance. This allows plugins to force models to output valid json, or any other structure they define. This can be useful when communicating between a language model and a typed API

## Guide

- If you are looking to use Floneum, you can read the [User Documentation](./user/index.md).

- If you are looking to develop plugins for Floneum, you can read the [Developer Documentation](./developer/index.md)

- If you are interested in contributing to Floneum, you can read the [Contributing Documentation](./contributing/index.md)
