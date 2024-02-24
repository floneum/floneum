# [Floneum](./floneum/floneum)/[Kalosm](./interfaces/kalosm/)

Floneum is a graph editor that makes it easy to develop your own AI workflows

<img width="1512" alt="Screenshot 2023-06-18 at 4 26 11 PM" src="https://floneum.com/assets/question_answer_example.png">

## Features

- Visual interface: You can use Floneum without any knowledge of programming. The visual graph editor makes it easy to combine community-made plugins with local AI models
- Instantly run local large language models: Floneum does not require any external dependencies or even a GPU to run. It uses [LLM](https://github.com/rustformers/llm) to run large language models locally. Because of this, you can run Floneum with your data without worrying about privacy
- Plugins: By combining large language models with plugins, you can improve their performance and make models work better for your specific use case. All plugins run in an isolated environment so you don't need to trust any plugins you load. Plugins can only interact with their environment in a safe way
- Multi-language plugins: Plugins can be used in any language that supports web assembly. In addition to the API that can be accessed in any language, Floneum has a rust wrapper with ergonomic macros that make it simple to create plugins
- Controlled text generation: Plugins can control the output of the large language models with a process similar to JSONformer or guidance. This allows plugins to force models to output valid JSON, or any other structure they define. This can be useful when communicating between a language model and a typed API

## Floneum Quickstart

[Download the latest release](https://github.com/floneum/floneum/releases/tag/v0.2.0), run the binary, wait a few seconds for all of the plugins to download and start building!

## Documentation

- If you are looking to use Floneum, you can read the [User Documentation](https://floneum.com/docs/user/).

- If you are looking to develop plugins for Floneum, you can read the [Developer Documentation](https://floneum.com/docs/developer/)

## Kalosm

[Kalosm](./interfaces/kalosm/) is a simple interface for pre-trained models in rust that backs Floneum. It makes it easy to interact with pre-trained, language, audio, and image models.

There are three different packages in Kalosm:

kalosm::language - A simple interface for text generation and embedding models and surrounding tools. It includes support for search databases, and text collection from websites, RSS feeds, and search engines.
kalosm::audio - A simple interface for audio transcription and surrounding tools. It includes support for microphone input and the whisper model.
kalosm::vision - A simple interface for image generation and segmentation models and surrounding tools. It includes support for the wuerstchen and segment-anything models and integration with the image crate.
A complete guide for Kalosm is available on the Kalosm website, and examples are available in the examples folder.

Kalosm is a simple interface for pre-trained models in rust. It makes it easy to interact with pre-trained, language, audio, and image models.

There are three different packages in Kalosm:
- `kalosm::language` - A simple interface for text generation and embedding models and surrounding tools. It includes support for search databases, and text collection from websites, RSS feeds, and search engines.
- `kalosm::audio` - A simple interface for audio transcription and surrounding tools. It includes support for microphone input and the `whisper` model.
- `kalosm::vision` - A simple interface for image generation and segmentation models and surrounding tools. It includes support for the `wuerstchen` and `segment-anything` models and integration with the [image](https://docs.rs/image/latest/image/) crate.

A complete guide for Kalosm is available on the [Kalosm website](https://floneum.com/kalosm/), and examples are available in the [examples folder](https://github.com/floneum/floneum/tree/master/interfaces/kalosm/examples).

### Kalosm Quickstart!

1) Install [rust](https://rustup.rs/)
2) Create a new project:
```sh
cargo new next-gen-ai
cd ./next-gen-ai
```
3) Add Kalosm as a dependency
```sh
cargo add kalosm
cargo add tokio --features full
```
4) Add this code to your `main.rs` file
```rust, no_run
use std::io::Write;

use kalosm::{*, language::*};

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "The following is a 300 word essay about Paris:";
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(1000).await.unwrap();

    let mut sentences = stream.words();
    while let Some(text) = sentences.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
```
5) Run your application with:
```sh
cargo run --release
```

## Community

If you are interested in either project, you can join the [discord](https://discord.gg/dQdmhuB8q5) to discuss the project and get help.

## Contributing

- Report issues on our [issue tracker](https://github.com/floneum/floneum/issues).
- Help other users in the discord
- If you are interested in contributing, feel free to reach out on discord
