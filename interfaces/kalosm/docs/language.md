# Kalosm Language

Language processing utilities for the Kalosm framework.


The language part of kalosm has a few core parts:
- Models: Text generation and embedding models
- Context: Document collection, format support, search and chunking
- Integrations: SurrealDB, Serper, and other integrations
- Structured Generation: Structured generation with regex and other constraints


## Text Generation Models

[`Model`] and [`ModelExt`] are the core traits for text generation models. Any model that implements these traits can be used with Kalosm.


The simplest way to use a model is to create a [`Model`] and call [`ModelExt::stream_text`] to stream text from it:

```rust, no_run
use kalosm::language::*;
#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");
    let stream = llm
        // Any model that implements the Model trait can be used to stream text
        .stream_text(prompt)
        // You can pass parameters to the model to control the output
        .with_max_length(300)
        // And run .await to start streaming
        .await
        .unwrap();
    // You can then use the stream however you need
    stream.to_std_out().await.unwrap();
}
```


### Structured Generation

Structured generation gives you more control over the output of the text generation. 


## Embedding Models

- [`Embedder`] and [`EmbedderExt`]: The core traits for text embedding models. This trait is implemented for all models that can be used with the Kalosm framework.
```rust, no_run
use kalosm::language::*;
#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");
    let document = Url::parse("https://floneum.com/kalosm/docs/user/introduction").unwrap().into_document().await.unwrap();
    let stream = llm
        // Any model that implements the Embedder trait can be used to stream text
        .stream_text(prompt)
        // You can pass parameters to the model to control the output
        .with_max_length(300)
        // And run .await to start streaming
        .await
        .unwrap();
    // You can then use the stream however you need
}
```