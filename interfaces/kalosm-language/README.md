# Kalosm Language

Language processing utilities for the Kalosm framework.


The language part of Kalosm has a few core parts:
- Models: [Text generation](prelude::ModelExt) and [embedding models](prelude::EmbedderExt)
- Context: [Document collection](prelude::Document), [format support](prelude::FsDocument), [search](prelude::SearchQuery) and [chunking](prelude::Chunker)
- Integrations: SurrealDB, [Serper](prelude::SearchQuery), and other integrations

## Text Generation Models

[`Model`](prelude::Model) and [`ModelExt`](prelude::ModelExt) are the core traits for text generation models. Any model that implements these traits can be used with Kalosm.


The simplest way to use a model is to create a [llama model](prelude::Llama) and call [stream_text](prelude::ModelExt::stream_text) on it:

```rust, no_run
use kalosm::language::*;
#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");
    // Any model that implements the [`TextCompletionModel`] trait can be used to stream text
    let mut stream = llm.complete(prompt);
    // You can then use the stream however you need. to_std_out will print the text to the console as it is generated
    stream.to_std_out().await.unwrap();
}
```

### Tasks

You can define a Task with a description then run it with an input. The task will cache the description to repeated calls faster. Tasks work with chat models.

```rust, no_run
use kalosm::language::*;
#[tokio::main]
async fn main() {
    // Create a new model
    let model = Llama::new_chat().await.unwrap();
    // Create a new task that summarizes text
    let task = model.task("You take a long description and summarize it into a single short sentence");
    let mut output = task("You can define a Task with a description then run it with an input. The task will cache the description to repeated calls faster. Tasks work with chat models.");
    // Then stream the output to the console
    output.to_std_out().await.unwrap();
}
```

### Structured Generation

Structured generation gives you more control over the output of the text generation. You can derive a parser for your data to easily get structured data out of an LLM:
```rust, no_run
use kalosm::language::*;
#[derive(Parse, Clone)]
struct Pet {
    name: String,
    age: u32,
    description: String,
}
```

Then you can generate text that works with the parser in a [`Task`](prelude::Task):

```rust, no_run
# use kalosm::language::*;
# use std::sync::Arc;
#[derive(Parse, Debug, Clone)]
struct Pet {
    name: String,
    age: u32,
    description: String,
}

#[tokio::main]
async fn main() {
    // First create a model. Chat models tend to work best with structured generation
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
    let parser = Arc::new(Pet::new_parser());
    // Then create a task with the parser as constraints
    let task = model.task("You generate realistic JSON placeholders")
        .with_constraints(parser);
    // Finally, run the task
    let pet: Pet = task("Generate a pet in the form {\"name\": \"Pet name\", \"age\": 0, \"description\": \"Pet description\"}").await.unwrap();
    println!("{pet:?}");
}
```

## Embedding Models

[`Embedder`](prelude::Embedder) and [`EmbedderExt`](prelude::EmbedderExt) are the core traits for text embedding models. Any model that implements these traits can be used with Kalosm.


The simplest way to use an embedding model is to create a [bert model](prelude::Bert) and call [`embed`](prelude::EmbedderExt::embed) on it. The [`Embedding`](prelude::Embedding) you get back represents the meaning of the text in a numerical format:

```rust, no_run
use kalosm::language::*;
#[tokio::main]
async fn main() {
    // First create a model. Bert::new() is a good default embedding model for general tasks
    let model = Bert::new().await.unwrap();
    // Then embed some text into the vector space
    let embedding = model.embed("Kalosm is a library for building AI applications").await.unwrap();
    // And some more text
    let embedding = model.embed(prompt_input("Text: ").unwrap()).await.unwrap();
    // You can compare the cosine similarity of the two embeddings to see how similar they are
    println!("cosine similarity: {}", embedding.cosine_similarity(&embedding));
}
```

## Context

Gathering context is a key part of building LLM applications. Providing the right context to the model makes the output more relevant and useful. It can also help to prevent hallucinations.

Kalosm provides tools to generate gather, and process context from a variety of sources.

### Gathering context

Kalosm provides utilities for collecting context from a variety of sources:
- Local files (.txt, .md, .html, .docx, .pdf)
- RSS feeds
- Websites
- Search engines
- Microphone input and audio input through [whisper transcriptions](https://docs.rs/rwhisper/latest/rwhisper/struct.Whisper.html)

Each of these sources implements either [`IntoDocument`](prelude::IntoDocument) or [`IntoDocuments`](prelude::IntoDocuments) to convert the data into a [`Document`](prelude::Document) with the contents and metadata about the document.

```rust, no_run
use kalosm::language::*;
use std::convert::TryFrom;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try to extract an article from a URL
    let page = Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news")?;
    let document = page.into_document().await?;
    println!("Title: {}", document.title());
    println!("Body: {}", document.body());

    Ok(())
}
```

### Chunking context

After you have gathered context, it is often useful to chunk it into smaller pieces for search. Kalosm provides utilities for chunking context into documents, sentences, paragraphs, or semantic chunks. Kalosm will embed each chunk as it splits the document into smaller pieces. One of the most powerful chunker is the semantic chunker, which lets you chunk documents into semantically similar chunks without explicitly setting the size of the chunks:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First, create an embedding model for semantic chunking
    let model = Bert::new().await?;
    // Then create a document folder with some documents
    let documents = DocumentFolder::new("./documents")?.into_documents().await?;
    // Then chunk the documents into sentences
    let chunked = SemanticChunker::new().chunk_batch(&documents, &model).await?;
    println!("{:?}", chunked);
    Ok(())
}
```

### Embedding-powered search

After you have chunked your context, you can use the embeddings for search or retrieval augmented generation. Embedding-based search lets you find documents that are semantically similar to a specific word or phrase even if no words are an exact match:

```rust, no_run
use kalosm::language::*;
use surrealdb::{engine::local::SurrealKv, Surreal};

#[tokio::main]
async fn main() {
    // Create database connection
    let db = Surreal::new::<SurrealKv>(std::env::temp_dir().join("temp.db")).await.unwrap();

    // Select a specific namespace / database
    db.use_ns("search").use_db("documents").await.unwrap();

    // Create a table in the surreal database to store the embeddings
    let document_table = db
        .document_table_builder("documents")
        .build::<Document>()
        .await
        .unwrap();

    // Add documents to the database
    document_table.add_context(DocumentFolder::new("./documents").unwrap()).await.unwrap();

    loop {
        // Get the user's question
        let user_question = prompt_input("Query: ").unwrap();

        let nearest_5 = document_table
            .search(user_question)
            .with_results(5)
            .await
            .unwrap();

        println!("{:?}", nearest_5);
    }
}
```
