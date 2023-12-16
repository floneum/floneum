# Kalosm

Kalosm is a simple interface for pre-trained models in rust. It makes it easy to interact with pre-trained, language, audio, and image models.

There are three different packages in Kalosm:
- `kalosm::language` - A simple interface for text generation and embedding models and surrounding tools. It includes support for search databases, and text collection from websites, RSS feeds, and search engines.
- `kalosm::sound` - A simple interface for audio transcription and surrounding tools. It includes support for microphone input and the `whisper` model.
- `kalosm::vision` - A simple interface for image generation and segmentation models and surrounding tools. It includes support for the `wuerstchen` and `segment-anything` models and integration with the [image](https://docs.rs/image/latest/image/) crate.

## Quickstart!

1) Install [rust](https://rustup.rs/)
2) Create a new project:
```sh
cargo new next-gen-ai
cd ./next-gen-ai
```
3) Add Kalosm as a dependency
```sh
cargo add kalosm --git "https://github.com/floneum/floneum"
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

## What can you build with Kalosm?

You can think of Kalosm as the plumbing between different pre-trained models and each other or the surrounding world. Kalosm makes it easy to build applications that use pre-trained models to generate text, audio, and images. Here are some examples of what you can build with Kalosm:

### Local text generation

The simplest way to use Kalosm is to pull in one of the local large language models and use it to generate text. Kalosm supports a streaming API that allows you to generate text in real time without blocking your main thread:

```rust, no_run
use std::io::Write;

use futures_util::stream::StreamExt;
use kalosm::language::*;
use kalosm::TextStream;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut llm = Phi::start().await;
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(1000).await.unwrap();

    let mut words = stream.words();
    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
```

### Cloud models

Kalosm also supports cloud models like GPT4 with the same streaming API:

```rust, no_run
// You must set the environment variable OPENAI_API_KEY (https://platform.openai.com/account/api-keys) to run this example.

use std::io::Write;
use futures_util::stream::StreamExt;
use kalosm::{language::*, TextStream};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut llm = Gpt4::start().await;
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(300).await.unwrap();

    let mut words = stream.words();
    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
```

### Collecting text data

Kalosm makes it easy to collect text data from a variety of sources. For example, you can use Kalosm to collect text from a local folder of documents, an RSS stream, a website, or a search engine:

```rust, no_run
use kalosm::language::*;
use std::convert::TryFrom;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read an RSS stream
    let nyt = RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());
    // Read a local folder of documents
    let mut documents = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();
    // Read a website (either from the raw HTML or inside of a headless browser)
    let page = Page::new(Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news").unwrap(), BrowserMode::Static).unwrap();
    let document = page.article().await.unwrap();
    println!("Title: {}", document.title());
    println!("Body: {}", document.body());
    // Read pages from a search engine (You must have the SERPER_API_KEY environment variable set to run this example)
    let query = "What is the capital of France?";
    let api_key = std::env::var("SERPER_API_KEY").unwrap();
    let search_query = SearchQuery::new(query, &api_key, 5);
    let documents = search_query.into_documents().await.unwrap();
    let mut text = String::new();
    for document in documents {
        for word in document.body().split(' ').take(300) {
            text.push_str(word);
            text.push(' ');
        }
        text.push('\n');
    }
    println!("{}", text);

    Ok(())
}
```

### Embedding powered search

Once you have your data, Kalosm includes tools to create traditional fuzzy search indexes and vector search indexes. These indexes can be used to search for specific text in a large corpus of documents. Fuzzy search indexes are useful for finding documents that contain a specific word or phrase. Vector search indexes are useful for finding documents that are semantically similar to a specific word or phrase. Kalosm makes it easy to create and use these indexes with your embedding model of choice:

```rust, no_run
use kalosm::language::*;
use std::io::Write;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let documents = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();

    let mut database = DocumentDatabase::new(
        Bert::builder().build().unwrap(),
        ChunkStrategy::Sentence {
            sentence_count: 1,
            overlap: 0,
        },
    );
    database.extend(documents.clone()).await.unwrap();
    let mut fuzzy = FuzzySearchIndex::default();
    fuzzy.extend(documents).await.unwrap();

    loop {
        print!("Query: ");
        std::io::stdout().flush().unwrap();
        let mut user_question = String::new();
        std::io::stdin().read_line(&mut user_question).unwrap();

        println!(
            "vector: {:?}",
            database
                .search(&user_question, 5)
                .await
                .iter()
                .collect::<Vec<_>>()
        );
        println!(
            "fuzzy: {:?}",
            fuzzy
                .search(&user_question, 5)
                .await
                .iter()
                .collect::<Vec<_>>()
        );
    }

    Ok(())
}
```

### Resource augmented generation

A large part of making modern LLMs performant is curating the context the models have access to. Resource augmented generation (or RAG) helps you do this by inserting context into the prompt based on a search query. For example, you can Kalosm to create a chatbot that uses context from local documents to answer questions:

```rust, no_run
use futures_util::StreamExt;
use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nyt =
        RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());

    let mut fuzzy = FuzzySearchIndex::default();
    fuzzy.extend(nyt).await.unwrap();

    loop {
        print!("Query: ");
        std::io::stdout().flush().unwrap();
        let mut user_question = String::new();
        std::io::stdin().read_line(&mut user_question).unwrap();
        let context = fuzzy.search(&user_question, 5).await;

        let mut llm = Llama::default();

        let context = context
            .iter()
            .take(2)
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "# Question:
    {user_question}
    # Context:
    {context}
    # Answer:
    "
        );

        let mut stream = llm.stream_text(&prompt).with_max_length(300).await.unwrap();

        while let Some(text) = stream.next().await {
            print!("{}", text);
            std::io::stdout().flush().unwrap();
        }
    }

    Ok(())
}
```

### Voice transcription

Kalosm makes it easy to build up context about the world around your application and use it to generate text. For example, you can use Kalosm to transcribe audio from a microphone, insert that into a vector database and answer questions about the audio in real-time:

```rust, no_run
use futures_util::StreamExt;
use kalosm::*;
use kalosm_language::*;
use kalosm_sound::*;
use std::sync::Arc;
use tokio::{
    sync::RwLock,
    time::{Duration, Instant},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::MediumEn)
        .build()?;

    let document_engine = Arc::new(RwLock::new(FuzzySearchIndex::default()));
    {
        let document_engine = document_engine.clone();
        std::thread::spawn(move || {
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async move {
                    let recording_time = Duration::from_secs(30);
                    loop {
                        let input = kalosm_sound::MicInput::default()
                            .record_until(Instant::now() + recording_time)
                            .await
                            .unwrap();

                        if let Ok(mut transcribed) = model.transcribe(input) {
                            while let Some(transcribed) = transcribed.next().await {
                                if transcribed.probability_of_no_speech() < 0.90 {
                                    let text = transcribed.text();
                                    document_engine.write().await.add(text).await.unwrap();
                                }
                            }
                        }
                    }
                })
        });
    }

    let mut model = Llama::new_chat();
    let mut chat = Chat::builder(&mut model).with_system_prompt("The assistant help answer questions based on the context given by the user. The model knows that the information the user gives it is always true.").build();

    loop {
        let user_question = prompt_input("\n> ").unwrap();

        let mut engine = document_engine.write().await;

        let context = {
            let context = engine.search(&user_question, 5).await;
            let context = context
                .iter()
                .take(5)
                .map(|x| x.to_string())
                .collect::<Vec<_>>();
            context.join("\n")
        };

        let prompt = format!(
            "Here is the relevant context:\n{context}\nGiven that context, answer the following question:\n{user_question}"
        );

        println!("{}", prompt);

        let output_stream = chat.add_message(prompt).await.unwrap();
        print!("Bot: ");
        output_stream.to_std_out().await.unwrap();
    }
}
```

### Image generation

In addition to language, audio, and embedding models, Kalosm also supports image generation. For example, you can use Kalosm to generate images from text:

```rust, no_run
use kalosm::vision::*;

let model = Wuerstchen::builder().build().unwrap();
let settings = WuerstchenInferenceSettings::new(
    "a cute cat with a hat in a room covered with fur with incredible detail",
)
.with_n_steps(2);
let images = model.run(settings).unwrap();
for (i, img) in images.iter().enumerate() {
    img.save(&format!("{}.png", i)).unwrap();
}
```

### Image segmentation

Kalosm also supports image segmentation with the segment-anything model:

```rust, no_run
use kalosm::vision::*;

let model = SegmentAnything::builder().build().unwrap();
let image = image::open("examples/landscape.jpg").unwrap();
let images = model.segment_everything(image).unwrap();
for (i, img) in images.iter().enumerate() {
    img.save(&format!("{}.png", i)).unwrap();
}
```
