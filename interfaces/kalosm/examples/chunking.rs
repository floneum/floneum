use kalosm::language::*;
use kalosm_language::search::{Chunker, Hypothetical};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let nyt =
        RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());

    let documents = nyt.into_documents().await.unwrap();
    println!("documents: {:?}", documents);

    let mut chat = Llama::new_chat();

    let hypothetical = Hypothetical::new(&mut chat);

    let mut embedder = Bert::default();
    let chunked = hypothetical
        .chunk_batch(&documents, &mut embedder)
        .await
        .unwrap();
    println!("chunked: {:?}", chunked);
}
