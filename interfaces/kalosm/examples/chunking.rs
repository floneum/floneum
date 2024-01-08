use kalosm::language::*;
use kalosm_language::search::{Chunker, Hypothetical};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let documents = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();
    let documents = documents.into_documents().await.unwrap()[..5].to_vec();

    let mut llm = Phi::v2().unwrap();

    let hypothetical = Hypothetical::new(&mut llm).with_chunking(
        kalosm_language::search::ChunkStrategy::Sentence {
            sentence_count: 3,
            overlap: 0,
        },
    );

    let mut embedder = Bert::default();
    let chunked = hypothetical
        .chunk_batch(&documents, &mut embedder)
        .await
        .unwrap();
    println!("chunked: {:?}", chunked);
}
