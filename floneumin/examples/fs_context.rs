use floneumin_language::context::io::DocumentFolder;
use floneumin_language::index::vector::ChunkStrategy;
use floneumin_language::index::{keyword::FuzzySearchIndex, vector::DocumentDatabase, SearchIndex};
use floneumin_language::local::BertSpace;
use floneumin_language::local::LocalBert;
use std::io::Write;
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let documents = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();

    let mut database = DocumentDatabase::<BertSpace, LocalBert>::new(ChunkStrategy::Sentence {
        sentence_count: 1,
        overlap: 0,
    });
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
}
