use kalosm::language::*;
use kalosm::*;
use surrealdb::{engine::local::RocksDb, Surreal};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let nyt =
        RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());

    // Create database connection
    let db = Surreal::new::<RocksDb>("temp.db").await.unwrap();

    // Select a specific namespace / database
    db.use_ns("test").use_db("test").await.unwrap();

    // Create a new document database table
    let mut document_table = db.document_table_builder("nyt").build().unwrap();
    let documents = nyt.into_documents().await.unwrap();
    for document in documents {
        document_table.insert(document).await.unwrap();
    }

    loop {
        let user_question = prompt_input("Query: ").unwrap();
        let user_question_embedding = document_table
            .embedding_model_mut()
            .embed(&user_question)
            .await
            .unwrap();

         println!(
            "nearest: {:?}",
            document_table
                .select_nearest_embedding(user_question_embedding, 2)
                .await
                .iter()
                .collect::<Vec<_>>()
        );
    }
}
