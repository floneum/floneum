use comfy_table::{Cell, Color, Row, Table};
use kalosm::language::*;
use kalosm::*;
use surrealdb::{engine::local::RocksDb, Surreal};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let exists = std::path::Path::new("./db").exists();

    // Create database connection
    let db = Surreal::new::<RocksDb>("./db/temp.db").await.unwrap();

    // Select a specific namespace / database
    db.use_ns("test").use_db("test").await.unwrap();

    let chunker = SemanticChunker::new();

    let mut document_table = db
        .document_table_builder("documents")
        .with_embedding_model(
            Bert::builder()
                .with_source(BertSource::snowflake_arctic_embed_extra_small())
                .build()
                .await
                .unwrap(),
        )
        .with_chunker(chunker)
        .at("./db/embeddings.db")
        .build::<Document>()
        .await
        .unwrap();

    if !exists {
        let start_time = std::time::Instant::now();
        std::fs::create_dir_all("documents").unwrap();
        let context = [
            "https://floneum.com/kalosm/docs",
            "https://floneum.com/kalosm/docs/reference/web_scraping",
            "https://floneum.com/kalosm/docs/reference/transcription",
            "https://floneum.com/kalosm/docs/reference/image_segmentation",
            "https://floneum.com/kalosm/docs/reference/image_generation",
            "https://floneum.com/kalosm/docs/reference/llms",
            "https://floneum.com/kalosm/docs/reference/llms/structured_generation",
            "https://floneum.com/kalosm/docs/reference/llms/context",
            "https://floneum.com/kalosm/docs/guides/retrieval_augmented_generation",
        ]
        .iter()
        .map(|url| Url::parse(url).unwrap());

        // Create a new document database table
        document_table.add_context(context).await.unwrap();
        println!("Added context in {:?}", start_time.elapsed());
    }

    loop {
        let user_question = prompt_input("Query: ").unwrap();
        let user_question = format!(
            "Represent this sentence for searching relevant passages: {}",
            user_question
        );
        let user_question_embedding = document_table
            .embedding_model_mut()
            .embed(&user_question)
            .await
            .unwrap();

        let nearest_5 = document_table
            .select_nearest_embedding(user_question_embedding, 5)
            .await
            .unwrap();

        let mut table = Table::new();
        table.set_content_arrangement(comfy_table::ContentArrangement::DynamicFullWidth);
        table.load_preset(comfy_table::presets::UTF8_FULL);
        table.apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS);
        table.set_header(vec!["Score", "Value"]);

        for result in nearest_5 {
            let mut row = Row::new();
            let color = if result.distance < 0.25 {
                Color::Green
            } else if result.distance < 0.75 {
                Color::Yellow
            } else {
                Color::Red
            };
            row.add_cell(Cell::new(result.distance).fg(color))
                .add_cell(Cell::new(result.text()));
            table.add_row(row);
        }

        println!("{}", table);
    }
}
