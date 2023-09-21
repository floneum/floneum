use floneumin_language::{
    context::{rss::RssFeed, Url},
    index::{keyword::FuzzySearchIndex, vector::DocumentDatabase, SearchIndex},
    local::LocalSession,
    model::LlamaSevenChatSpace,
};
use std::io::Write;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let nyt =
        RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());

    let mut database =
        DocumentDatabase::<LlamaSevenChatSpace, LocalSession<LlamaSevenChatSpace>>::new();
    database.extend(nyt.clone()).await.unwrap();
    let mut fuzzy = FuzzySearchIndex::default();
    fuzzy.extend(nyt).await.unwrap();

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
