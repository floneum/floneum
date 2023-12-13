// You must set the environment variable OPENAI_API_KEY (https://platform.openai.com/account/api-keys) to run this example.

use std::io::Write;

use futures_util::stream::StreamExt;
use kalosm_language::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let nyt =
        RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());

    let mut vector = DocumentDatabase::new(
        AdaEmbedder::default(),
        ChunkStrategy::Sentence {
            sentence_count: 3,
            overlap: 0,
        },
    );
    vector.extend(nyt).await.unwrap();

    loop {
        print!("Query: ");
        std::io::stdout().flush().unwrap();
        let mut user_question = String::new();
        std::io::stdin().read_line(&mut user_question).unwrap();
        let context = vector.search(&user_question, 5).await;

        let llm = LocalSession::<LlamaSevenChatSpace>::start().await;

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
}
