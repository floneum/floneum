use futures_util::StreamExt;
use kalosm_language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
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
