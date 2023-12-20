use kalosm::language::*;

#[tokio::main]
async fn main() {
    let nyt =
        RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());

    let mut fuzzy = FuzzySearchIndex::default();
    fuzzy.extend(nyt).await.unwrap();

    loop {
        let user_question = prompt_input("Query: ").unwrap();
        let context = fuzzy.search(&user_question, 5).await;

        let llm = Llama::new_chat();

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

        let stream = llm.stream_text(&prompt).with_max_length(300).await.unwrap();

        stream.to_std_out().await.unwrap();
    }
}
