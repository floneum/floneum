use floneumin_language::{
    CreateModel, FuzzySearchIndex, GenerationParameters, LlamaSevenChatSpace, LocalSession, Model,
    SearchIndex,
};
use floneumin_sound::model::whisper::*;
use futures_util::StreamExt;
use std::{
    io::Write,
    sync::{Arc, RwLock},
};
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = WhisperBuilder::default()
        .model(WhisperModelSource::SmallEn)
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
                        let input = floneumin_sound::source::mic::MicInput::default()
                            .record_until(Instant::now() + recording_time)
                            .await
                            .unwrap();

                        if let Ok(mut transcribed) = model.transcribe(input) {
                            while let Some(transcribed) = transcribed.next().await {
                                if transcribed.probability_of_no_speech() < 0.90 {
                                    let text = transcribed.text();
                                    println!("Adding to context: {}", text);
                                    document_engine.write().unwrap().add(text).await.unwrap();
                                }
                            }
                        }
                    }
                })
        });
    }

    loop {
        println!();
        print!("Query: ");
        std::io::stdout().flush().unwrap();
        let mut user_question = String::new();
        std::io::stdin().read_line(&mut user_question).unwrap();
        let engine = document_engine.read().unwrap();

        let mut llm = LocalSession::<LlamaSevenChatSpace>::start().await;

        let context = {
            let context = engine.search(&user_question, 5).await;
            context
                .iter()
                .take(5)
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        };

        let prompt = format!(
            "# Question:
    {user_question}
    # Context:
    {context}
    # Answer:
    "
        );

        println!("{}", prompt);

        let mut stream = llm
            .stream_text(
                &prompt,
                GenerationParameters::default().with_max_length(300),
            )
            .await
            .unwrap();

        loop {
            // set up a CTRL-C handler to stop the stream
            let quit_stream = tokio::signal::ctrl_c();
            tokio::select! {
                text = stream.next() => {
                    match text{
                        Some(text) => {
                            print!("{}", text);
                            std::io::stdout().flush().unwrap();
                        },
                        None => {
                            break;
                        }
                    }
                },
                _ = quit_stream => {
                    println!("Stopping stream...");
                    break;
                }
            }
        }
    }
}
