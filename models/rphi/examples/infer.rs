use rphi::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let model = Phi::default();
    let prompt = "The capital of France is";
    let mut result = model.stream_text(prompt).await.unwrap();

    print!("{prompt}");
    while let Some(token) = result.next().await {
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
}
