use rphi::prelude::*;

#[tokio::main]
async fn main() {
    let mut model = Phi::default();
    let prompt = "The capital of France is ";
    let mut result = model.stream_text(prompt).await.unwrap();

    print!("{prompt}");
    while let Some(token) = result.next().await {
        print!("{token}");
    }
}
