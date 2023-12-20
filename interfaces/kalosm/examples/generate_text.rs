
use kalosm::language::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    let  stream = Llama::default()
        .stream_text(prompt)
        .with_max_length(300)
        .await
        .unwrap();

    print!("{}", prompt);

    stream.to_std_out().await.unwrap();
}
