use kalosm_language::*;

#[tokio::main]
async fn main() {
    Phi::v2()
        .unwrap()
        .stream_text("```py\ndef factorial")
        .with_max_length(1000)
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
}
