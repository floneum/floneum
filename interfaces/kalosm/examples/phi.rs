use kalosm_language::*;

#[tokio::main]
async fn main() {
    Phi::start()
        .await
        .stream_text("The capital of Paris is ")
        .with_max_length(1000)
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
}
