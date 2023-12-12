use axum::{
    body::Body,
    extract::{Path, State},
    response::IntoResponse,
    routing::get,
    Router,
};
use kalosm::language::*;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    println!("Downloading and starting model...");
    let model = Llama::builder()
        .with_source(LlamaSource::mistral_7b())
        .build()
        .unwrap();
    println!("Model ready");
    let app = Router::new()
        .route("/:prompt", get(stream_response))
        .with_state(Arc::new(RwLock::new(model)));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn stream_response(
    Path(prompt): Path<String>,
    State(model): State<Arc<RwLock<Llama>>>,
) -> impl IntoResponse {
    println!("Responding to {prompt}");
    let model_stream = model.write().await.stream_text(&prompt).await.unwrap();
    println!("stream ready");
    fn infallible(t: String) -> Result<String, std::convert::Infallible> {
        Ok(t)
    }
    // Stream the html to the client
    // First add the head
    let head = format!("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>kalosm</title></head><body><pre>{prompt} ");
    let head = infallible(head);
    let head = futures_util::stream::once(async { head });
    // Then the body
    let body = model_stream.map(infallible);
    // Then the tail
    let tail = "</pre></body></html>";
    let tail = infallible(tail.to_string());
    let tail = futures_util::stream::once(async { tail });
    // And return the stream
    Body::from_stream(head.chain(body).chain(tail))
}
