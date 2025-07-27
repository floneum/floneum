# Text Completion Session

The [`TextCompletionSession`] trait holds the state of a text completion model after it has been fed some text. It can be used in combination with [`TextCompletionModel`] to feed text and cache the results.

## Saving and Loading Sessions

Sessions can be serialized and deserialized to and from bytes using the [`TextCompletionSession::to_bytes`] and [`TextCompletionSession::from_bytes`] methods. This can be useful for saving and loading sessions to disk. Caching a session avoids re-processing the text again when the session is resumed.

```rust, no_run
use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let mut session = llm.new_session().unwrap();

    // Feed some text into the session
    llm.stream_text_with_callback(&mut session, "The capital of France is ".into(), GenerationParameters::new().with_max_length(0), |_| Ok(())).await.unwrap();

    // Save the session to bytes
    let session_as_bytes = session.to_bytes().unwrap();
    
    // Load the session from bytes
    let mut session = LlamaSession::from_bytes(&session_as_bytes).unwrap();

    // Feed some more text into the session
    llm.stream_text_with_callback(&mut session, "The capital of France is ".into(), GenerationParameters::new(), |token| {println!("{token}"); Ok(())}).await.unwrap();
}
```

## Cloning Sessions

Not all models support cloning sessions, but if a model does support cloning sessions, you can clone a session using the [`TextCompletionSession::try_clone`] method to clone a session state while retaining the original session.

```rust, no_run
use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let mut session = llm.new_session().unwrap();

    // Feed some text into the session
    llm.stream_text_with_callback(&mut session, "The capital of France is ".into(), GenerationParameters::new().with_max_length(0), |_| Ok(())).await.unwrap();

    // Clone the session
    let cloned_session = session.try_clone().unwrap();

    // Feed some more text into the cloned session
    llm.stream_text_with_callback(&mut session, "The capital of France is ".into(), GenerationParameters::new(), |token| {println!("{token}"); Ok(())}).await.unwrap();
}
```
