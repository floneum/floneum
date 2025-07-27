# Chat Session

The [`ChatSession`] trait holds the state of a text completion model after it has been fed some text. It can be used in combination with [`ChatModel`] to feed text and cache the results.

## Saving and Loading Sessions

Sessions can be serialized and deserialized to and from bytes using the [`ChatSession::to_bytes`] and [`ChatSession::from_bytes`] methods. This can be useful for saving and loading sessions to disk. Caching a session avoids re-processing the text again when the session is resumed.

```rust, no_run
use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Llama::new_chat().await.unwrap();
    let mut chat = llm.chat();

    // Feed some text into the session
    chat(&"What is the capital of France?").to_std_out().await.unwrap();

    // Save the session to bytes
    let session = chat.session().unwrap();
    let session_as_bytes = session.to_bytes().unwrap();
    
    // Load the session from bytes
    let mut session = LlamaChatSession::from_bytes(&session_as_bytes).unwrap();
    let mut chat = llm.chat().with_session(session);

    // Feed some more text into the session
    chat(&"What was my first question?").to_std_out().await.unwrap();
}
```

## Session History

You can use the [`ChatSession::history`] method to get messages that have already been fed to the model:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
let mut llm = Llama::new_chat().await.unwrap();
let mut chat = llm.chat();
// Add a message to the session
chat(&"Hello, world!").to_std_out().await.unwrap();
// Get the history of the session
let history = chat.session().unwrap().history();
assert_eq!(history.len(), 1);
assert_eq!(history[0].role(), MessageType::UserMessage);
assert_eq!(history[0].content(), "Hello, world!");
# }
```

## Cloning Sessions

Not all chat models support cloning sessions, but if a model does support
cloning sessions, you can clone a session using the [`ChatSession::try_clone`] method
to clone a session state while retaining the original session.

```rust, no_run
use kalosm::language::*;
use std::io::Write;
#[tokio::main]
async fn main() {
    let mut llm = Llama::new_chat().await.unwrap();
    let mut chat = llm.chat();
    // Feed some text into the session
    chat(&"What is the capital of France?").to_std_out().await.unwrap();
    let mut session = chat.session().unwrap();
    // Clone the session
    let cloned_session = session.try_clone().unwrap();
    // Feed some more text into the cloned session
    let mut chat = llm.chat().with_session(cloned_session);
    chat(&"What was my first question?").to_std_out().await.unwrap();
}
```