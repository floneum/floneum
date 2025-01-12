Let's start with a simple chat application:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
// Before you create a chat session, you need a model. Llama::new_chat will create a good default chat model.
let model = Llama::new_chat().await.unwrap();
// Then you can build a chat session that uses that model
let mut chat = model.chat()
    // The builder exposes methods for settings like the system prompt and constraints the bot response must follow
    .with_system_prompt("The assistant will act like a pirate");

loop {
    // To use the chat session, you need to add messages to it
    let mut response_stream = chat.add_message(prompt_input("\n> ").unwrap());
    // And then display the response stream to the user
    response_stream.to_std_out().await.unwrap();
}
# }
```

If you run the application, you may notice that it takes more time for the assistant to start responding to long prompts.
The LLM needs to read and transform the prompt into a format it understands before it can start generating a response.
Kalosm stores that state in a chat session, which can be saved and loaded from the filesystem to make loading existing chat sessions faster.

You can save and load chat sessions from the filesystem using the [`ChatSession::to_bytes`] and [`ChatBuilder::from_bytes`] methods:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
// First, create a model to chat with
let model = Llama::new_chat().await.unwrap();
// Then try to load the chat session from the filesystem
let save_path = std::path::PathBuf::from("./chat.llama");
let mut chat = model.chat();
if let Some(old_session) = std::fs::read(&save_path)
    .ok()
    .and_then(|bytes| LlamaChatSession::from_bytes(&bytes).ok())
{
    chat = chat.with_session(old_session);
}

// Then you can add messages to the chat session as usual
let mut response_stream = chat.add_message(prompt_input("\n> ").unwrap());
// And then display the response stream to the user
response_stream.to_std_out().await.unwrap();

// After you are done, you can save the chat session to the filesystem
let session = chat.session().unwrap();
let bytes = session.to_bytes().unwrap();
std::fs::write(&save_path, bytes).unwrap();
# }
```

LLMs are powerful because of their generality, but sometimes you need more control over the output. For example, you might want the assistant to start with a certain phrase, or to follow a certain format.

In kalosm, you can use constraints to guide the model's response. Constraints are a way to specify the format of the output. When generating with constraints, the model will always respond with the specified format.

Let's create a chat application that uses constraints to guide the assistant's response to always start with "Yes!":

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
let model = Llama::new_chat().await.unwrap();
// Create constraints that parses Yes! and then stops on the end of the assistant's response
let constraints = LiteralParser::new("Yes!")
    .then(model.default_assistant_constraints());
// Create a chat session with the model and the constraints
let mut chat = model.chat();

// Chat with the user
loop {
    let mut output_stream = chat.add_message(prompt_input("\n> ").unwrap()).with_constraints(constraints.clone());
    output_stream.to_std_out().await.unwrap();
}
# }
```
