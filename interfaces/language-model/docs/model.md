
## Text Generation Models

[`Model`] and [`ModelExt`] are the core traits for text generation models. Any model that implements these traits can be used with Kalosm.


The simplest way to use a model is to create a [`Model`] and call [`ModelExt::stream_text`] to stream text from it:

```rust, no_run
use kalosm::language::*;
#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");
    let stream = llm
        // Any model that implements the Model trait can be used to stream text
        .stream_text(prompt)
        // You can pass parameters to the model to control the output
        .with_max_length(300)
        // And run .await to start streaming
        .await
        .unwrap();
    // You can then use the stream however you need. to_std_out will print the text to the console as it is generated
    stream.to_std_out().await.unwrap();
}
```

### Tasks

You can define a Task with a description then run it with an input. The task will cache the description to repeated calls faster. Tasks work with both chat and non-chat models, but they tend to perform significantly better with chat models.

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
// Create a new task that 
let task = Task::new("You take a long description and summarize it into a single short sentence");
let output = task.run("You can define a Task with a description then run it with an input. The task will cache the description to repeated calls faster. Tasks work with both chat and non-chat models, but they tend to perform significantly better with chat models.");
// Then stream the output to the console
output.to_std_out().await.unwrap();
# }
```

### Structured Generation

Structured generation gives you more control over the output of the text generation. You have a few different ways to use structured generation:
1) Derive a parser for your data
2) Create a parser from the set of prebuilt combinators
3) Create a parser from a regex

The simplest way to get started is to derive a parser for your data:
```rust, no_run
use kalosm::language::*;
#[derive(Parse, Clone)]
struct Pet {
    name: String,
    age: u32,
    description: String,
}
```

Then you can generate text that works with the parser in a [`Task`](https://docs.rs/kalosm/latest/kalosm/language/struct.Task.html):

```rust, no_run
# use kalosm::language::*;
# #[derive(Parse, Clone, Debug)]
# struct Pet {
#     name: String,
#     age: u32,
#     description: String,
# }
#[tokio::main]
async fn main() {
    // First create a model. Chat models tend to work best with structured generation
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
    let parser = Pet::new_parser();
    // Then create a task with the parser as constraints
    let task = Task::builder("You generate realistic JSON placeholders")
        .with_constraints(parser)
        .build();
    // Finally, run the task
    let pet: Pet = task.run("Generate a pet in the form {\"name\": \"Pet name\", \"age\": 0, \"description\": \"Pet description\"}", &model).await.unwrap();
    println!("{pet:?}");
}
```

#### Creating a Parser from the Set of Prebuilt Combinators

Kalosm also provides a set of prebuilt combinators for creating more complex parsers. You can use these combinators to create a parser with a custom format:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    // First create a model. Chat models tend to work best with structured generation
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your custom format
    let parser = LiteralParser::from("[")
        .then(String::new_parser())
        .then_literal(", ")
        .then(u8::new_parser())
        .then_literal(", ")
        .then(String::new_parser())
        .then_literal("]");
    // Then create a task with the parser as constraints
    let task = Task::builder("You generate realistic JSON placeholders")
        .with_constraints(parser)
        .build();
    // Finally, run the task
    let ((name, age), description) = task.run("Generate a pet in the form [\"Pet name\", age number, \"Pet description\"]").await.unwrap();
    println!("{name} {age} {description}");
}
```

#### Creating a Parser from a Regex

You can also create a parser from a regex:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    // First create a model. Chat models tend to work best with structured generation
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your data. Any 
    let parser = RegexParser::new(r"\[(\w+), (\d+), (\w+)\]").unwrap();
    // Then create a task with the parser as constraints
    let task = Task::builder("You generate realistic JSON placeholders")
        .with_constraints(parser)
        .build();
    // Finally, run the task. Unlike derived and custom parsers, regex parsers do not provide a useful output type
    task.run("Generate a pet in the form [\"Pet name\", age number, \"Pet description\"]").to_std_out().await.unwrap();
}
```
