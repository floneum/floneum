# Tasks

Any model that implements [`ChatModel`] or [`StructuredChatModel`] can be used with tasks to repeatedly perform work with the same system prompt.

You can create a task with the [`ChatModelExt::task`] method with a description of the task and then call the task like a function to start generating a response:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let task = model.task("You are an editing assistant who offers suggestions for improving the quality of the text. You will be given some text and will respond with a list of suggestions for how to improve the text.");
    let mut stream = task(&"this isnt correct. or is it?");
    stream.to_std_out().await.unwrap();
}
```

Once you have the response builder, you can modify it with any of the methods on [`ChatResponseBuilder`]. For example, you can change the sampler with [`ChatResponseBuilder::with_sampler`]:
```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let task = model.task("You are an editing assistant who offers suggestions for improving the quality of the text. You will be given some text and will respond with a list of suggestions for how to improve the text.");
    let mut stream = task(&"this isnt correct. or is it?").with_sampler(GenerationParameters::default());
    stream.to_std_out().await.unwrap();
}
```

## Structured Generation

You can use structured generation to force the output of the task to fit a specific format. Before you add structured generation to the tasks, you need to define a parser.

### Defining the parser

There are a few different ways create a parser for structured generation:

1. Derive a parser for your data
2. Create a parser from the set of prebuilt combinators
3. Create a parser from a regex

#### Deriving a parser from a struct

The simplest way to get started is to derive a parser for your data:

```rust, no_run
# use kalosm::language::*;
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
# use std::sync::Arc;
# #[derive(Parse, Clone, Debug)]
# struct Pet {
#     name: String,
#     age: u32,
#     description: String,
# }
#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
    let parser = Pet::new_parser();
    // Create a task with the constraints
    let task = model.task("You generate realistic JSON placeholders for pets in the form {\"name\": \"Pet name\", \"age\": 0, \"description\": \"Pet description\"}")
        // The task constraints must be clone. If they don't implement Clone, you can wrap them in an Arc
        .with_constraints(Arc::new(parser));
    // Then run the task
    let pet: Pet = task(&"Ruffles is a 3 year old adorable dog").await.unwrap();
    println!("{pet:?}");
}
```

#### Creating a Parser from the Set of Prebuilt Combinators

Kalosm also provides a set of prebuilt combinators for creating more complex parsers. You can use these combinators to create a parser with a custom format:

```rust, no_run
use kalosm::language::*;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your custom format
    let parser = LiteralParser::from("[")
        .ignore_output_then(String::new_parser())
        .then_literal(", ")
        .then(u8::new_parser())
        .then_literal(", ")
        .then(String::new_parser())
        .then_literal("]");
    // Create a task with the constraints
    let task = model.task("You generate realistic JSON placeholders for pets in the form [\"Pet name\", age number, \"Pet description\"]")
        // The task constraints must be clone. If they don't implement Clone, you can wrap them in an Arc
        .with_constraints(Arc::new(parser));
    // Then run the task
    let ((name, age), description) = task(&"Ruffles is a 3 year old adorable dog").await.unwrap();
    println!("{name} {age} {description}");
}
```

#### Creating a Parser from a Regex

You can also create a parser from a regex:

```rust, no_run
use kalosm::language::*;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your data. Any
    let parser = RegexParser::new(r"\[(\w+), (\d+), (\w+)\]").unwrap();
    // Create a task with the constraints
    let task = model.task("You generate realistic JSON placeholders for pets in the form [\"Pet name\", age number, \"Pet description\"]")
        // The task constraints must be clone. If they don't implement Clone, you can wrap them in an Arc
        .with_constraints(Arc::new(parser));
    // Finally, run the task. Unlike derived and custom parsers, regex parsers do not provide a useful output type
    task(&"Ruffles is a 3 year old adorable dog").to_std_out().await.unwrap();
}
```

### Tasks with Constraints

Once you have a parser, you can force the model to generate text that conforms to that parser with the [`Task::with_constraints`]:

```rust, no_run
use kalosm::language::*;
use std::sync::Arc;

#[derive(Parse, Clone, Debug)]
struct Pet {
    name: String,
    age: u32,
    description: String,
}

#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new_chat().await.unwrap();
    // Then create a parser for your data.
    // Any type that implements the `Parse` trait has the `new_parser` method
    let parser = Pet::new_parser();
    // Create a task with the constraints
    let task = model.task("You generate realistic JSON placeholders for pets in the form {\"name\": \"Pet name\", \"age\": 0, \"description\": \"Pet description\"}")
            // The task constraints must be clone. If they don't implement Clone, you can wrap them in an Arc
        .with_constraints(Arc::new(parser));
    // Then run the task
    let pet: Pet = task(&"Ruffles is a 3 year old adorable dog").await.unwrap();
    println!("{pet:?}");
}
```
