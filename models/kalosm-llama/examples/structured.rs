#![allow(unused)]
use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let llm = Llama::phi_3().await.unwrap();
    let prompt = "Generate a list of 4 pets in JSON form with a name, description, color, and diet";

    #[derive(Debug, Clone, Parse)]
    struct Pet {
        name: String,
        description: String,
        color: String,
        size: Size,
        diet: Diet,
    }

    #[derive(Debug, Clone, Parse)]
    enum Diet {
        #[parse(rename = "carnivore")]
        Carnivore,
        #[parse(rename = "herbivore")]
        Herbivore,
        #[parse(rename = "omnivore")]
        Omnivore,
    }

    #[derive(Debug, Clone, Parse)]
    enum Size {
        #[parse(rename = "small")]
        Small,
        #[parse(rename = "medium")]
        Medium,
        #[parse(rename = "large")]
        Large,
    }

    println!("# with constraints");

    let task = Task::builder("You generate realistic JSON placeholders")
        .with_constraints(<[Pet; 4] as Parse>::new_parser())
        .build();
    let stream = task.run(prompt, &llm);

    time_stream(stream).await;

    println!("\n\n# without constraints");

    let task = Task::builder("You generate realistic JSON placeholders").build();
    let stream = task.run(prompt, &llm);

    time_stream(stream).await;
}

async fn time_stream(mut stream: impl TextStream + Unpin) {
    let start_time = std::time::Instant::now();
    let mut tokens = 0;
    let mut string_length = 0;
    while let Some(token) = stream.next().await {
        tokens += 1;
        string_length += token.len();
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
    let elapsed = start_time.elapsed();
    println!(
        "\n\nGenerated {} tokens ({} characters) in {:?}",
        tokens, string_length, elapsed
    );
    println!(
        "Tokens per second: {:.2}",
        tokens as f64 / elapsed.as_secs_f64()
    );
}
