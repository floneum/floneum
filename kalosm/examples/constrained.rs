use futures_util::stream::StreamExt;
use kalosm_language::*;
use kalosm_sample::*;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "Five US states in central US are ";

    println!("# with constraints");
    print!("{}", prompt);
    let states = [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
    ];
    let states_parser = states
        .into_iter()
        .map(LiteralParser::from)
        .collect::<Vec<_>>();

    let states = IndexParser::new(states_parser);

    let validator = states
        .then(LiteralParser::from(", "))
        .repeat(5..=5)
        .then(LiteralParser::from("\n"));
    let validator_state = validator.create_parser_state();
    let mut words = llm
        .stream_structured_text_with_sampler(
            prompt,
            validator,
            validator_state,
            Arc::new(Mutex::new(
                GenerationParameters::default().bias_only_sampler(),
            )),
            Arc::new(Mutex::new(
                GenerationParameters::default().mirostat2_sampler(),
            )),
        )
        .await
        .unwrap();

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }

    println!("{:#?}", words.result().await);

    println!("\n\n# without constraints");
    print!("{}", prompt);

    let mut words = llm.stream_text(prompt).with_max_length(100).await.unwrap();
    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
