use futures_util::StreamExt;
use rwhisper::*;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::SmallEn)
        .build()?;

    let mut text = kalosm_sound::MicInput::default()
        .stream()
        .unwrap()
        .subscribe_stream(Duration::from_secs(30))
        .text(model);

    while let Some(transcribed) = text.next().await {
        let text = transcribed.text();
        print!("{}", text);
    }

    Ok(())
}
