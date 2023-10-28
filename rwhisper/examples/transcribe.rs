use futures_util::StreamExt;
use rwhisper::*;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new small whisper model.
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::SmallEn)
        .build()?;

    // Record audio from the microphone for 5 seconds.
    let audio = kalosm_sound::MicInput::default()
        .record_until(Instant::now() + Duration::from_secs(5))
        .await?;

    // Transcribe the audio.
    let mut text = model.transcribe(audio)?;

    // As the model transcribes the audio, print the text to the console.
    while let Some(text) = text.next().await {
        print!("{}", text.text());
    }

    Ok(())
}
