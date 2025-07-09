use kalosm::sound::*;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Record audio from the microphone for 10 seconds.
    let audio = MicInput::default()
        .record_until(Instant::now() + Duration::from_secs(10))
        .await;

    // Create a new small whisper model.
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::QuantizedLargeV3Turbo)
        .build()
        .await?;

    // Transcribe the audio.
    let mut text = model.transcribe(audio).with_language(WhisperLanguage::Hindi);

    // As the model transcribes the audio, print the text to the console.
    text.to_std_out().await?;

    Ok(())
}
