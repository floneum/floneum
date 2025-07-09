use kalosm::sound::*;
use rodio::Decoder;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // tracing_subscriber::fmt::init();

    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::QuantizedLargeV3Turbo)
        .build()
        .await?;

    // Load audio from a file
    let contents = std::fs::read("./media/audio_samples/sample_spanish.mp3").unwrap();
    let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();

    // Transcribe the source audio into text
    let mut text = model
        .transcribe(audio)
        .with_language(WhisperLanguage::Spanish);

    text.to_std_out().await?;
    Ok(())
}
