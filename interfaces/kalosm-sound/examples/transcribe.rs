use kalosm_sound::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new small whisper model.
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::DistilLargeV3)
        .build()
        .await?;
    let mic = MicInput::default();
    let stream = mic.stream().unwrap();
    let vad = stream.voice_activity_stream().rechunk_voice_activity();
    let text_stream = vad.text(model);

    text_stream.to_std_out().await.unwrap();

    Ok(())
}
