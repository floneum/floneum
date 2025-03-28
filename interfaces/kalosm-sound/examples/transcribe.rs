use kalosm_sound::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new whisper model.
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::DistilLargeV3)
        .build()
        .await?;

    // Stream audio from the microphone
    let mic = MicInput::default();
    let stream = mic.stream();

    // Chunk that audio into chunks based on voice activity
    let vad = stream.denoise_and_detect_voice_activity().rechunk_voice_activity();

    // And then transcribe the audio into text
    let mut text_stream = vad.transcribe(model);

    // Finally, print the text to the console
    text_stream.to_std_out().await.unwrap();

    Ok(())
}
