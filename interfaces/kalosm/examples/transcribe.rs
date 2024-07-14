use kalosm::sound::*;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new whisper model.
    let model = Whisper::new().await?;

    // Stream audio from the microphone
    let mic = MicInput::default();
    let stream = mic.stream().unwrap();

    // Transcribe the audio into text in chunks based on voice activity.
    let text_stream = stream.transcribe(model);

    // Finally, print the text to the console
    text_stream.to_std_out().await.unwrap();

    Ok(())
}
