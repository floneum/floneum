use futures_util::StreamExt;
use kalosm_sound::*;
use rodio::OutputStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mic_input = MicInput::default();
    let stream = mic_input.stream()?;
    let mut vad = stream.voice_activity_stream();
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    while let Some(VoiceActivityDetectorOutput {
        probability,
        samples,
    }) = vad.next().await
    {
        println!("probability: {probability}");
        if probability > 0.5 {
            stream_handle.play_raw(samples).unwrap();
        }
    }

    Ok(())
}
