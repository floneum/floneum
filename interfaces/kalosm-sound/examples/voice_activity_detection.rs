use futures_util::StreamExt;
use kalosm_sound::*;
use rodio::OutputStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mic_input = MicInput::default();
    let stream = mic_input.stream();
    let mut vad = stream.denoise_and_detect_voice_activity();
    let (_device, stream_handle) = OutputStream::try_default().unwrap();
    let sink = rodio::Sink::try_new(&stream_handle)?;
    while let Some(VoiceActivityDetectorOutput {
        probability,
        samples,
    }) = vad.next().await
    {
        println!("probability: {probability}");
        if probability > 0.5 {
            sink.append(samples);
        }
    }

    Ok(())
}
