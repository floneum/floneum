use denoised::DenoisedExt;
use futures_util::StreamExt;
use kalosm_sound::*;
use rodio::OutputStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mic_input = MicInput::default();
    let stream = mic_input.stream()?;
    let mut vad = stream.denoise();
    let (_device, stream_handle) = OutputStream::try_default().unwrap();
    let sink = rodio::Sink::try_new(&stream_handle)?;
    while let Some(VoiceActivityDetectorOutput {
        probability,
        samples,
    }) = vad.next().await
    {
        println!("probability: {probability}");
        sink.append(samples);
    }

    Ok(())
}
