use std::path::PathBuf;

use futures_util::StreamExt;
use kalosm_sound::*;
use rodio::OutputStream;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let file = PathBuf::from("examples/samples_jfk.wav");
    let decoder = rodio::Decoder::new(std::fs::File::open(file)?)?;
    let mut vad = decoder.denoise_and_detect_voice_activity();
    let (_device, stream_handle) = OutputStream::try_default().unwrap();
    let sink = rodio::Sink::try_new(&stream_handle)?;
    while let Some(VoiceActivityDetectorOutput {
        probability,
        samples,
    }) = vad.next().await
    {
        if probability > 0.1 {
            sink.append(samples);
        }
    }

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    Ok(())
}
