use futures_util::StreamExt;
use kalosm::audio::*;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new small whisper model.
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::DistilLargeV2)
        .build()?;

    loop {
        // Record audio from the microphone for 5 seconds.
        let audio = kalosm_sound::MicInput::default()
            .record_until(Instant::now() + Duration::from_secs(5))
            .await?;

        // Transcribe the audio.
        let mut transcribed = model.transcribe(audio)?;
        let mut current_time_stamp = 0.0;

        // As the model transcribes the audio, print the text to the console.
        while let Some(transcribed) = transcribed.next().await {
            let start = current_time_stamp + transcribed.start();
            let end = start + transcribed.duration();
            if transcribed.probability_of_no_speech() < 0.90 {
                let text = transcribed.text();
                println!("({:01} - {:01}): {}", start, end, text);
            } else {
                println!(
                    "({:01} - {:01}): <no speech> ({})",
                    start,
                    end,
                    transcribed.text()
                );
            }
            current_time_stamp = end;
        }
    }
}
