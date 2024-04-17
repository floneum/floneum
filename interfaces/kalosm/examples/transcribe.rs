use futures_util::StreamExt;
use kalosm::audio::*;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new small whisper model.
    let model = Whisper::new().await?;

    let mut current_time_stamp = 0.0;
    // Record audio and add them to a queue in the background
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                loop {
                    let audio = kalosm_sound::MicInput::default()
                        .record_until(Instant::now() + Duration::from_secs(5))
                        .await
                        .unwrap();
                    let _ = tx.send(audio);
                }
            });
    });

    loop {
        let Some(audio) = rx.recv().await else {
            break Ok(());
        };

        // Transcribe the audio.
        let mut transcribed = model.transcribe(audio)?;

        // As the model transcribes the audio, print the text to the console.
        while let Some(transcribed) = transcribed.next().await {
            let start = current_time_stamp + transcribed.start();
            let end = start + transcribed.duration();
            if transcribed.probability_of_no_speech() < 0.10 {
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
