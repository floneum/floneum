use std::time::Duration;

use floneumin_sound::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let input = floneumin_sound::MicInput::default();
    let stream = input.stream()?;
    let model = Whisper::builder()
        .with_source(WhisperSource::SmallEn)
        .build()?;
    let mut transcribed = stream.subscribe_stream(Duration::from_secs(30)).text(model);

    let mut current_time_stamp = 0.0;

    println!("starting transcription loop");

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

    Ok(())
}
