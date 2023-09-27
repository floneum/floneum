use std::time::Duration;

use floneumin_sound::model::whisper::*;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut model = WhisperBuilder::default()
        .model(WhichModel::TinyEn)
        .build()?;

    let input = floneumin_sound::source::mic::MicInput::default();
    let stream = input.stream()?;
    let mut chunks = stream.stream().subscribe_stream(Duration::from_secs(30));

    let mut current_time_stamp = 0.0;

    println!("starting transcription loop");
    loop {
        let mut next_time_stamp = current_time_stamp;
        let input = chunks
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("no more chunks"))?;

        let mut transcribed = model.transcribe(input).await?;

        while let Some(transcribed) = transcribed.next().await {
            if transcribed.probability_of_no_speech() < 0.90 {
                let start = current_time_stamp + transcribed.start();
                let end = start + transcribed.duration();
                next_time_stamp = end;

                let text = transcribed.text();
                println!("({:01} - {:01}): {}", start, end, text);
            }
        }

        current_time_stamp = next_time_stamp;
    }
}
