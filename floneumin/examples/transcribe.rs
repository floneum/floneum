use floneumin_sound::model::whisper::*;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut model = WhisperBuilder::default()
        .model(WhichModel::LargeV2)
        .build()?;

    let (tx, mut rx) = tokio::sync::mpsc::channel(5);
    std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let recording_time = Duration::from_secs(10);
                loop {
                    let input = floneumin_sound::source::mic::MicInput::default()
                        .record_until(Instant::now() + recording_time)
                        .await
                        .unwrap();
                    tx.send(input).await.unwrap();
                }
            })
    });

    let mut current_time_stamp = 0.0;

    println!("starting transcription loop");
    loop {
        let mut next_time_stamp = current_time_stamp;
        let input = rx.recv().await.unwrap();

        let transcribed = model.transcribe(input)?;

        for transcribed in transcribed {
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
