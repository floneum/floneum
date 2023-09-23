use floneumin_sound::model::whisper::*;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut model = WhisperBuilder::default()
        .model(WhichModel::SmallEn)
        .build()?;

    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let recording_time = Duration::from_secs(20);
                loop {
                    let input = floneumin_sound::source::mic::MicInput::default()
                        .record_until(Instant::now() + recording_time)
                        .await
                        .unwrap();
                    tx.send(input).await.unwrap();
                }
            })
    });

    loop {
        println!("recording");
        let input = rx.recv().await.unwrap();

        println!("transcribing");

        let start_time = Instant::now();

        let transcribed = model.transcribe(input)?;

        println!("transcribed: {:?}", transcribed);
        println!("took: {:?}", start_time.elapsed());
    }
}
