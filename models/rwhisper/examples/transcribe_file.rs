use kalosm::sound::*;
use rodio::Decoder;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();

    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::distil_large_v3_5())
        .build()
        .await?;

    // Load audio from a file
    let contents = std::fs::read("./models/rwhisper/examples/samples_jfk.wav").unwrap();

    loop {
        let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();
        let start_time = std::time::Instant::now();
        let text = model.transcribe(audio).all_text().await;
        println!("{text}");
        let elapsed = start_time.elapsed();
        println!("Transcription took: {:.2?}", elapsed);
    }

    Ok(())
}
