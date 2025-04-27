use kalosm::sound::*;
use rodio::{Decoder, Source};
use std::{io::Cursor, time::Duration};

#[tokio::test]
async fn transcribe_the_odyssey() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();
    // First download the audio file from the internet
    let url =
        "https://ia802301.us.archive.org/5/items/odyssey_2409_librivox/odyssey_01_homer_128kb.mp3";
    let response = reqwest::get(url).await?;
    let content = response.bytes().await?;

    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::QuantizedTinyEn)
        .build()
        .await?;

    let source = Cursor::new(content);
    // Decode that sound file into a source
    let audio = Decoder::new(source).unwrap();
    let audio = audio.take_duration(Duration::from_secs(60 * 10));

    // Transcribe the source audio into text
    let mut text = model.transcribe(audio).timestamped();

    // As the model transcribes the audio, print the text to the console
    while let Some(segment) = text.next().await {
        for chunk in segment.chunks() {
            if let Some(timestamp) = chunk.timestamp() {
                println!("{:0.2}..{:0.2}", timestamp.start, timestamp.end);
                println!("{chunk}");
            } else {
                println!("no timestamp for {chunk}");
            }
        }
    }

    Ok(())
}
