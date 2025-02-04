use kalosm::sound::*;
use rodio::Decoder;
use std::fs::File;
use std::io::BufReader;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();

    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::QuantizedLargeV3Turbo)
        .build()
        .await?;

    // Load audio from a file
    let file = BufReader::new(File::open("./models/rwhisper/examples/samples_jfk.wav").unwrap());
    // Decode that sound file into a source
    let audio = Decoder::new(file).unwrap();

    // Transcribe the source audio into text
    let mut text = model.transcribe(audio).timestamped();

    // As the model transcribes the audio, print the text to the console
    while let Some(segment) = text.next().await {
        for chunk in segment.chunks() {
            let timestamp = chunk.timestamp().unwrap();
            println!("{:0.2}..{:0.2}", timestamp.start, timestamp.end);
            println!("{chunk}");
        }
    }

    Ok(())
}
