use kalosm::sound::*;
use rodio::{Decoder, OutputStream, Sink, Source};
use std::fs::File;
use std::io::BufReader;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();

    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::QuantizedTinyEn)
        .build()
        .await?;

    // Load audio from a file
    let file = BufReader::new(File::open("./models/rwhisper/examples/samples_jfk.wav").unwrap());
    // Decode that sound file into a source
    let audio = Decoder::new(file).unwrap();

    // Transcribe the source audio into text
    let mut text = model.transcribe(audio)?;

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    // As the model transcribes the audio, print the text to the console
    while let Some(segment) = text.next().await {
        for chunk in segment.chunks() {
            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

            if let Some(timestamp) = chunk.timestamp() {
                // playback the audio
                let file = BufReader::new(
                    File::open("./models/rwhisper/examples/samples_jfk.wav").unwrap(),
                );
                // Decode that sound file into a source
                let decoder = Decoder::new(file).unwrap();
                let sample_rate = Source::sample_rate(&decoder);
                let channels = decoder.channels();
                let source = decoder
                    .skip((sample_rate as f32 * timestamp.start) as usize)
                    .take((sample_rate as f32 * (timestamp.end - timestamp.start)) as usize)
                    .collect::<Vec<_>>(); // decode the full song
                sink.append(rodio::buffer::SamplesBuffer::new(
                    channels,
                    sample_rate,
                    source,
                ));
                println!("{:0.2}..{:0.2}", timestamp.start, timestamp.end);
            }
            println!("{chunk}");
        }
    }

    sink.sleep_until_end();

    Ok(())
}
