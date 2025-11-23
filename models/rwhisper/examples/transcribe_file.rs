use kalosm::sound::*;
use rodio::Decoder;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::tiny_en())
        .build()
        .await?;

    // Load audio from a file
    let contents = std::fs::read("./models/rwhisper/examples/samples_jfk.wav").unwrap();
    let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();

    let (_stream, stream_handle) = rodio::OutputStream::try_default()?;
    let sink = rodio::Sink::try_new(&stream_handle).unwrap();
    let rate = audio.sample_rate() as f32;

    // Transcribe the source audio into text
    let mut text = model.transcribe(audio);

    // As the model transcribes the audio, print the text to the console
    while let Some(segment) = text.next().await {
        for chunk in segment.chunks() {
            print!("{chunk}");
            // Play the audio chunk
            if let Some(timestamp) = chunk.timestamp() {
                let start = timestamp.start;
                let end = timestamp.end;
                let start = (start * rate) as usize;
                let end = (end * rate) as usize;
                let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();
                let audio_chunk = audio.skip(start).take(end - start).collect::<Vec<_>>();
                let audio_source = rodio::buffer::SamplesBuffer::new(1, rate as u32, audio_chunk);
                sink.append(audio_source);
                sink.sleep_until_end();
            }
        }
    }

    Ok(())
}
