use std::{path::PathBuf, time::Duration};

use futures_util::StreamExt;
use kalosm_sound::*;
use rodio::{OutputStream, Source};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a channel that sends and receives the audio data.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Create a thread that reads the audio data from the microphone and sends it to the channel.
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let file = PathBuf::from("examples/samples_jfk.wav");
    let stream = rodio::Decoder::new(std::fs::File::open(file).unwrap()).unwrap();
                let mut vad = stream
                    .denoise_and_detect_voice_activity()
                    .rechunk_voice_activity(Duration::from_millis(600), 0.15);
                while let Some(samples) = vad.next().await {
                    // Send the audio data to the channel.
                    let _ = tx.send(samples);
                }
            });
    });

    // Create a new small whisper model.
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::MediumEn)
        .build()
        .await?;

    let (_device, stream_handle) = OutputStream::try_default().unwrap();
    let sink = rodio::Sink::try_new(&stream_handle)?;
    while let Some(samples) = rx.recv().await {
        println!("received: {:?}", samples.total_duration());
        let sample_rate = rodio::Source::sample_rate(&samples);
        let samples_cloned = samples.collect::<Vec<_>>();
        let samples: rodio::buffer::SamplesBuffer<f32> = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples_cloned.clone());
        let samples_clone = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples_cloned);
        sink.append(samples);
        // Transcribe the audio.
        let mut text = model.transcribe(samples_clone)?;

        // As the model transcribes the audio, print the text to the console.
        while let Some(text) = text.next().await {
            if text.probability_of_no_speech() < 0.1 {
                print!("transcribed: {}", text.text());
            } else {
                print!("transcribed (no speech): {}", text.text());
            }
            println!();
        }
    }

    Ok(())
}
