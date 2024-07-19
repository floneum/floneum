# Kalosm Sound

Kalosm Sound is a collection of audio models and utilities for the Kalosm framework. It supports several voice activity detection models, and provides utilities for transcribing audio into text.


## Sound Streams

Models in kalosm sound work with any [`AsyncSource`]. You can use [`MicInput::stream`] to stream audio from the microphone, or any synchronous audio source that implements [`rodio::Source`] like a mp3 or wav file.


## Voice Activity Detection

VAD models are used to detect when a speaker is speaking in a given audio stream. The simplest way to use a VAD model is to create an audio stream and call [`VoiceActivityDetectorExt::voice_activity_stream`] to stream audio chunks that are actively being spoken:

```rust, no_run
use kalosm::sound::*;
#[tokio::main]
async fn main() {
    // Get the default microphone input
    let mic = MicInput::default();
    // Stream the audio from the microphone
    let stream = mic.stream().unwrap();
    // Detect voice activity in the audio stream
    let vad = stream.voice_activity_stream();
    while let Some(input) = vad.next().await {
        println!("Probability: {}", input.probability);
    }
}
```

Kalosm also provides [`VoiceActivityDetectorExt::rechunk_voice_activity`] to collect chunks of consecutive audio samples with a high vad probability. This can be useful for applications like speech recognition where context between consecutive audio samples is important.

```rust, no_run
use kalosm::sound::*;
#[tokio::main]
async fn main() {
    // Get the default microphone input
    let mic = MicInput::default();
    // Stream the audio from the microphone
    let stream = mic.stream().unwrap();
    // Chunk the audio into chunks of speech
    let vad = stream.voice_activity_stream();
    let mut audio_chunks = vad.rechunk_voice_activity();
    // Print the chunks as they are streamed in
    while let Some(input) = audio_chunks.next().await {
        println!("New voice activity chunk: {:?}", input);
    }
}
```

## Transcription

You can use the [`Whisper`] model to transcribe audio into text. Kalosm can transcribe any [`AsyncSource`] into a transcription stream with the [`AsyncSourceTranscribeExt::transcribe`] method:

```rust, no_run
use kalosm::sound::*;
#[tokio::main]
async fn main() {
    // Get the default microphone input
    let mic = MicInput::default();
    // Stream the audio from the microphone
    let stream = mic.stream().unwrap();
    // Transcribe the audio into text with the default Whisper model
    let transcribe = stream.transcribe(Whisper::new().await.unwrap());
    // Print the text as it is streamed in
    transcribe.to_std_out().await.unwrap();
}
```
