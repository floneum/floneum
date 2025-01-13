# Kalosm Sound

Kalosm Sound is a collection of audio models and utilities for the Kalosm framework. It supports several [voice activity detection models](crate::VoiceActivityDetectorExt), and provides utilities for [transcribing audio into text](crate::AsyncSourceTranscribeExt).


## Sound Streams

Models in kalosm sound work with any [`AsyncSource`]. You can use [`MicInput::stream`] to stream audio from the microphone, or any synchronous audio source that implements [`rodio::Source`] like a mp3 or wav file.

You can transform the audio streams with:
- [`VoiceActivityDetectorExt::voice_activity_stream`]: Detect voice activity in the audio data
- [`DenoisedExt::denoise_and_detect_voice_activity`]: Denoise the audio data and detect voice activity
- [`AsyncSourceTranscribeExt::transcribe`]: Chunk an audio stream based on voice activity and then transcribe the chunked audio data
- [`VoiceActivityStreamExt::rechunk_voice_activity`]: Chunk an audio stream based on voice activity
- [`VoiceActivityStreamExt::filter_voice_activity`]: Filter chunks of audio data based on voice activity
- [`TranscribeChunkedAudioStreamExt::transcribe`]: Transcribe a chunked audio stream


## Voice Activity Detection

VAD models are used to detect when a speaker is speaking in a given audio stream. The simplest way to use a VAD model is to create an audio stream and call [`VoiceActivityDetectorExt::voice_activity_stream`] to stream audio chunks that are actively being spoken:

```rust, no_run
use kalosm::sound::*;
#[tokio::main]
async fn main() {
    // Get the default microphone input
    let mic = MicInput::default();
    // Stream the audio from the microphone
    let stream = mic.stream();
    // Detect voice activity in the audio stream
    let mut vad = stream.voice_activity_stream();
    while let Some(input) = vad.next().await {
        println!("Probability: {}", input.probability);
    }
}
```

Kalosm also provides [`VoiceActivityStreamExt::rechunk_voice_activity`] to collect chunks of consecutive audio samples with a high vad probability. This can be useful for applications like speech recognition where context between consecutive audio samples is important.

```rust, no_run
use kalosm::sound::*;
use rodio::Source;
#[tokio::main]
async fn main() {
    // Get the default microphone input
    let mic = MicInput::default();
    // Stream the audio from the microphone
    let stream = mic.stream();
    // Chunk the audio into chunks of speech
    let vad = stream.voice_activity_stream();
    let mut audio_chunks = vad.rechunk_voice_activity();
    // Print the chunks as they are streamed in
    while let Some(input) = audio_chunks.next().await {
        println!("New voice activity chunk with duration {:?}", input.total_duration());
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
    let stream = mic.stream();
    // Transcribe the audio into text with the default Whisper model
    let mut transcribe = stream.transcribe(Whisper::new().await.unwrap());
    // Print the text as it is streamed in
    transcribe.to_std_out().await.unwrap();
}
```
