use rwhisper::ChunkedTranscriptionTask;

use super::denoise::*;
use super::voice_audio_detector_ext::*;
use crate::AsyncSource;

/// An extension trait for [`AsyncSource`] that integrates with [`crate::Whisper`].
pub trait AsyncSourceTranscribeExt: AsyncSource + Unpin + Send + Sized + 'static {
    /// Chunk the audio stream into segments based on voice activity and then transcribe those segments.  The model will transcribe segments of speech that are separated by silence.
    ///
    /// ```rust, no_run
    /// use kalosm::sound::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), anyhow::Error> {
    ///     // Create a new whisper model.
    ///     let model = Whisper::new().await?;
    ///
    ///     // Stream audio from the microphone
    ///     let mic = MicInput::default();
    ///     let stream = mic.stream();
    ///
    ///     // Transcribe the audio into text in chunks based on voice activity.
    ///     let mut text_stream = stream.transcribe(model);
    ///
    ///     // Finally, print the text to the console
    ///     text_stream.to_std_out().await.unwrap();
    ///
    ///     Ok(())
    /// }
    /// ```
    fn transcribe(
        self,
        model: rwhisper::Whisper,
    ) -> ChunkedTranscriptionTask<VoiceActivityRechunkerStream<DenoisedStream<Self>>>
    {
        rwhisper::TranscribeChunkedAudioStreamExt::transcribe(
            self.denoise_and_detect_voice_activity()
                .rechunk_voice_activity(),
            model,
        )
    }
}

impl<S: AsyncSource + Unpin + Send + Sized + 'static> AsyncSourceTranscribeExt for S {}
