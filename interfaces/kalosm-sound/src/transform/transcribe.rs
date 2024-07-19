use kalosm_streams::text_stream::ChannelTextStream;

use super::voice_audio_detector::*;
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
    ///     let stream = mic.stream().unwrap();
    ///
    ///     // Transcribe the audio into text in chunks based on voice activity.
    ///     let text_stream = stream.transcribe(model);
    ///
    ///     // Finally, print the text to the console
    ///     text_stream.to_std_out().await.unwrap();
    ///
    ///     Ok(())
    /// }
    /// ```
    fn transcribe(self, model: rwhisper::Whisper) -> ChannelTextStream<rwhisper::Segment> {
        rwhisper::TranscribeChunkedAudioStreamExt::transcribe(
            self.voice_activity_stream().rechunk_voice_activity(),
            model,
        )
    }
}

impl<S: AsyncSource + Unpin + Send + Sized + 'static> AsyncSourceTranscribeExt for S {}
