use futures_util::{Stream, StreamExt};

/// A stream of text from a tokio channel.
pub struct ChannelTextStream<S: AsRef<str>> {
    receiver: tokio::sync::mpsc::UnboundedReceiver<S>,
}

impl<S: AsRef<str>> std::fmt::Debug for ChannelTextStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelTextStream").finish()
    }
}

impl<S: AsRef<str>> From<tokio::sync::mpsc::UnboundedReceiver<S>> for ChannelTextStream<S> {
    fn from(receiver: tokio::sync::mpsc::UnboundedReceiver<S>) -> Self {
        Self { receiver }
    }
}

impl<S: AsRef<str>> ChannelTextStream<S> {
    /// Write the stream to a writer.
    pub async fn write_to<W: std::io::Write>(mut self, mut writer: W) -> std::io::Result<()> {
        while let Some(text) = self.next().await {
            writer.write_all(text.as_ref().as_bytes())?;
            writer.flush()?;
        }
        Ok(())
    }

    /// Write the stream to standard output.
    pub async fn to_std_out(self) -> std::io::Result<()> {
        self.write_to(std::io::stdout()).await
    }
}

impl<S: AsRef<str>> Stream for ChannelTextStream<S> {
    type Item = S;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> core::task::Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}
