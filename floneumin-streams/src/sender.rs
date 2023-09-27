use futures_util::Stream;

pub struct ChannelTextStream<S: AsRef<str>> {
    receiver: tokio::sync::mpsc::UnboundedReceiver<S>,
}

impl<S: AsRef<str>> From<tokio::sync::mpsc::UnboundedReceiver<S>> for ChannelTextStream<S> {
    fn from(receiver: tokio::sync::mpsc::UnboundedReceiver<S>) -> Self {
        Self { receiver }
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
