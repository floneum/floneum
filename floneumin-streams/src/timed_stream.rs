use std::collections::VecDeque;

use futures_util::Stream;
use pin_project::pin_project;

pub trait TimeStamped {
    fn start(&self) -> std::time::Instant;
    fn end(&self) -> std::time::Instant;
}

pub trait TimeSeriesStream<I: TimeStamped>: Stream<Item = I> {
    fn window(self, duration: std::time::Duration) -> WindowedStream<Self, I>
    where
        Self: Sized,
    {
        WindowedStream::new(self, duration)
    }
}

#[pin_project]
pub struct WindowedStream<S: Stream<Item = I>, I: TimeStamped> {
    #[pin]
    backing: S,
    duration: std::time::Duration,
    window: VecDeque<I>,
}

impl<S: Stream<Item = I>, I: TimeStamped> WindowedStream<S, I> {
    fn new(backing: S, duration: std::time::Duration) -> Self {
        Self {
            backing,
            duration,
            window: Default::default(),
        }
    }
}

impl<S: Stream<Item = I>, I: TimeStamped + Clone> Stream for WindowedStream<S, I> {
    type Item = Vec<I>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut self_ = self.project();
        let window_start_time = self_
            .window
            .back()
            .map(|e| e.start())
            .unwrap_or_else(std::time::Instant::now);
        let window_end_time = window_start_time + *self_.duration;
        loop {
            // First poll the backing stream
            let item = self_.backing.as_mut().poll_next(cx);
            match item {
                std::task::Poll::Ready(Some(item)) => {
                    let end = item.end();
                    self_.window.push_back(item);
                    // If this item will push the window past the end time, return the window
                    if end > window_end_time {
                        return std::task::Poll::Ready(Some(
                            self_.window.iter().cloned().collect::<Vec<_>>(),
                        ));
                    }
                }
                std::task::Poll::Ready(None) => {
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}
