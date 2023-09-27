use pin_project::pin_project;
use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::Stream;

pub trait TextStream<I: AsRef<str>>: Stream<Item = I> {
    fn sentences(self) -> SentenceStream<Self, I>
    where
        Self: Sized,
    {
        SentenceStream::new(self)
    }

    fn words(self) -> WordStream<Self, I>
    where
        Self: Sized,
    {
        WordStream::new(self)
    }

    fn paragraphs(self) -> ParagraphStream<Self, I>
    where
        Self: Sized,
    {
        ParagraphStream::new(self)
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> TextStream<I> for S {}

pub trait Pattern {
    fn matches(&self, char: char) -> bool;
}

#[pin_project]
pub struct SegmentedStream<S: Stream<Item = I>, I: AsRef<str>, P: Pattern> {
    #[pin]
    backing: S,
    queue: VecDeque<String>,
    incomplete: String,
    pattern: P,
}

impl<S: Stream<Item = I>, I: AsRef<str>, P: Pattern> SegmentedStream<S, I, P> {
    pub fn new(backing: S, pattern: P) -> Self {
        Self {
            backing,
            queue: Default::default(),
            incomplete: Default::default(),
            pattern,
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>, P: Pattern> Stream for SegmentedStream<S, I, P> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let projected = self.project();
        let mut backing = projected.backing;
        let incomplete = projected.incomplete;
        let queue = projected.queue;
        if let Some(next) = queue.pop_front() {
            return Poll::Ready(Some(next));
        }
        loop {
            let poll = backing.as_mut().poll_next(cx);
            match poll {
                Poll::Ready(Some(item)) => {
                    let item = item.as_ref();
                    let mut completed = None;
                    for char in item.chars() {
                        if projected.pattern.matches(char) {
                            incomplete.push(char);
                            let full_sentence = std::mem::take(incomplete);
                            if completed.is_some() {
                                queue.push_back(full_sentence);
                            } else {
                                completed = Some(full_sentence);
                            }
                        } else {
                            incomplete.push(char);
                        }
                    }
                    if let Some(completed) = completed {
                        return Poll::Ready(Some(completed));
                    }
                }
                Poll::Ready(None) => {
                    if !incomplete.is_empty() {
                        return Poll::Ready(Some(std::mem::take(incomplete)));
                    } else {
                        return Poll::Ready(None);
                    }
                }
                _ => {
                    return Poll::Pending;
                }
            }
        }
    }
}

struct SentencePattern;

impl Pattern for SentencePattern {
    fn matches(&self, char: char) -> bool {
        char == '.' || char == '?' || char == '!'
    }
}

#[pin_project]
pub struct SentenceStream<S: Stream<Item = I>, I: AsRef<str>> {
    #[pin]
    segmented: SegmentedStream<S, I, SentencePattern>,
}

impl<S: Stream<Item = I>, I: AsRef<str>> SentenceStream<S, I> {
    pub fn new(backing: S) -> Self {
        Self {
            segmented: SegmentedStream::new(backing, SentencePattern),
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> Stream for SentenceStream<S, I> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().segmented.poll_next(cx)
    }
}

#[pin_project]
pub struct WordStream<S: Stream<Item = I>, I: AsRef<str>> {
    #[pin]
    segmented: SegmentedStream<S, I, WordPattern>,
}

impl<S: Stream<Item = I>, I: AsRef<str>> WordStream<S, I> {
    pub fn new(backing: S) -> Self {
        Self {
            segmented: SegmentedStream::new(backing, WordPattern),
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> Stream for WordStream<S, I> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().segmented.poll_next(cx)
    }
}

struct WordPattern;

impl Pattern for WordPattern {
    fn matches(&self, char: char) -> bool {
        char.is_whitespace()
    }
}

#[pin_project]
pub struct ParagraphStream<S: Stream<Item = I>, I: AsRef<str>> {
    #[pin]
    segmented: SegmentedStream<S, I, ParagraphPattern>,
}

impl<S: Stream<Item = I>, I: AsRef<str>> ParagraphStream<S, I> {
    pub fn new(backing: S) -> Self {
        Self {
            segmented: SegmentedStream::new(backing, ParagraphPattern),
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> Stream for ParagraphStream<S, I> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().segmented.poll_next(cx)
    }
}

struct ParagraphPattern;

impl Pattern for ParagraphPattern {
    fn matches(&self, char: char) -> bool {
        char == '\n'
    }
}
