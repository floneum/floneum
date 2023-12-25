#![allow(non_snake_case, non_upper_case_globals)]
//! This benchmark tests just the overhead of Dioxus itself.
//!
//! For the JS Framework Benchmark, both the framework and the browser is benchmarked together. Dioxus prepares changes
//! to be made, but the change application phase will be just as performant as the vanilla wasm_bindgen code. In essence,
//! we are measuring the overhead of Dioxus, not the performance of the "apply" phase.
//!
//!
//! Pre-templates (Mac M1):
//! - 3ms to create 1_000 rows
//! - 30ms to create 10_000 rows
//!
//! Post-templates
//! - 580us to create 1_000 rows
//! - 6.2ms to create 10_000 rows
//!
//! As pure "overhead", these are amazing good numbers, mostly slowed down by hitting the global allocator.
//! These numbers don't represent Dioxus with the heuristic engine installed, so I assume it'll be even faster.

use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion};
use kalosm_llama::{prelude::*, LlamaModel};

criterion_group!(mbenches, generation);
criterion_main!(mbenches);

fn generation(c: &mut Criterion) {
    c.bench_function("feed text short", |b| {
        let model =
            LlamaModel::from_builder(Llama::builder().with_source(LlamaSource::mistral_7b()))
                .unwrap();
        let prompt = "Hello world";

        b.iter(|| {
            let mut session = model.new_session().unwrap();
            model.feed_text(&mut session, prompt)
        })
    });

    c.bench_function("feed text long", |b| {
        let model =
            LlamaModel::from_builder(Llama::builder().with_source(LlamaSource::mistral_7b()))
                .unwrap();
        let prompt = "Hello world".repeat(10);

        b.iter(|| {
            let mut session = model.new_session().unwrap();
            model.feed_text(&mut session, &prompt)
        })
    });

    c.bench_function("generate text", |b| {
        let model =
            LlamaModel::from_builder(Llama::builder().with_source(LlamaSource::mistral_7b()))
                .unwrap();
        let prompt = "Hello world";

        b.iter(|| {
            let mut session = model.new_session().unwrap();
            model.stream_text_with_sampler(
                &mut session,
                prompt,
                Some(10),
                None,
                Arc::new(Mutex::new(GenerationParameters::default().sampler())),
                |_| Ok(kalosm_language_model::ModelFeedback::Continue),
            )
        })
    });

    c.bench_function("generate text long", |b| {
        let model =
            LlamaModel::from_builder(Llama::builder().with_source(LlamaSource::mistral_7b()))
                .unwrap();
        let prompt = "Hello world";

        b.iter(|| {
            let mut session = model.new_session().unwrap();
            model.stream_text_with_sampler(
                &mut session,
                prompt,
                Some(100),
                None,
                Arc::new(Mutex::new(GenerationParameters::default().sampler())),
                |_| Ok(kalosm_language_model::ModelFeedback::Continue),
            )
        })
    });
}
