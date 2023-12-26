#![allow(non_snake_case, non_upper_case_globals)]

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
}
