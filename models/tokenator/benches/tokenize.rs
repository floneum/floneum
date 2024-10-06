use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use tokenator::*;
use tokenizers::Tokenizer;

fn load_tokenizers() -> (Tokenizer, tokenator::FastBPETokenizer) {
    const FAST_FILE: &str = "tokenizer-fast.bin";
    const HF_FILE: &str = "tokenizer.json";
    let bytes = std::fs::read(HF_FILE).unwrap();
    let tokenizer = if let Ok(tokenizer) = std::fs::read(FAST_FILE) {
        postcard::from_bytes::<tokenator::FastBPETokenizer>(&tokenizer).unwrap()
    } else {
        let tokenizer = tokenator::FastBPETokenizer::load_from_bytes(&bytes);
        std::fs::write(FAST_FILE, postcard::to_stdvec(&tokenizer).unwrap()).unwrap();
        tokenizer
    };
    let hf_tokenizer = Tokenizer::from_bytes(bytes).unwrap();
    (hf_tokenizer, tokenizer)
}

pub fn tokenize_small(c: &mut Criterion) {
    let (hf_tokenizer, tokenizer) = load_tokenizers();

    let text = std::fs::read_to_string("bigfile.txt").unwrap();

    // read the first argument as a file path to read from
    let mut group = c.benchmark_group("tokenize-small");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));
    let samples = 200;
    let step = samples / 100;
    for size in 1..=50 {
        let size = size * step;
        group.throughput(Throughput::Bytes(size as u64));
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_millis(500));
        let text = (0..)
            .flat_map(|_| text.chars())
            .take(size)
            .collect::<String>();
        group.bench_with_input(BenchmarkId::new("Fast", size), &text, |b, text| {
            let mut input_tokens = Vec::new();
            let mut levels = Vec::new();

            b.iter(|| MergeLayerQueue::resolve(&mut input_tokens, text, &tokenizer, &mut levels))
        });
        group.bench_with_input(BenchmarkId::new("HuggingFace", size), &text, |b, text| {
            b.iter(|| hf_tokenizer.encode(text.clone(), true).unwrap())
        });
    }
    group.finish();
}

pub fn tokenize_large(c: &mut Criterion) {
    let (hf_tokenizer, tokenizer) = load_tokenizers();

    let text = std::fs::read_to_string("bigfile.txt").unwrap();

    // read the first argument as a file path to read from
    let mut group = c.benchmark_group("tokenize-large");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));
    for size in 1..=10 {
        let size = 10_usize.pow(6) * size;
        group.throughput(Throughput::Bytes(size as u64));
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_secs(10));
        let text = (0..)
            .flat_map(|_| text.chars())
            .take(size)
            .collect::<String>();
        group.bench_with_input(BenchmarkId::new("Fast", size), &text, |b, text| {
            let mut input_tokens = Vec::new();
            let mut levels = Vec::new();

            b.iter(|| MergeLayerQueue::resolve(&mut input_tokens, text, &tokenizer, &mut levels))
        });
        group.bench_with_input(BenchmarkId::new("HuggingFace", size), &text, |b, text| {
            b.iter(|| hf_tokenizer.encode(text.clone(), true).unwrap())
        });

        let text = (0..)
            .flat_map(|_| text.chars())
            .filter(|c| c.is_alphabetic())
            .take(size)
            .map(char::from)
            .collect::<String>();
        assert_eq!(text.len(), size);
        group.bench_with_input(BenchmarkId::new("Fast-Word", size), &text, |b, text| {
            let mut input_tokens = Vec::new();
            let mut levels = Vec::new();

            b.iter(|| MergeLayerQueue::resolve(&mut input_tokens, text, &tokenizer, &mut levels))
        });
        group.bench_with_input(
            BenchmarkId::new("HuggingFace-Word", size),
            &text,
            |b, text| b.iter(|| hf_tokenizer.encode(text.clone(), true).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, tokenize_small, tokenize_large);
criterion_main!(benches);
