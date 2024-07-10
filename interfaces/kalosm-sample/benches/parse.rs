#![allow(non_snake_case, non_upper_case_globals)]

use criterion::{criterion_group, criterion_main, Criterion};
use kalosm_sample::*;

criterion_group!(mbenches, generation);
criterion_main!(mbenches);

fn generation(c: &mut Criterion) {
    c.bench_function("parse sequence static", |b| {
        let parser = SequenceParser::new(
            LiteralParser::from("Hello, "),
            LiteralParser::from("world!"),
        );
        let state = parser.create_parser_state();
        b.iter(|| parser.parse(&state, b"Hello, world!"))
    });
    c.bench_function("parse sequence words", |b| {
        let parser = SequenceParser::new(WordParser::new(), WordParser::new());
        let state = parser.create_parser_state();
        b.iter(|| parser.parse(&state, b"Hello world"))
    });
    c.bench_function("parse sequence regex static", |b| {
        let parser = RegexParser::new(r"Hello, world!").unwrap();
        let state = parser.create_parser_state();
        b.iter(|| parser.parse(&state, b"Hello, world!"))
    });
    c.bench_function("parse sequence regex words", |b| {
        let parser = RegexParser::new(r"\w{1,20} \w{1,20}").unwrap();
        let state = parser.create_parser_state();
        b.iter(|| parser.parse(&state, b"Hello world"))
    });
}
