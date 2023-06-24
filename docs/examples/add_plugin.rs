#![allow(unused)]

fn main() {}

// ANCHOR: plugin
use floneum_rust::*;

#[export_plugin]
/// adds two numbers
fn add(first: i64, second: i64) -> i64 {
    first + second
}
// END_ANCHOR: plugin
