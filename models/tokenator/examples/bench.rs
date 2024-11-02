use tokenator::*;

fn load_tokenizers() -> FastBPETokenizer {
    const FAST_FILE: &str = "tokenizer-fast.bin";
    const HF_FILE: &str = "tokenizer.json";
    let bytes = std::fs::read(HF_FILE).unwrap();
    if let Ok(tokenizer) = std::fs::read(FAST_FILE) {
        postcard::from_bytes::<FastBPETokenizer>(&tokenizer).unwrap()
    } else {
        let tokenizer = FastBPETokenizer::load_from_bytes(&bytes);
        std::fs::write(FAST_FILE, postcard::to_stdvec(&tokenizer).unwrap()).unwrap();
        tokenizer
    }
}

pub fn main() {
    let tokenizer = load_tokenizers();

    let text = std::fs::read_to_string("bigfile.txt").unwrap();

    // read the first argument as a file path to read from
    let size = 10_usize.pow(5);
    let text = (0..)
        .flat_map(|_| text.chars())
        .filter(|c| c.is_alphabetic())
        .take(size)
        .collect::<String>();

    let mut input_tokens = Vec::new();
    let mut merge_queue = Vec::new();

    for _ in 0..100 {
        MergeLayerQueue::resolve(&mut input_tokens, &text, &tokenizer, &mut merge_queue);
    }
}
