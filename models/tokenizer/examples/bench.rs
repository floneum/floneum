use fast_bpe::*;

fn load_tokenizers() -> fast_bpe::FastBPETokenizer {
    const FAST_FILE: &str = "tokenizer-fast.bin";
    const HF_FILE: &str = "tokenizer.json";
    let bytes = std::fs::read(HF_FILE).unwrap();
    if let Ok(tokenizer) = std::fs::read(FAST_FILE) {
        postcard::from_bytes::<fast_bpe::FastBPETokenizer>(&tokenizer).unwrap()
    } else {
        let tokenizer = fast_bpe::FastBPETokenizer::load_from_bytes(&bytes);
        std::fs::write(FAST_FILE, postcard::to_stdvec(&tokenizer).unwrap()).unwrap();
        tokenizer
    }
}

pub fn main() {
    let tokenizer = load_tokenizers();

    let text = std::fs::read_to_string("bigfile.txt").unwrap();

    // read the first argument as a file path to read from
    let size = 10_usize.pow(6);
    let text = (0..)
        .flat_map(|_| text.chars())
        .filter(|c| c.is_ascii_alphanumeric())
        .take(size)
        .collect::<String>();

    let mut input_tokens = Vec::new();
    let mut merge_queue = MergeLayerQueue::new();

    loop {
        merge_queue.resolve(&mut input_tokens, &text, &tokenizer);
    }
}
