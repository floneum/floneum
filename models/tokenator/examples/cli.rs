use std::io::Write;
use tokenator::*;

const FAST_FILE: &str = "tokenizer-fast.bin";
const HF_FILE: &str = "tokenizer.json";

fn main() {
    tracing_subscriber::fmt::init();

    let tokenizer = if let Ok(tokenizer) = std::fs::read(FAST_FILE) {
        postcard::from_bytes::<FastBPETokenizer>(&tokenizer).unwrap()
    } else {
        let bytes = std::fs::read(HF_FILE).unwrap();
        let tokenizer = FastBPETokenizer::load_from_bytes(&bytes);
        std::fs::write(FAST_FILE, postcard::to_stdvec(&tokenizer).unwrap()).unwrap();
        tokenizer
    };

    loop {
        // read a line from stdin
        let mut line = String::new();
        print!("> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut line).unwrap();

        let text = line.trim();
        let mut input_tokens = Vec::new();
        let mut merge_queue = Vec::new();

        let index = MergeLayerQueue::resolve(&mut input_tokens, text, &tokenizer, &mut merge_queue);
        pretty_print_tokens(
            input_tokens.iter().take(index).map(|t| t.token()),
            &tokenizer,
        );
        println!();
    }
}
