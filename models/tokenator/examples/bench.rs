use tokenator::*;

const FAST_FILE: &str = "tokenizer-fast.bin";
const HF_FILE: &str = "tokenizer.json";

fn load_tokenizers() -> FastBPETokenizer {
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
    let huggingface_tokenizer = tokenizers::Tokenizer::from_file(HF_FILE).unwrap();

    let text = std::fs::read_to_string("bigfile.txt").unwrap();

    // read the first argument as a file path to read from
    let mut input_tokens = Vec::new();
    let mut input_levels = Vec::new();
    let mut input_merges = Vec::new();
    let mut input_merge_priorities = Vec::new();

    for scale in 12..32 {
        let mut total_fast = 0.0;
        let mut total_hf = 0.0;
        let mut iterations = 0;
        for _ in 0..1000000 / (2_usize.pow(scale)) {
            let size = 2_usize.pow(scale);
            let text = (0..)
                .flat_map(|_| text.chars())
                .filter(|c| c.is_alphabetic())
                .skip(size / 2)
                .skip_while({
                    let mut i = 0;
                    move |c| {
                        i += c.len_utf8();
                        i < 75 * 16
                    }
                })
                .take(size)
                .take(16)
                .collect::<String>();
            input_tokens.clear();
            input_levels.clear();
            input_merges.clear();
            input_merge_priorities.clear();
            let start = std::time::Instant::now();
            tokenizer.tokenize(
                &text,
                &mut input_tokens,
                &mut input_levels,
                &mut input_merges,
                &mut input_merge_priorities,
            );
            total_fast += start.elapsed().as_secs_f64();

            let start = std::time::Instant::now();
            huggingface_tokenizer.encode(text.clone(), true).unwrap();
            total_hf += start.elapsed().as_secs_f64();
            iterations += 1;
        }
        println!("scale {scale}");
        println!(
            "fast {:?}",
            std::time::Duration::from_secs_f64(total_fast / iterations as f64)
        );
        println!(
            "hf {:?}",
            std::time::Duration::from_secs_f64(total_hf / iterations as f64)
        );
    }
}
