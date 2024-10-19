use pretty_assertions::assert_eq;
use rand::Rng;
use tokenator::*;
use tokenizers::Tokenizer;

#[test]
fn fuzz() {
    const FAST_FILE: &str = "tokenizer-fast.bin";
    const HF_FILE: &str = "tokenizer.json";
    let tokenizer = if let Ok(tokenizer) = std::fs::read(FAST_FILE) {
        postcard::from_bytes::<FastBPETokenizer>(&tokenizer).unwrap()
    } else {
        let bytes = std::fs::read(HF_FILE).unwrap();
        let tokenizer = FastBPETokenizer::load_from_bytes(&bytes);
        std::fs::write(FAST_FILE, postcard::to_stdvec(&tokenizer).unwrap()).unwrap();
        tokenizer
    };

    let hf_tokenizer = Tokenizer::from_file(HF_FILE).unwrap();

    for [size, count] in [[10, 1000], [10_000, 50]] {
        for _ in 0..count {
            let text = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(rand::random::<usize>() % size)
                .map(char::from)
                .collect::<String>();

            let mut input_tokens = Vec::new();
            let mut merge_queue = Vec::new();

            let fast_tokens = {
                let index = MergeLayerQueue::resolve(
                    &mut input_tokens,
                    &text,
                    &tokenizer,
                    &mut merge_queue,
                );
                input_tokens
                    .iter()
                    .take(index)
                    .map(|t| t.token())
                    .collect::<Vec<_>>()
            };
            let hf_tokens = hf_tokenizer.encode(text.clone(), true).unwrap();
            // Try to reduce the reproduction
            if fast_tokens != hf_tokens.get_ids() {
                let start = {
                    let mut start = 0;
                    while start < text.chars().count() {
                        start += 1;
                        let text = text.chars().skip(start).collect::<String>();
                        let fast_tokens = {
                            let index = MergeLayerQueue::resolve(
                                &mut input_tokens,
                                &text,
                                &tokenizer,
                                &mut merge_queue,
                            );
                            input_tokens
                                .iter()
                                .take(index)
                                .map(|t| t.token())
                                .collect::<Vec<_>>()
                        };
                        let hf_tokens = hf_tokenizer.encode(text.clone(), true).unwrap();
                        if fast_tokens == hf_tokens.get_ids() {
                            start -= 1;
                            break;
                        }
                    }
                    start
                };
                let len = {
                    let mut len = text.chars().skip(start).count();
                    while len > 0 {
                        len -= 1;
                        let text = text.chars().skip(start).take(len).collect::<String>();
                        let fast_tokens = {
                            let index = MergeLayerQueue::resolve(
                                &mut input_tokens,
                                &text,
                                &tokenizer,
                                &mut merge_queue,
                            );
                            input_tokens
                                .iter()
                                .take(index)
                                .map(|t| t.token())
                                .collect::<Vec<_>>()
                        };
                        let hf_tokens = hf_tokenizer.encode(text.clone(), true).unwrap();
                        if fast_tokens == hf_tokens.get_ids() {
                            len += 1;
                            break;
                        }
                    }
                    len
                };

                let text = text.chars().skip(start).take(len).collect::<String>();
                let fast_tokens = {
                    let index = MergeLayerQueue::resolve(
                        &mut input_tokens,
                        &text,
                        &tokenizer,
                        &mut merge_queue,
                    );
                    input_tokens
                        .iter()
                        .take(index)
                        .map(|t| t.token())
                        .collect::<Vec<_>>()
                };
                let hf_tokens = hf_tokenizer.encode(text.clone(), true).unwrap();
                assert_eq!(fast_tokens, hf_tokens.get_ids(), "failed to encode {text}");
                panic!("{start} {len} {text}");
            }
        }
    }
}
