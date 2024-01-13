mod attention_layer;
mod rope;
mod silu;

struct LlamaConfig {
    rope_theta: f32,
    context_length: usize,
    head_dimension: usize,
}

struct Llama {}
