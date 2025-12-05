use fusor_core::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new().await?;

    // Simple test case - 4D tensors [batch, heads, seq, dim]
    let q_data = [vec![vec![vec![1.0f32; 64]; 128]; 8]]; // [1, 8, 128, 64]
    let k_data = [vec![vec![vec![1.0f32; 64]; 128]; 8]]; // [1, 8, 128, 64]
    let v_data = [vec![vec![vec![1.0f32; 64]; 128]; 8]]; // [1, 8, 128, 64]

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);

    let scale = 1.0 / (2.0_f32.sqrt());
    let start = std::time::Instant::now();
    for i in 1..1000 {
        let output = q.flash_attention(&k, &v, scale);
        _ = output.as_slice().await;
        if i % 100 == 0 {
            let elapsed = start.elapsed().as_micros() as f32;
            println!("Iteration {}: {:.2} µs/it", i, elapsed / i as f32);
        }
    }

    let start = std::time::Instant::now();
    for i in 1..1000 {
        // Standard attention: Q @ K^T * scale -> softmax -> @ V
        let scores = q.mat_mul(&k.t()) * scale;
        let attn_weights = scores.softmax_last_dim();
        let output = attn_weights.mat_mul(&v);
        let _ = output.as_slice().await.unwrap();
        _ = output.as_slice().await;
        if i % 100 == 0 {
            let elapsed = start.elapsed().as_micros() as f32;
            println!("Iteration {}: {:.2} µs/it", i, elapsed / i as f32);
        }
    }

    Ok(())
}
