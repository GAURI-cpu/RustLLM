use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // Load model + tokenizer (same as Step 4)
    println!("AVX enabled: {}", candle_core::utils::with_avx());
    let device = Device::Cpu;
    let mut file = std::fs::File::open("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(anyhow::Error::msg)?;

    // Tokens that mean "I'm done"
    let stop_tokens: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    // PREFILL: run the whole prompt once
    let prompt = "The capital of France is";
    let mut tokens = tokenizer.encode(prompt, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    print!("{}", prompt);

    let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits: Vec<f32> = model.forward(&input, 0)?.squeeze(0)?.to_vec1()?;
    let mut next = argmax(&logits);

    // DECODE: one new token at a time
    let start = Instant::now();
    let mut generated = 0;
    for _ in 0..50 {
        if stop_tokens.contains(&next) {
            break;
        }
        tokens.push(next);
        generated += 1;

        // show the new word immediately
        let word = tokenizer.decode(&[next], false).map_err(anyhow::Error::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;

        // feed ONLY the new token; the KV cache remembers the rest
        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        let position = tokens.len() - 1;
        let logits: Vec<f32> = model.forward(&input, position)?.squeeze(0)?.to_vec1()?;
        next = argmax(&logits);
    }

    
    let secs = start.elapsed().as_secs_f64();
    println!("\n\n[{} tokens in {:.2}s = {:.1} tokens/sec]", generated, secs, generated as f64 / secs);
    Ok(())
}

// Return the position of the highest score = the most likely token id
fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    best as u32
}