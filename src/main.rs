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
    let mut rng = Rng::from_time();

    // Tokens that mean "I'm done"
    let stop_tokens: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    // PREFILL: run the whole prompt once
    let question = "What is Rust? Answer in two sentences.";
    let prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        question
    );
    let mut tokens = tokenizer.encode(prompt.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    println!("you › {}", question);
    print!("bot › ");

    let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits: Vec<f32> = model.forward(&input, 0)?.squeeze(0)?.to_vec1()?;
    let mut next = sample(&logits, &tokens, 0.7, 40, 1.1, &mut rng);

    // DECODE: one new token at a time
    let start = Instant::now();
    let mut generated = 0;
    for _ in 0..200 {
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
        next = sample(&logits, &tokens, 0.7, 40, 1.1, &mut rng);
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

// A tiny random number generator (xorshift)
struct Rng(u64);

impl Rng {
    fn from_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Rng(nanos | 1)
    }

    // a random number between 0.0 and 1.0
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

// Pick the next token: repeat penalty -> temperature -> top-k -> weighted random pick
fn sample(logits: &[f32], history: &[u32], temperature: f32, top_k: usize, repeat_penalty: f32, rng: &mut Rng) -> u32 {
    let mut scores = logits.to_vec();

    // 1. Repeat penalty: make recently used tokens less likely
    let start = history.len().saturating_sub(64);
    for &token in &history[start..] {
        let s = &mut scores[token as usize];
        if *s > 0.0 { *s /= repeat_penalty; } else { *s *= repeat_penalty; }
    }

    // temperature 0 = no randomness, just take the best
    if temperature <= 0.0 {
        return argmax(&scores);
    }

    // 2. Temperature, then 3. keep only the top-k candidates
    let mut candidates: Vec<(usize, f32)> = scores.iter().map(|s| s / temperature).enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    candidates.truncate(top_k);

    // 4. Softmax weights
    let max = candidates[0].1;
    let weights: Vec<f32> = candidates.iter().map(|(_, s)| (s - max).exp()).collect();
    let total: f32 = weights.iter().sum();

    // 5. Spin the wheel
    let mut r = rng.next_f32() * total;
    for (i, w) in weights.iter().enumerate() {
        if r < *w {
            return candidates[i].0 as u32;
        }
        r -= w;
    }
    candidates[0].0 as u32
}