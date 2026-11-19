use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("AVX enabled: {}", candle_core::utils::with_avx());
    let device = Device::Cpu;
    let mut file = std::fs::File::open("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(anyhow::Error::msg)?;
    let mut rng = Rng::from_time();

    let stop_tokens: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    // The whole conversation so far: (role, text)
    let mut history: Vec<(String, String)> = Vec::new();

    println!("Chat started. Type /reset to forget the conversation, /exit to quit.\n");

    loop {
        // 1. Read what the user types
        print!("you › ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            break; // input closed
        }
        let input = line.trim();

        // 2. Commands
        if input.is_empty() {
            continue;
        }
        if input == "/exit" {
            break;
        }
        if input == "/reset" {
            history.clear();
            println!("(conversation cleared)\n");
            continue;
        }

        // 3. Add the message to the conversation and build the full prompt
        history.push(("user".to_string(), input.to_string()));
        let prompt = build_prompt(&history);

        // 4. Generate the reply
        print!("bot › ");
        let reply = generate(&mut model, &tokenizer, &device, &prompt, &stop_tokens, &mut rng)?;

        // 5. Remember the reply too, so the next question has context
        history.push(("assistant".to_string(), reply));
    }

    Ok(())
}

// Turn the conversation into ChatML text, ending with an open assistant turn
fn build_prompt(history: &[(String, String)]) -> String {
    let mut prompt = String::from("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    for (role, text) in history {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, text));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// Run prefill + decode for one reply; prints it live and returns the full text
fn generate(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    stop_tokens: &[u32],
    rng: &mut Rng,
) -> Result<String> {
    // New turn: forget the old KV cache, we re-read the whole conversation
    model.clear_kv_cache();

    let mut tokens = tokenizer.encode(prompt, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let prompt_len = tokens.len();

    // PREFILL
    let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits: Vec<f32> = model.forward(&input, 0)?.squeeze(0)?.to_vec1()?;
    let mut next = sample(&logits, &tokens, 0.7, 40, 1.1, rng);

    // DECODE
    let start = Instant::now();
    let mut reply = String::new();
    for _ in 0..500 {
        if stop_tokens.contains(&next) {
            break;
        }
        tokens.push(next);

        // Clean streaming: decode everything generated so far, print only the new part.
        // Skip printing while a character is still half-finished (shows as �).
        let text = tokenizer.decode(&tokens[prompt_len..], false).map_err(anyhow::Error::msg)?;
        if text.len() > reply.len() && !text.ends_with('\u{FFFD}') {
            print!("{}", &text[reply.len()..]);
            std::io::stdout().flush()?;
            reply = text;
        }

        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let position = tokens.len() - 1;
        let logits: Vec<f32> = model.forward(&input, position)?.squeeze(0)?.to_vec1()?;
        next = sample(&logits, &tokens, 0.7, 40, 1.1, rng);
    }

    let generated = tokens.len() - prompt_len;
    let secs = start.elapsed().as_secs_f64();
    println!("\n[{} tokens, {:.1} tok/s]\n", generated, generated as f64 / secs);
    Ok(reply)
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