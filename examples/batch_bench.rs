//! Does batching make kinSin faster on THIS computer? Measure it.
//!
//!   cargo run --release --example batch_bench
//!   cargo run --release --example batch_bench -- --model models/kinsin-1.5b-v2-q4_0.gguf --tokenizer models/tokenizer.json
//!   cargo run --release --example batch_bench -- --steps 64 --sizes 1,2,4,8,16
//!
//! For each bus size (1, 2, 4, 8 users at once) it writes the same number of tokens
//! for every user and prints:
//!   total speed     = tokens per second for everyone together  (should go UP with batching)
//!   per-user speed  = how fast one person sees their answer    (goes down a little)

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;

const PROMPT: &str = "<|im_start|>system\nYou are kinSin, a helpful assistant.<|im_end|>\n\
<|im_start|>user\nExplain how a CPU runs a program, step by step.<|im_end|>\n<|im_start|>assistant\n";

fn main() -> Result<()> {
    let mut model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string();
    let mut tokenizer_path = "models/tokenizer.json".to_string();
    let mut steps = 32usize;
    let mut sizes = vec![1usize, 2, 4, 8];

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args.get(i + 1).cloned().with_context(|| format!("{flag} needs a value"))?;
        match flag {
            "--model" => model_path = value,
            "--tokenizer" => tokenizer_path = value,
            "--steps" => steps = value.parse().context("--steps must be a number")?,
            "--sizes" => {
                sizes = value
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()
                    .context("--sizes must look like 1,2,4,8")?
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
        i += 2;
    }
    anyhow::ensure!(steps > 0, "--steps must be at least 1");
    anyhow::ensure!(!sizes.is_empty() && !sizes.contains(&0), "every batch size must be at least 1");

    let device = Device::Cpu;
    println!("AVX enabled: {}", candle_core::utils::with_avx());
    println!("Loading {model_path} ...");
    let mut file = std::fs::File::open(&model_path).with_context(|| format!("could not open model file '{model_path}'"))?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let prompt: Vec<u32> = tokenizer.encode(PROMPT, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    println!("Prompt: {} tokens. Writing {steps} tokens per user.\n", prompt.len());

    // Warm-up: the first runs are slow (memory gets paged in, the CPU clocks up, buffers
    // for each batch size get allocated). One short run per size, so none of it is counted.
    print!("Warming up ...");
    std::io::stdout().flush()?;
    run(&mut model, &prompt, 1, steps, &device)?;
    for &users in &sizes {
        run(&mut model, &prompt, users, steps.min(8), &device)?;
    }
    println!(" done\n");

    println!("{:>6} | {:>13} | {:>15} | {:>9}", "users", "total speed", "per-user speed", "vs 1 user");
    println!("{}", "-".repeat(53));
    let mut single = None;
    for &users in &sizes {
        print!("{users:>6} | ");
        std::io::stdout().flush()?;
        let seconds = run(&mut model, &prompt, users, steps, &device)?;
        let total = (users * steps) as f64 / seconds;
        let per_user = steps as f64 / seconds;
        let base = *single.get_or_insert(total);   // the first size in --sizes is the baseline (1 by default)
        println!("{:>7.1} tok/s | {:>9.1} tok/s | {:>8.2}x", total, per_user, total / base);
    }
    println!("\n\"vs 1 user\" above 1.0 = batching serves more people with the same CPU.");
    Ok(())
}

/// Read the prompt for `users` people at once, then write `steps` tokens for all of them.
/// Returns the seconds spent writing (the part users wait for, token after token).
fn run(model: &mut ModelWeights, prompt: &[u32], users: usize, steps: usize, device: &Device) -> Result<f64> {
    model.clear_kv_cache();
    // Everyone on the bus gets the same prompt, so all rows have the same length.
    // The prompt goes in one token at a time as a (users, 1) tensor: Candle 0.11's
    // quantized_qwen2 can't read a whole multi-row prompt at once (its last-position slice
    // isn't contiguous when users > 1, and the quantized matmul refuses it).
    // This part isn't timed, so being slower here doesn't change the results.
    let mut logits = None;
    for (pos, &token) in prompt.iter().enumerate() {
        let column = Tensor::new(&[token], device)?.unsqueeze(0)?.repeat((users, 1))?; // (users, 1)
        logits = Some(model.forward(&column, pos)?);
    }
    let logits = logits.context("the prompt is empty")?; // (users, vocab)
    let mut next = logits.argmax(candle_core::D::Minus1)?.unsqueeze(1)?; // (users, 1)

    let start = Instant::now();
    for step in 0..steps {
        let logits = model.forward(&next, prompt.len() + step)?;
        next = logits.argmax(candle_core::D::Minus1)?.unsqueeze(1)?;
    }
    // Make sure the last step really finished before stopping the clock
    let _ = next.to_vec2::<u32>()?;
    Ok(start.elapsed().as_secs_f64())
}
