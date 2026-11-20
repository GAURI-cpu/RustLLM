//! Does MY engine give the same word scores as Candle? (Step 4 check)
//!   cargo run --release --example compare_model -- models/kinsin-1.5b-v2-q4_0.gguf

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use kinsin_engine::model::Model;

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: compare_model <model.gguf>");
    let tokens: Vec<u32> = vec![9707, 11, 358, 1079, 264]; // "Hello, I am a" as Qwen word numbers

    // Candle: the answer key
    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut candle = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)?;

    // Mine
    let started = std::time::Instant::now();
    let mut mine = Model::load(&path)?;
    println!("Loaded my engine in {:.1}s", started.elapsed().as_secs_f32());

    // 1. The prompt: Candle reads it all at once, mine word by word
    let input = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let mut their_scores = candle.forward(&input, 0)?.squeeze(0)?.to_vec1::<f32>()?;
    let started = std::time::Instant::now();
    let mut my_scores = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        my_scores = mine.forward(t, pos);
    }
    println!("My engine read {} words in {:.1}s", tokens.len(), started.elapsed().as_secs_f32());

    // 2. Then pick the next word 5 times, comparing at every step
    let mut pos = tokens.len();
    for step in 0..6 {
        let biggest = my_scores.iter().zip(&their_scores).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let (m, t) = (argmax(&my_scores), argmax(&their_scores));
        println!(
            "step {step}: my next word {m:>6}   Candle's {t:>6}   {}   max score difference {biggest:.3}",
            if m == t { "same ✓" } else { "DIFFERENT ✗" }
        );
        if step == 5 {
            break;
        }
        let next = t as u32; // both continue with the same word, so they stay comparable
        my_scores = mine.forward(next, pos);
        let input = Tensor::new(&[next], &Device::Cpu)?.unsqueeze(0)?;
        their_scores = candle.forward(&input, pos)?.squeeze(0)?.to_vec1::<f32>()?;
        pos += 1;
    }
    Ok(())
}