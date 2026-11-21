//! A FAIR speed race: my engine vs Candle, same model, same words, same conditions.
//!   cargo run --release --example race -- models/kinsin-1.5b-v2-q4_0.gguf
//!
//! - Both read the same 35-word prompt, then write the same 32 words.
//! - They take turns (mine, Candle, mine, Candle...), so heat and background apps hit both equally.
//! - 5 rounds each; we report the MEDIAN (the middle result), which ignores one-off hiccups.
//! - Only the maths is timed: no text decoding, no printing.

use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use kinsin_engine::model::Model;

const ROUNDS: usize = 5;
const WRITE: usize = 32;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Returns (prompt seconds, words per second while writing)
fn run_mine(model: &mut Model, prompt: &[u32]) -> (f64, f64) {
    model.reset();
    let t = Instant::now();
    let mut scores = model.forward_prompt(prompt, 0); // the whole prompt in one batch
    let read = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut pos = prompt.len();
    for _ in 0..WRITE {
        let next = argmax(&scores);
        scores = model.forward(next, pos);
        pos += 1;
    }
    (read, WRITE as f64 / t.elapsed().as_secs_f64())
}

fn run_candle(model: &mut ModelWeights, prompt: &[u32]) -> anyhow::Result<(f64, f64)> {
    model.clear_kv_cache();
    let t = Instant::now();
    let input = Tensor::new(prompt, &Device::Cpu)?.unsqueeze(0)?;
    let mut scores = model.forward(&input, 0)?.squeeze(0)?.to_vec1::<f32>()?;
    let read = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut pos = prompt.len();
    for _ in 0..WRITE {
        let next = argmax(&scores);
        let input = Tensor::new(&[next], &Device::Cpu)?.unsqueeze(0)?;
        scores = model.forward(&input, pos)?.squeeze(0)?.to_vec1::<f32>()?;
        pos += 1;
    }
    Ok((read, WRITE as f64 / t.elapsed().as_secs_f64()))
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: race <model.gguf>");
    // 35 word numbers in ChatML shape (the same for both engines)
    let prompt: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 595, 258, 50, 258, 11, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
        840, 20772, 1246, 264, 13940, 8473, 264, 2025, 11, 3019, 553, 3019, 13, 151645, 198, 151644,
    ];

    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut candle = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)?;
    let mut mine = Model::load(&path)?;

    println!("Warming up both engines...");
    run_mine(&mut mine, &prompt);
    run_candle(&mut candle, &prompt)?;

    let (mut my_read, mut my_write, mut c_read, mut c_write) = (vec![], vec![], vec![], vec![]);
    for round in 1..=ROUNDS {
        let (r, w) = run_mine(&mut mine, &prompt);
        my_read.push(r);
        my_write.push(w);
        let (r2, w2) = run_candle(&mut candle, &prompt)?;
        c_read.push(r2);
        c_write.push(w2);
        println!("round {round}: mine {w:5.1} words/s   Candle {w2:5.1} words/s");
    }

    let (mw, cw) = (median(my_write), median(c_write));
    let (mr, cr) = (median(my_read), median(c_read));
    println!("\nMEDIAN of {ROUNDS} rounds ({} prompt words, {WRITE} written words):", prompt.len());
    println!("                    my engine     Candle");
    println!("  reading prompt     {mr:7.2} s   {cr:7.2} s");
    println!("  writing           {mw:6.1} w/s  {cw:6.1} w/s");
    println!("\n  Writing: my engine is {:.2}x Candle's speed", mw / cw);
    Ok(())
}