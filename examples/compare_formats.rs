//! Does MY engine agree with Candle on a model that uses many storage types?
//!   python scripts/make_test_model.py models/test-mix.gguf mix
//!   cargo run --release --example compare_formats -- models/test-mix.gguf
//! Works on any Qwen2 GGUF. Word numbers are given directly, so no tokenizer is needed.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use kinsin_engine::model::Model;

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: compare_formats <model.gguf>");

    // Which storage types does this file use?
    let g = kinsin_engine::gguf::Gguf::open(&path)?;
    let mut kinds: Vec<String> = g.tensors.iter().map(|t| kinsin_engine::quant::kind_name(t.kind)).collect();
    kinds.sort();
    kinds.dedup();
    println!("Storage types in this file: {}", kinds.join(", "));

    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut candle = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)?;
    let mut mine = Model::load(&path)?;

    let prompt: Vec<u32> = vec![1, 17, 42, 99, 5, 230];
    let input = Tensor::new(prompt.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let mut theirs = candle.forward(&input, 0)?.squeeze(0)?.to_vec1::<f32>()?;
    let mut ours = mine.forward_prompt(&prompt, 0);

    let mut pos = prompt.len();
    let mut same = 0;
    let steps = 8;
    for step in 0..steps {
        let biggest = theirs.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let diff = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let (a, b) = (argmax(&ours), argmax(&theirs));
        if a == b {
            same += 1;
        }
        println!(
            "step {step}: next word mine {a:>4} · Candle {b:>4} {}  (biggest score {biggest:.3}, biggest difference {diff:.5})",
            if a == b { "✓" } else { "✗" }
        );
        // Both continue with Candle's choice, so they always see the same words
        let input = Tensor::new(&[b as u32], &Device::Cpu)?.unsqueeze(0)?;
        theirs = candle.forward(&input, pos)?.squeeze(0)?.to_vec1::<f32>()?;
        ours = mine.forward(b as u32, pos);
        pos += 1;
    }
    println!("Same next word {same}/{steps} times");
    Ok(())
}