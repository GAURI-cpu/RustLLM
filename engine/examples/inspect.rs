//! Look inside a GGUF model file with my own reader.
//!   cargo run --release -p kinsin-engine --example inspect -- models/kinsin-1.5b-v2-q4_0.gguf

use std::collections::BTreeMap;

use kinsin_engine::gguf::{Gguf, Value};

/// GGUF storage types: the number in the file -> a readable name
fn type_name(kind: u32) -> String {
    match kind {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        6 => "Q5_0".into(),
        7 => "Q5_1".into(),
        8 => "Q8_0".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        30 => "BF16".into(),
        other => format!("type {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: inspect <model.gguf>");
    let model = Gguf::open(&path)?;
    let arch = model.get_str("general.architecture").unwrap_or("?").to_string();
    // Settings are named "<architecture>.<setting>", e.g. "qwen2.block_count"
    let setting = |name: &str| model.get_u32(&format!("{arch}.{name}")).map(|v| v.to_string()).unwrap_or("?".into());

    println!("File:            {path}");
    println!("GGUF version:    {}", model.version);
    println!("Architecture:    {arch}");
    println!("Name:            {}", model.get_str("general.name").unwrap_or("?"));
    println!("Layers:          {}", setting("block_count"));
    println!("Hidden size:     {}", setting("embedding_length"));
    println!("Attention heads: {} (key/value heads: {})", setting("attention.head_count"), setting("attention.head_count_kv"));
    println!("Context length:  {}", setting("context_length"));
    if let Some(Value::Array(words)) = model.metadata.get("tokenizer.ggml.tokens") {
        println!("Vocabulary:      {} tokens", words.len());
    }
    println!("Info entries:    {}", model.metadata.len());
    println!("Tensors:         {}", model.tensors.len());
    println!("Weights start at byte {}", model.data_start);

    println!("\nFirst 10 tensors:");
    for t in model.tensors.iter().take(10) {
        println!("  {:<28} {:>14}  {}", t.name, format!("{:?}", t.dims), type_name(t.kind));
    }

    // How many tensors use each storage type (Step 2 must unpack all of these)
    println!("\nTensors per type:");
    let mut per_type: BTreeMap<String, usize> = BTreeMap::new();
    for t in &model.tensors {
        *per_type.entry(type_name(t.kind)).or_default() += 1;
    }
    for (kind, count) in per_type {
        println!("  {kind:<6} {count}");
    }

    // Each tensor holds (dim1 × dim2 × …) numbers; add them all up
    let numbers: u64 = model.tensors.iter().map(|t| t.dims.iter().product::<u64>()).sum();
    println!("\nTotal weights:   {:.2} billion numbers", numbers as f64 / 1e9);
    Ok(())
}