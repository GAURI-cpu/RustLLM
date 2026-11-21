//! Do MY unpacked weights match Candle's? (Step 2.4 check)
//!   cargo run --release --example compare_weights -- models/kinsin-1.5b-v2-q4_0.gguf

use candle_core::Device;
use candle_core::quantized::gguf_file;
use kinsin_engine::gguf::Gguf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: compare_weights <model.gguf> [tensor names]");
    let mine = Gguf::open(&path)?;

    // Candle's reader, as the answer key
    let mut file = std::fs::File::open(&path)?;
    let candle = gguf_file::Content::read(&mut file)?;

    // Tensors to check: the ones you name, or one of each type
    let mut names: Vec<String> = std::env::args().skip(2).collect();
    if names.is_empty() {
        names = ["output_norm.weight", "blk.0.attn_q.weight", "blk.27.ffn_down.weight", "token_embd.weight"]
            .map(String::from)
            .to_vec();
    }

    for name in names {
        let my_numbers = mine.read_tensor_f32(&name)?;
        let their_numbers = candle
            .tensor(&mut file, &name, &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let biggest_difference = my_numbers
            .iter()
            .zip(&their_numbers)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("{name:<26} {:>11} numbers   max difference {biggest_difference}", my_numbers.len());
    }
    Ok(())
}