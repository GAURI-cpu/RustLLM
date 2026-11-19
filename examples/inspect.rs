use anyhow::Result;
use candle_core::quantized::gguf_file;

fn main() -> Result<()> {
    let path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    // TODO 1 + 2: open the file and read its header
    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;

    // TODO 3: what kind of model is this?
    let arch = content.metadata.get("general.architecture").unwrap().to_string()?;
    println!("Architecture:   {}", arch);

    // TODO 4: how big is it?
    let layers = content.metadata.get("qwen2.block_count").unwrap().to_u32()?;
    let context = content.metadata.get("qwen2.context_length").unwrap().to_u32()?;
    println!("Layers:         {}", layers);
    println!("Context length: {}", context);

    // TODO 5: how many weight blocks?
    println!("Tensors:        {}", content.tensor_infos.len());

    // BONUS: list tensor names in alphabetical order
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();

    println!("\nLayer 0 contains:");
    for name in names.iter().filter(|n| n.starts_with("blk.0.")) {
        let info = &content.tensor_infos[*name];
        println!("  {:<28} {:<14} {:?}", name, format!("{:?}", info.shape.dims()), info.ggml_dtype);
    }

    println!("\nOutside the layers:");
    for name in names.iter().filter(|n| !n.starts_with("blk.")) {
        let info = &content.tensor_infos[*name];
        println!("  {:<28} {:<14} {:?}", name, format!("{:?}", info.shape.dims()), info.ggml_dtype);
    }

    Ok(())
}