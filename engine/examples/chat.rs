//! Chat with kinSin running on MY OWN engine (no Candle).
//!   cargo run --release -p kinsin-engine --example chat -- models/kinsin-1.5b-v2-q4_0.gguf models/tokenizer.json

use std::io::Write;
use std::time::Instant;

use anyhow::Context;
use kinsin_engine::model::Model;
use tokenizers::Tokenizer;

/// The word with the highest score.
fn argmax(scores: &[f32]) -> u32 {
    scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or("models/kinsin-1.5b-v2-q4_0.gguf".into());
    let tokenizer_path = args.next().unwrap_or("models/tokenizer.json".into());

    // Text <-> word numbers
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let end = tokenizer.token_to_id("<|im_end|>").context("tokenizer has no <|im_end|>")?;

    println!("Loading {model_path} into MY engine...");
    let started = Instant::now();
    let mut model = Model::load(&model_path)?;
    println!("Loaded in {:.1}s. Type a message (empty line to quit).", started.elapsed().as_secs_f32());

    let mut pos = 0; // how many words the model has seen (the KV cache grows with it)
    let mut next_input =
        "<|im_start|>system\nYou are kinSin, a helpful assistant created by Gauri (she/her).<|im_end|>\n".to_string();

    loop {
        print!("\nyou > ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() {
            break;
        }

        // 1. Read the new message (ChatML format), word by word
        next_input += &format!("<|im_start|>user\n{line}<|im_end|>\n<|im_start|>assistant\n");
        let ids = tokenizer.encode(next_input.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        next_input.clear();
        let reading = Instant::now();
        let mut scores = model.forward_prompt(&ids, pos); // the whole message in one batch
        pos += ids.len();
        let read_time = reading.elapsed().as_secs_f32();

        // 2. Write the answer: pick the best word, feed it back, repeat
        print!("kinSin > ");
        let writing = Instant::now();
        let mut reply: Vec<u32> = Vec::new();
        let mut shown = 0; // how much of the reply text is already on screen
        loop {
            let next = argmax(&scores);
            if next == end || reply.len() >= 300 {
                break;
            }
            reply.push(next);
            // Show the new text (wait if a letter is only half-finished)
            let text = tokenizer.decode(&reply, true).map_err(anyhow::Error::msg)?;
            if !text.ends_with('\u{FFFD}') && text.len() > shown && text.is_char_boundary(shown) {
                print!("{}", &text[shown..]);
                std::io::stdout().flush()?;
                shown = text.len();
            }
            scores = model.forward(next, pos);
            pos += 1;
        }
        // Close kinSin's turn so the next message continues the same conversation
        model.forward(end, pos);
        pos += 1;
        next_input.push('\n');

        let write_time = writing.elapsed().as_secs_f32();
        println!(
            "\n  [read {} words in {read_time:.1}s · wrote {} words at {:.2} words/s]",
            ids.len(),
            reply.len(),
            reply.len() as f32 / write_time
        );
    }
    Ok(())
}