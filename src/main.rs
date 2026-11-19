//! RustLLM: CPU-only LLM inference in Rust.
//!   cargo run --release                         chat in the terminal
//!   cargo run --release -- --serve              start the HTTP server on port 8080
//!   cargo run --release -- --serve --port 3001  another port
//!   cargo run --release -- --model models/other-model.gguf
//!   cargo run --release -- --context 8192      allow longer conversations
//!   cargo run --release -- --serve --ocr-models models/ocr   where the photo-reading models are
//!   cargo run --release -- --serve --whisper-model models/whisper   where the speech-to-text model is

mod chat;
mod engine;
mod ocr;
mod sampler;
mod server;
mod voice;

use std::io::Write;
use std::path::Path;

use anyhow::Result;

use chat::{Message, Role, build_prompt_within};
use engine::Engine;
use ocr::Ocr;
use sampler::SamplingOptions;
use voice::Voice;

fn main() -> Result<()> {
    // Defaults, which can be changed from the command line
    let mut model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string();
    let mut tokenizer_path = "models/tokenizer.json".to_string();
    let mut serve = false;
    let mut port: u16 = 8080;
    let mut context: usize = 4096;
    let mut ocr_models = "models/ocr".to_string();
    let mut whisper_model = "models/whisper".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                model_path = args[i + 1].clone();
                i += 1;
            }
            "--tokenizer" if i + 1 < args.len() => {
                tokenizer_path = args[i + 1].clone();
                i += 1;
            }
            "--serve" => serve = true,
            "--port" if i + 1 < args.len() => {
                port = args[i + 1].parse()?;
                i += 1;
            }
            "--context" if i + 1 < args.len() => {
                context = args[i + 1].parse()?;
                i += 1;
            }
            "--ocr-models" if i + 1 < args.len() => {
                ocr_models = args[i + 1].clone();
                i += 1;
            }
            "--whisper-model" if i + 1 < args.len() => {
                whisper_model = args[i + 1].clone();
                i += 1;
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
        i += 1;
    }

    println!("AVX enabled: {}", candle_core::utils::with_avx());
    println!("Loading {} ...", model_path);
    let mut engine = Engine::load(&model_path, &tokenizer_path, context)?;

    if serve {
        let ocr = load_ocr(Path::new(&ocr_models));
        let voice = load_voice(Path::new(&whisper_model));
        // Start the async runtime and run the server until Ctrl+C
        let runtime = tokio::runtime::Runtime::new()?;
        return runtime.block_on(server::serve(engine, model_path, port, ocr, voice));
    }

    let opts = SamplingOptions::default();
    let max_reply_tokens = 500.min(context / 2);

    let mut history: Vec<Message> = Vec::new();
    println!("Chat started. Type /reset to forget the conversation, /exit to quit.\n");

    loop {
        print!("you › ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();

        match input {
            "" => continue,
            "/exit" => break,
            "/reset" => {
                history.clear();
                println!("(conversation cleared)\n");
                continue;
            }
            _ => {}
        }

        history.push(Message { role: Role::User, text: input.to_string() });
        // Keep only as much history as fits, leaving room for the reply
        let prompt = match build_prompt_within(&history, context - max_reply_tokens, |t| engine.count_tokens(t)) {
            Ok(prompt) => prompt,
            Err(e) => {
                println!("error: {e}\n");
                history.pop(); // forget the message that didn't fit
                continue;
            }
        };

        print!("bot › ");
        let (reply, stats) = engine.generate(&prompt, &opts, max_reply_tokens, |piece| {
            print!("{}", piece);
            let _ = std::io::stdout().flush();
            true // keep going
        })?;
        println!(
            "\n[{} prompt tokens | {} new tokens @ {:.1} tok/s | {}]\n",
            stats.prompt_tokens,
            stats.generated_tokens,
            stats.tokens_per_sec,
            stats.finish_reason.as_str()
        );

        history.push(Message { role: Role::Assistant, text: reply });
    }

    Ok(())
}

/// Photo reading is optional: without the OCR models the server still runs, just without it.
fn load_ocr(dir: &Path) -> Option<Ocr> {
    let shown = dir.display();
    if !Ocr::models_present(dir) {
        println!("Photo reading: off (put text-detection.onnx and text-recognition.onnx in {shown})");
        return None;
    }
    match Ocr::load(dir) {
        Ok(ocr) => {
            println!("Photo reading: on ({shown})");
            Some(ocr)
        }
        Err(e) => {
            println!("Photo reading: off ({e:#})");
            None
        }
    }
}

/// Voice input is optional too: without the Whisper files the server runs without it.
fn load_voice(dir: &Path) -> Option<Voice> {
    let shown = dir.display();
    if !Voice::files_present(dir) {
        println!("Voice input: off (put config.json, tokenizer.json and model.safetensors from Whisper in {shown})");
        return None;
    }
    match Voice::load(dir) {
        Ok(voice) => {
            println!("Voice input: on ({shown})");
            Some(voice)
        }
        Err(e) => {
            println!("Voice input: off ({e:#})");
            None
        }
    }
}
