//! RustLLM: CPU-only LLM inference in Rust.
//!   cargo run --release                         chat in the terminal
//!   cargo run --release -- --serve              start the HTTP server on port 8080
//!   cargo run --release -- --serve --port 3001  another port
//!   cargo run --release -- --model models/other-model.gguf

mod chat;
mod engine;
mod sampler;
mod server;

use std::io::Write;

use anyhow::Result;

use chat::{Message, Role, build_prompt};
use engine::Engine;
use sampler::SamplingOptions;

fn main() -> Result<()> {
    // Defaults, which can be changed from the command line
    let mut model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string();
    let mut tokenizer_path = "models/tokenizer.json".to_string();
    let mut serve = false;
    let mut port: u16 = 8080;

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
            other => anyhow::bail!("unknown option: {other}"),
        }
        i += 1;
    }

    println!("AVX enabled: {}", candle_core::utils::with_avx());
    println!("Loading {} ...", model_path);
    let mut engine = Engine::load(&model_path, &tokenizer_path)?;

    if serve {
        // Start the async runtime and run the server until Ctrl+C
        let runtime = tokio::runtime::Runtime::new()?;
        return runtime.block_on(server::serve(engine, model_path, port));
    }

    let opts = SamplingOptions::default();

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
        let prompt = build_prompt(&history);

        print!("bot › ");
        let (reply, stats) = engine.generate(&prompt, &opts, 500, |piece| {
            print!("{}", piece);
            let _ = std::io::stdout().flush();
            true // keep going
        })?;
        println!(
            "\n[{} prompt tokens | {} new tokens @ {:.1} tok/s]\n",
            stats.prompt_tokens, stats.generated_tokens, stats.tokens_per_sec
        );

        history.push(Message { role: Role::Assistant, text: reply });
    }

    Ok(())
}