//!   GET  /                      -> the chat web page
//!   GET  /health                -> "ok"
//!   GET  /v1/models             -> which model is loaded
//!   POST /v1/chat/completions   -> a reply, streamed or all at once

use std::sync::mpsc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

use crate::chat::{Message, Role, build_prompt};
use crate::engine::{Engine, Stats};
use crate::sampler::SamplingOptions;

// ---------- What the client sends (same shape as OpenAI) ----------

#[derive(Deserialize)]
pub struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

// ---------- The queue between the web server and the model ----------

/// One request waiting for the model.
struct Job {
    prompt: String,
    opts: SamplingOptions,
    max_tokens: usize,
    /// Where the worker sends the reply, piece by piece.
    reply_to: tokio::sync::mpsc::UnboundedSender<Piece>,
}

/// What the worker sends back while answering.
enum Piece {
    Text(String),
    Done(Stats),
    Failed(String),
}

/// Shared by every request handler: the entrance to the queue.
#[derive(Clone)]
struct AppState {
    queue: mpsc::Sender<Job>,
    model_name: String,
}

/// The worker: owns the engine and answers jobs one by one, forever.
fn run_worker(mut engine: Engine, jobs: mpsc::Receiver<Job>) {
    for job in jobs {
        let reply_to = job.reply_to;
        let result = engine.generate(&job.prompt, &job.opts, job.max_tokens, |piece| {
            // If sending fails, the client has disconnected: stop generating.
            reply_to.send(Piece::Text(piece.to_string())).is_ok()
        });
        let _ = match result {
            Ok((_, stats)) => reply_to.send(Piece::Done(stats)),
            Err(e) => reply_to.send(Piece::Failed(e.to_string())),
        };
    }
}

// ---------- Starting the server ----------

pub async fn serve(engine: Engine, model_name: String, port: u16) -> anyhow::Result<()> {
    let (queue, jobs) = mpsc::channel::<Job>();
    std::thread::spawn(move || run_worker(engine, jobs));

    let state = AppState { queue, model_name };
    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("../web/index.html")) }))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);
    
    let address = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("Server running on http://localhost:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ---------- GET /v1/models ----------

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": state.model_name, "object": "model", "owned_by": "rustllm" }]
    }))
}

// ---------- POST /v1/chat/completions ----------

async fn chat_completions(State(state): State<AppState>, Json(request): Json<ChatRequest>) -> Response {
    // 1. Turn the request's messages into our own Message type
    let mut history = Vec::new();
    for m in &request.messages {
        match Role::parse(&m.role) {
            Some(role) => history.push(Message { role, text: m.content.clone() }),
            None => return error(StatusCode::BAD_REQUEST, &format!("unknown role '{}'", m.role)),
        }
    }
    if history.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    // 2. Settings: the request's values, or our defaults
    let defaults = SamplingOptions::default();
    let opts = SamplingOptions {
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_k: request.top_k.unwrap_or(defaults.top_k),
        repeat_penalty: defaults.repeat_penalty,
    };
    let max_tokens = request.max_tokens.unwrap_or(512).min(2048);

    // 3. Put the job in the queue
    let (reply_to, pieces) = tokio::sync::mpsc::unbounded_channel();
    let job = Job { prompt: build_prompt(&history), opts, max_tokens, reply_to };
    if state.queue.send(job).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped");
    }

    // 4. Answer: streamed, or all at once
    if request.stream {
        stream_reply(pieces, state.model_name)
    } else {
        full_reply(pieces, state.model_name).await
    }
}

/// Streaming: send each piece as a Server-Sent Event the moment it is ready.
fn stream_reply(pieces: tokio::sync::mpsc::UnboundedReceiver<Piece>, model: String) -> Response {
    let events = UnboundedReceiverStream::new(pieces).map(move |piece| {
        let data = match piece {
            Piece::Text(text) => json!({
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": null }]
            })
            .to_string(),
            // Last chunk carries the finish reason and the stats
            Piece::Done(stats) => json!({
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": usage(&stats)
            })
            .to_string(),
            Piece::Failed(message) => json!({ "error": { "message": message } }).to_string(),
        };
        Ok::<Event, std::convert::Infallible>(Event::default().data(data))
    });
    // When the worker finishes, the channel closes; then send OpenAI's end marker
    let end = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    Sse::new(events.chain(end)).into_response()
}

/// Not streaming: wait for the whole reply, then send one JSON answer.
async fn full_reply(mut pieces: tokio::sync::mpsc::UnboundedReceiver<Piece>, model: String) -> Response {
    let mut content = String::new();
    while let Some(piece) = pieces.recv().await {
        match piece {
            Piece::Text(text) => content.push_str(&text),
            Piece::Done(stats) => {
                return Json(json!({
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": content },
                        "finish_reason": "stop"
                    }],
                    "usage": usage(&stats)
                }))
                .into_response();
            }
            Piece::Failed(message) => return error(StatusCode::INTERNAL_SERVER_ERROR, &message),
        }
    }
    error(StatusCode::INTERNAL_SERVER_ERROR, "model worker stopped")
}

fn usage(stats: &Stats) -> serde_json::Value {
    json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
        "tokens_per_sec": (stats.tokens_per_sec * 10.0).round() / 10.0
    })
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": { "message": message } }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stream_sends_text_then_stats_then_done() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Piece::Text("Hello".to_string())).unwrap();
        tx.send(Piece::Done(Stats { prompt_tokens: 3, generated_tokens: 1, tokens_per_sec: 20.0 })).unwrap();
        drop(tx); // the worker is finished

        let response = stream_reply(rx, "test-model".to_string());
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();

        let events: Vec<&str> = body.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert_eq!(events.len(), 3);
        assert!(events[0].contains(r#""content":"Hello""#));
        assert!(events[1].contains(r#""finish_reason":"stop""#));
        assert_eq!(events[2], "data: [DONE]");
    }
}