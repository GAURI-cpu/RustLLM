//! HTTP server with an OpenAI-compatible API.
//!
//!   GET  /                      -> the chat web page
//!   GET  /health                -> "ok"
//!   GET  /v1/models             -> which model is loaded
//!   POST /v1/chat/completions   -> a reply, streamed or all at once
//!   POST /v1/ocr                -> the text in a photo (body = the image bytes)
//!
//! The model can only work on one request at a time on a CPU, so requests
//! wait in a queue and a single worker thread answers them in order.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::DefaultBodyLimit;
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

use crate::chat::{Message, Role, build_prompt_within};
use crate::engine::{Engine, Stats};
use crate::ocr::Ocr;
use crate::sampler::SamplingOptions;

// ---------- Limits that keep the server healthy ----------

/// Requests allowed in the system at once (1 being answered + the rest waiting).
const MAX_PENDING: usize = 8;
/// Largest request body accepted, in bytes.
const MAX_BODY_BYTES: usize = 64 * 1024;
/// Largest photo accepted by /v1/ocr, in bytes.
const MAX_PHOTO_BYTES: usize = 12 * 1024 * 1024;
/// Most messages in one conversation.
const MAX_MESSAGES: usize = 200;
/// Longest single message, in characters.
const MAX_MESSAGE_CHARS: usize = 16_000;
/// Default and maximum reply length, in tokens.
const DEFAULT_MAX_TOKENS: usize = 512;
const LIMIT_MAX_TOKENS: usize = 1024;
/// A reply that takes longer than this is cut off.
const GENERATION_TIMEOUT: Duration = Duration::from_secs(120);

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
    /// The conversation; the worker turns it into a prompt that fits the context.
    history: Vec<Message>,
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
    /// How many requests are waiting or being answered right now.
    pending: Arc<AtomicUsize>,
    model_name: String,
    /// Photo reading; None when the OCR models are missing.
    ocr: Option<Arc<Ocr>>,
    /// One photo at a time: OCR uses a lot of CPU.
    ocr_slot: Arc<tokio::sync::Semaphore>,
}

/// The worker: owns the engine and answers jobs one by one, forever.
fn run_worker(mut engine: Engine, jobs: mpsc::Receiver<Job>, pending: Arc<AtomicUsize>) {
    for job in jobs {
        answer(&mut engine, job);
        // This request is finished: make room in the queue
        pending.fetch_sub(1, Ordering::SeqCst);
    }
}

fn answer(engine: &mut Engine, job: Job) {
    let reply_to = job.reply_to;

    // Keep the prompt small enough to leave room for the reply
    let max_tokens = job.max_tokens.min(engine.max_context() / 2);
    let prompt_budget = engine.max_context() - max_tokens;
    let prompt = match build_prompt_within(&job.history, prompt_budget, |text| engine.count_tokens(text)) {
        Ok(prompt) => prompt,
        Err(e) => {
            let _ = reply_to.send(Piece::Failed(e.to_string()));
            return;
        }
    };

    let deadline = Instant::now() + GENERATION_TIMEOUT;
    let result = engine.generate(&prompt, &job.opts, max_tokens, |piece| {
        // Stop if the client disconnected, or if this reply is taking too long
        reply_to.send(Piece::Text(piece.to_string())).is_ok() && Instant::now() < deadline
    });
    let _ = match result {
        Ok((_, stats)) => reply_to.send(Piece::Done(stats)),
        Err(e) => reply_to.send(Piece::Failed(e.to_string())),
    };
}

// ---------- Starting the server ----------

pub async fn serve(engine: Engine, model_name: String, port: u16, ocr: Option<Ocr>) -> anyhow::Result<()> {
    let (queue, jobs) = mpsc::channel::<Job>();
    let pending = Arc::new(AtomicUsize::new(0));
    let worker_pending = pending.clone();
    std::thread::spawn(move || run_worker(engine, jobs, worker_pending));

    let state = AppState {
        queue,
        pending,
        model_name,
        ocr: ocr.map(Arc::new),
        ocr_slot: Arc::new(tokio::sync::Semaphore::new(1)),
    };
    let app = app(state);

    let address = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("Server running on http://localhost:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(|| async { Html(include_str!("../web/index.html")) }))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        // Photos are bigger than chat messages: only this route gets a larger limit
        .route("/v1/ocr", post(read_photo).layer(DefaultBodyLimit::max(MAX_PHOTO_BYTES)))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES)) // reject huge requests early
        .layer(CorsLayer::permissive()) // allow a website on another port to call us
        .with_state(state)
}

// ---------- GET /v1/models ----------

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": state.model_name, "object": "model", "owned_by": "rustllm" }],
        "kinsin": { "ocr": state.ocr.is_some() }
    }))
}

// ---------- POST /v1/ocr ----------

async fn read_photo(State(state): State<AppState>, body: Bytes) -> Response {
    let Some(ocr) = state.ocr.clone() else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "photo reading is not set up on this server (OCR models missing)");
    };
    if body.is_empty() {
        return error(StatusCode::BAD_REQUEST, "send the photo as the request body");
    }
    // Wait for our turn: one photo at a time
    let Ok(permit) = state.ocr_slot.clone().acquire_owned().await else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "photo reading stopped");
    };
    let started = Instant::now();
    // OCR is heavy CPU work: run it on a blocking thread, not on the async runtime.
    // The permit moves in too, so the slot stays taken until OCR really ends (even if the client leaves).
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        ocr.read_text(&body)
    })
    .await;
    match result {
        Ok(Ok(text)) => Json(json!({
            "text": text,
            "chars": text.chars().count(),
            "ms": started.elapsed().as_millis() as u64
        }))
        .into_response(),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e.to_string()),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "photo reading crashed"),
    }
}

// ---------- POST /v1/chat/completions ----------

async fn chat_completions(State(state): State<AppState>, Json(request): Json<ChatRequest>) -> Response {
    // 1. Check the input
    if request.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    if request.messages.len() > MAX_MESSAGES {
        return error(StatusCode::BAD_REQUEST, &format!("at most {MAX_MESSAGES} messages are allowed"));
    }
    let mut history = Vec::new();
    for m in &request.messages {
        let Some(role) = Role::parse(&m.role) else {
            return error(StatusCode::BAD_REQUEST, &format!("unknown role '{}'", m.role));
        };
        if m.content.chars().count() > MAX_MESSAGE_CHARS {
            return error(StatusCode::BAD_REQUEST, &format!("a message can be at most {MAX_MESSAGE_CHARS} characters"));
        }
        history.push(Message { role, text: m.content.clone() });
    }

    // 2. Settings: the request's values (within limits), or our defaults
    let defaults = SamplingOptions::default();
    let opts = SamplingOptions {
        temperature: request.temperature.unwrap_or(defaults.temperature).clamp(0.0, 2.0),
        top_k: request.top_k.unwrap_or(defaults.top_k).clamp(1, 200),
        repeat_penalty: defaults.repeat_penalty,
    };
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS).clamp(1, LIMIT_MAX_TOKENS);

    // 3. Is there room in the queue? If not, say "busy" right away instead of making people wait forever
    if state.pending.fetch_add(1, Ordering::SeqCst) >= MAX_PENDING {
        state.pending.fetch_sub(1, Ordering::SeqCst);
        return error(StatusCode::TOO_MANY_REQUESTS, "kinSin is busy right now, please try again in a moment");
    }

    // 4. Put the job in the queue
    let (reply_to, pieces) = tokio::sync::mpsc::unbounded_channel();
    let job = Job { history, opts, max_tokens, reply_to };
    if state.queue.send(job).is_err() {
        state.pending.fetch_sub(1, Ordering::SeqCst);
        return error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped");
    }

    // 5. Answer: streamed, or all at once
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
                "choices": [{ "index": 0, "delta": {}, "finish_reason": stats.finish_reason.as_str() }],
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
                        "finish_reason": stats.finish_reason.as_str()
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

    fn request(role: &str, content: &str) -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage { role: role.to_string(), content: content.to_string() }],
            stream: false,
            temperature: None,
            top_k: None,
            max_tokens: None,
        }
    }

    fn state_with_pending(n: usize) -> AppState {
        let (queue, _jobs) = mpsc::channel();
        AppState {
            queue,
            pending: Arc::new(AtomicUsize::new(n)),
            model_name: "test".to_string(),
            ocr: None,
            ocr_slot: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    #[tokio::test]
    async fn full_queue_answers_429() {
        let response = chat_completions(State(state_with_pending(MAX_PENDING)), Json(request("user", "hi"))).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn unknown_role_answers_400() {
        let response = chat_completions(State(state_with_pending(0)), Json(request("admin", "hi"))).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn too_long_message_answers_400() {
        let long = "x".repeat(MAX_MESSAGE_CHARS + 1);
        let response = chat_completions(State(state_with_pending(0)), Json(request("user", &long))).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stream_sends_text_then_stats_then_done() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Piece::Text("Hello".to_string())).unwrap();
        tx.send(Piece::Done(Stats {
            prompt_tokens: 3,
            generated_tokens: 1,
            tokens_per_sec: 20.0,
            finish_reason: crate::engine::FinishReason::Stop,
        }))
        .unwrap();
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

    #[tokio::test]
    async fn photo_without_ocr_models_answers_503() {
        let response = read_photo(State(state_with_pending(0)), Bytes::from_static(b"some photo bytes")).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn models_say_whether_photo_reading_is_on() {
        let Json(body) = list_models(State(state_with_pending(0))).await;
        assert_eq!(body["kinsin"]["ocr"], false);
    }

    /// Send a POST with a body of `size` bytes through the real router; return the status.
    async fn post_size(path: &str, size: usize) -> StatusCode {
        use tower::ServiceExt;
        let request = axum::http::Request::post(path)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(vec![b' '; size]))
            .unwrap();
        app(state_with_pending(0)).oneshot(request).await.unwrap().status()
    }

    #[tokio::test]
    async fn photo_route_accepts_big_bodies_but_others_keep_64kb_limit() {
        // A 5 MB photo gets past the size limit (and then hits "no OCR models" = 503)
        assert_eq!(post_size("/v1/ocr", 5 * 1024 * 1024).await, StatusCode::SERVICE_UNAVAILABLE);
        // Over 12 MB is refused
        assert_eq!(post_size("/v1/ocr", MAX_PHOTO_BYTES + 1).await, StatusCode::PAYLOAD_TOO_LARGE);
        // Chat requests still stop at 64 KB
        assert_eq!(post_size("/v1/chat/completions", MAX_BODY_BYTES + 1).await, StatusCode::PAYLOAD_TOO_LARGE);
    }
}
