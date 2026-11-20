//! HTTP server with an OpenAI-compatible API.
//!
//!   GET  /                      -> the chat web page
//!   GET  /health                -> "ok"
//!   GET  /v1/models             -> which model is loaded (+ whether photo reading is on)
//!   POST /v1/ocr                -> read the text in a photo (body = the image file)
//!   POST /v1/audio/transcriptions -> speech to text with Whisper (form with a "file" WAV, like OpenAI)
//!   POST /v1/chat/completions   -> a reply, streamed or all at once
//!   POST /v1/auth/signup        -> make an account   {name, email, password}  (sets the login cookie)
//!   POST /v1/auth/login         -> log in            {email, password}        (sets the login cookie)
//!   POST /v1/auth/logout        -> log out
//!   GET  /v1/auth/me            -> who is logged in (401 if nobody)
//!   GET  /v1/chats              -> the logged-in user's saved chats
//!   PUT  /v1/chats              -> replace the logged-in user's saved chats (a JSON array)
//!
//! The model can only work on one request at a time on a CPU, so requests
//! wait in a queue and a single worker thread answers them in order.
//! Fair-use limits (see limits.rs) decide who may join the queue, and while a
//! streamed question waits, the client is told its place in line.

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart};
use axum::extract::{ConnectInfo, State};
use axum::http::header::{CONTENT_TYPE, COOKIE, SET_COOKIE};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

use crate::accounts::{Accounts, AuthError, SESSION_DAYS, User};
use crate::chat::{Message, Role, build_prompt_within};
use crate::engine::{Engine, Stats};
use crate::limits::{Limits, Pass, Who};
use crate::ocr::Ocr;
use crate::voice::{self, Voice};
use crate::sampler::SamplingOptions;

// ---------- Limits that keep the server healthy ----------

/// Requests allowed in the system at once (1 being answered + the rest waiting).
const MAX_PENDING: usize = 8;
/// Largest request body accepted, in bytes.
const MAX_BODY_BYTES: usize = 64 * 1024;
/// Most messages in one conversation.
const MAX_MESSAGES: usize = 200;
/// Longest single message, in characters.
const MAX_MESSAGE_CHARS: usize = 16_000;
/// Default and maximum reply length, in tokens.
const DEFAULT_MAX_TOKENS: usize = 512;
const LIMIT_MAX_TOKENS: usize = 1024;
/// A reply that takes longer than this is cut off.
/// Largest photo accepted by /v1/ocr, in bytes.
const MAX_PHOTO_BYTES: usize = 12 * 1024 * 1024;
/// Largest recording accepted (2 minutes of 16 kHz mono WAV is about 4 MB).
const MAX_AUDIO_BYTES: usize = 16 * 1024 * 1024;
/// Largest chat list a user can save on the server.
const MAX_CHATS_BYTES: usize = 4 * 1024 * 1024;
/// Name of the login cookie.
const SESSION_COOKIE: &str = "kinsin_session";
const GENERATION_TIMEOUT: Duration = Duration::from_secs(120);
/// How often a waiting streamed request checks its place in the queue.
const QUEUE_UPDATE_EVERY: Duration = Duration::from_millis(700);

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
    /// Its ticket number: jobs are numbered 1, 2, 3, ... in the order they join the queue.
    number: u64,
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
    /// Photo reading; None if the OCR models aren't installed.
    ocr: Option<Arc<Ocr>>,
    /// Only one photo is read at a time, so OCR can't starve the chat of CPU.
    ocr_slot: Arc<tokio::sync::Semaphore>,
    /// Speech to text; None if the Whisper model isn't installed. The Mutex means one recording at a time.
    voice: Option<Arc<std::sync::Mutex<Voice>>>,
    /// Users, logins and saved chats.
    accounts: Arc<Accounts>,
    /// Add "Secure" to the login cookie (turn on when the site is served over https).
    secure_cookies: bool,
    /// Fair use: one question at a time, and so many an hour.
    limits: Arc<Limits>,
    /// Believe the proxy's client-IP headers (only safe behind a proxy that sets them).
    trust_proxy: bool,
    /// The last ticket number handed out.
    last_number: Arc<AtomicU64>,
    /// The ticket number of the job the worker is on (or last picked up).
    now_serving: Arc<AtomicU64>,
}

/// The worker: owns the engine and answers jobs one by one, forever.
fn run_worker(mut engine: Engine, jobs: mpsc::Receiver<Job>, pending: Arc<AtomicUsize>, now_serving: Arc<AtomicU64>) {
    for job in jobs {
        now_serving.store(job.number, Ordering::SeqCst);
        // Nobody is waiting for this one any more (closed the tab, pressed stop): skip it
        if !job.reply_to.is_closed() {
            answer(&mut engine, job);
        }
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

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    engine: Engine,
    model_name: String,
    port: u16,
    ocr: Option<Ocr>,
    voice: Option<Voice>,
    accounts: Accounts,
    secure_cookies: bool,
    limits: Limits,
    trust_proxy: bool,
) -> anyhow::Result<()> {
    let (queue, jobs) = mpsc::channel::<Job>();
    let pending = Arc::new(AtomicUsize::new(0));
    let now_serving = Arc::new(AtomicU64::new(0));
    let (worker_pending, worker_serving) = (pending.clone(), now_serving.clone());
    std::thread::spawn(move || run_worker(engine, jobs, worker_pending, worker_serving));

    let state = AppState {
        queue,
        pending,
        model_name,
        ocr: ocr.map(Arc::new),
        ocr_slot: Arc::new(tokio::sync::Semaphore::new(1)),
        voice: voice.map(|v| Arc::new(std::sync::Mutex::new(v))),
        accounts: Arc::new(accounts),
        secure_cookies,
        limits: Arc::new(limits),
        trust_proxy,
        last_number: Arc::new(AtomicU64::new(0)),
        now_serving,
    };
    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("../web/index.html")) }))
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        // photos are bigger than chat messages, so this one route gets its own size limit
        .route("/v1/ocr", post(read_photo).layer(DefaultBodyLimit::max(MAX_PHOTO_BYTES)))
        .route("/v1/audio/transcriptions", post(transcribe).layer(DefaultBodyLimit::max(MAX_AUDIO_BYTES)))
        .route("/v1/auth/signup", post(signup))
        .route("/v1/auth/login", post(login))
        .route("/v1/auth/logout", post(logout))
        .route("/v1/auth/me", get(me))
        .route("/v1/chats", get(load_chats).put(save_chats).layer(DefaultBodyLimit::max(MAX_CHATS_BYTES)))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES)) // reject huge requests early
        .layer(CorsLayer::permissive()) // allow a website on another port to call us
        .with_state(state);

    let address = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("Server running on http://localhost:{port}");
    // ConnectInfo gives handlers the caller's address (for guests' fair-use limits)
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

// ---------- GET /v1/models ----------

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": state.model_name, "object": "model", "owned_by": "rustllm" }],
        "kinsin": { "ocr": state.ocr.is_some(), "voice": state.voice.is_some() }
    }))
}

// ---------- POST /v1/ocr ----------

async fn read_photo(State(state): State<AppState>, photo: Bytes) -> Response {
    let Some(ocr) = state.ocr.clone() else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "photo reading is not set up on this server (OCR models missing)");
    };
    if photo.is_empty() {
        return error(StatusCode::BAD_REQUEST, "no photo was sent");
    }
    // Wait for our turn, then read the photo on a separate thread (OCR is heavy CPU work)
    let Ok(_slot) = state.ocr_slot.acquire().await else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "photo reading is unavailable");
    };
    let started = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || ocr.read_text(&photo)).await;
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

// ---------- POST /v1/audio/transcriptions ----------

/// Same request shape as OpenAI's: a multipart form with "file" (the recording) and an optional "language".
async fn transcribe(State(state): State<AppState>, mut form: Multipart) -> Response {
    let Some(voice) = state.voice.clone() else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "voice input is not set up on this server (Whisper model missing)");
    };
    let mut audio: Option<Bytes> = None;
    let mut language: Option<String> = None;
    loop {
        match form.next_field().await {
            Ok(Some(field)) => match field.name() {
                Some("file") => match field.bytes().await {
                    Ok(bytes) => audio = Some(bytes),
                    Err(e) => return error(StatusCode::BAD_REQUEST, &format!("could not read the recording: {e}")),
                },
                Some("language") => {
                    let value = field.text().await.unwrap_or_default().trim().to_lowercase();
                    if !value.is_empty() && value != "auto" {
                        language = Some(value);
                    }
                }
                _ => {}
            },
            Ok(None) => break,
            Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad form data: {e}")),
        }
    }
    let Some(audio) = audio else {
        return error(StatusCode::BAD_REQUEST, "no recording was sent (form field \"file\")");
    };

    let started = std::time::Instant::now();
    // Whisper is heavy CPU work: run it on a separate thread
    let result = tokio::task::spawn_blocking(move || {
        let samples = voice::wav_to_samples(&audio)?;
        let seconds = samples.len() as f64 / 16000.0;
        let mut voice = voice.lock().map_err(|_| anyhow::anyhow!("voice model is unavailable"))?;
        let (text, language) = voice.transcribe(&samples, language.as_deref())?;
        Ok::<_, anyhow::Error>((text, language, seconds))
    })
    .await;
    match result {
        Ok(Ok((text, language, seconds))) => Json(json!({
            "text": text,
            "language": language,
            "duration": (seconds * 10.0).round() / 10.0,
            "ms": started.elapsed().as_millis() as u64
        }))
        .into_response(),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e.to_string()),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "voice input crashed"),
    }
}

// ---------- Accounts: /v1/auth/... and /v1/chats ----------

#[derive(Deserialize)]
struct SignupRequest {
    name: String,
    email: String,
    password: String,
}

#[derive(Deserialize)]
struct LoginRequest {
    email: String,
    password: String,
}

async fn signup(State(state): State<AppState>, Json(request): Json<SignupRequest>) -> Response {
    let accounts = state.accounts.clone();
    // Hashing a password is slow on purpose, so it runs off the async threads
    let result = tokio::task::spawn_blocking(move || accounts.signup(&request.name, &request.email, &request.password)).await;
    logged_in(result, state.secure_cookies)
}

async fn login(State(state): State<AppState>, Json(request): Json<LoginRequest>) -> Response {
    let accounts = state.accounts.clone();
    let result = tokio::task::spawn_blocking(move || accounts.login(&request.email, &request.password)).await;
    logged_in(result, state.secure_cookies)
}

/// Success: give the browser its login cookie. HttpOnly = page scripts can't read it,
/// SameSite=Lax = other websites can't use it to send requests as you.
fn logged_in(result: Result<Result<(User, String), AuthError>, tokio::task::JoinError>, secure: bool) -> Response {
    match result {
        Ok(Ok((user, token))) => {
            let cookie = format!(
                "{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={}{}",
                SESSION_DAYS * 24 * 60 * 60,
                if secure { "; Secure" } else { "" }
            );
            ([(SET_COOKIE, cookie)], Json(json!({ "user": user }))).into_response()
        }
        Ok(Err(e)) => auth_error(e),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "login crashed"),
    }
}

async fn logout(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(token) = session_token(&headers) {
        state.accounts.logout(&token);
    }
    let cookie = format!("{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0");
    ([(SET_COOKIE, cookie)], Json(json!({ "ok": true }))).into_response()
}

async fn me(State(state): State<AppState>, headers: HeaderMap) -> Response {
    match current_user(&state, &headers) {
        Some(user) => Json(json!({ "user": user })).into_response(),
        None => error(StatusCode::UNAUTHORIZED, "not logged in"),
    }
}

async fn load_chats(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let Some(user) = current_user(&state, &headers) else {
        return error(StatusCode::UNAUTHORIZED, "log in to load saved chats");
    };
    match state.accounts.load_chats(user.id) {
        Ok(json) => ([(CONTENT_TYPE, "application/json")], json).into_response(),
        Err(e) => auth_error(e),
    }
}

async fn save_chats(State(state): State<AppState>, headers: HeaderMap, Json(chats): Json<serde_json::Value>) -> Response {
    let Some(user) = current_user(&state, &headers) else {
        return error(StatusCode::UNAUTHORIZED, "log in to save chats");
    };
    if !chats.is_array() {
        return error(StatusCode::BAD_REQUEST, "chats must be a JSON array");
    }
    match state.accounts.save_chats(user.id, &chats.to_string()) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => auth_error(e),
    }
}

/// The login token from the request's cookie, if it has a well-formed one.
fn session_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all(COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().strip_prefix(SESSION_COOKIE)?.strip_prefix('='))
        .find(|token| token.len() == 64 && token.chars().all(|c| c.is_ascii_hexdigit()))
        .map(String::from)
}

fn current_user(state: &AppState, headers: &HeaderMap) -> Option<User> {
    state.accounts.user_for_token(&session_token(headers)?)
}

fn auth_error(e: AuthError) -> Response {
    let status = match &e {
        AuthError::BadInput(_) => StatusCode::BAD_REQUEST,
        AuthError::EmailTaken => StatusCode::CONFLICT,
        AuthError::WrongLogin => StatusCode::UNAUTHORIZED,
        AuthError::TooManyTries => StatusCode::TOO_MANY_REQUESTS,
        AuthError::Internal(details) => {
            eprintln!("accounts error: {details}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    error(status, &e.to_string())
}

// ---------- POST /v1/chat/completions ----------

async fn chat_completions(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(request): Json<ChatRequest>,
) -> Response {
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

    // 3. Fair use: one question at a time per person, and so many an hour.
    //    The pass is held until the reply is finished (or the client goes away).
    let pass = match state.limits.admit(&visitor(&state, &headers, peer)) {
        Ok(pass) => pass,
        Err(refused) => return error(StatusCode::TOO_MANY_REQUESTS, &refused.to_string()),
    };

    // 4. Is there room in the queue? If not, say "busy" right away instead of making people wait forever
    if state.pending.fetch_add(1, Ordering::SeqCst) >= MAX_PENDING {
        state.pending.fetch_sub(1, Ordering::SeqCst);
        return error(StatusCode::TOO_MANY_REQUESTS, "kinSin is busy right now, please try again in a moment");
    }

    // 5. Put the job in the queue, with the next ticket number
    let (reply_to, pieces) = tokio::sync::mpsc::unbounded_channel();
    let number = state.last_number.fetch_add(1, Ordering::SeqCst) + 1;
    let job = Job { number, history, opts, max_tokens, reply_to };
    if state.queue.send(job).is_err() {
        state.pending.fetch_sub(1, Ordering::SeqCst);
        return error(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped");
    }

    // 6. Answer: streamed, or all at once
    if request.stream {
        let ticket = Ticket { number, now_serving: state.now_serving.clone() };
        stream_reply(pieces, state.model_name, pass, ticket)
    } else {
        let response = full_reply(pieces, state.model_name).await;
        drop(pass);
        response
    }
}

/// Who is asking: the logged-in user, or else the guest's IP address.
fn visitor(state: &AppState, headers: &HeaderMap, peer: SocketAddr) -> Who {
    if let Some(user) = current_user(state, headers) {
        return Who::user(user.id);
    }
    let ip = if state.trust_proxy { proxy_client_ip(headers) } else { None };
    Who::guest(ip.unwrap_or(peer.ip()))
}

/// The real client's IP as told by a proxy in front of us (Cloudflare's header first,
/// then the first address in X-Forwarded-For). Anything that isn't an IP is ignored.
/// Only used with --trust-proxy: otherwise anyone could pretend to be anyone.
fn proxy_client_ip(headers: &HeaderMap) -> Option<IpAddr> {
    let header = |name: &str| headers.get(name).and_then(|value| value.to_str().ok());
    header("cf-connecting-ip")
        .and_then(|value| value.trim().parse().ok())
        .or_else(|| header("x-forwarded-for")?.split(',').next()?.trim().parse().ok())
}

/// A streamed request's place in line: its number, and the number the worker is on.
struct Ticket {
    number: u64,
    now_serving: Arc<AtomicU64>,
}

impl Ticket {
    /// 0 = being answered now; N = the worker is N jobs away from this one.
    fn position(&self) -> u64 {
        self.number.saturating_sub(self.now_serving.load(Ordering::SeqCst))
    }
}

/// Streaming: send each piece as a Server-Sent Event the moment it is ready.
/// Until the first piece arrives, the client is told its place in line whenever it changes.
fn stream_reply(mut pieces: tokio::sync::mpsc::UnboundedReceiver<Piece>, model: String, pass: Pass, ticket: Ticket) -> Response {
    let (events, stream) = tokio::sync::mpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();
    tokio::spawn(async move {
        let _pass = pass; // this person counts as busy until the reply is over
        let mut started = false;
        let mut shown_position = 0; // 0 = nothing shown (yet, or any more)
        let first_check = tokio::time::Instant::now() + QUEUE_UPDATE_EVERY;
        let mut ticks = tokio::time::interval_at(first_check, QUEUE_UPDATE_EVERY);
        loop {
            let data = tokio::select! {
                piece = pieces.recv() => match piece {
                    Some(piece) => {
                        started = true;
                        piece_json(piece, &model)
                    }
                    None => break, // the worker is finished with this reply
                },
                _ = ticks.tick(), if !started => {
                    let position = ticket.position();
                    if position == shown_position {
                        continue;
                    }
                    shown_position = position;
                    json!({
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [],
                        "kinsin": { "queue_position": position }
                    })
                    .to_string()
                }
                // The client went away: stop (dropping `pieces` tells the worker to skip or stop)
                _ = events.closed() => return,
            };
            if events.send(Ok(Event::default().data(data))).is_err() {
                return;
            }
        }
        // The worker is finished: send OpenAI's end marker
        let _ = events.send(Ok(Event::default().data("[DONE]")));
    });
    Sse::new(UnboundedReceiverStream::new(stream)).into_response()
}

/// One streamed chunk, in OpenAI's shape.
fn piece_json(piece: Piece, model: &str) -> String {
    match piece {
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
    }
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

    /// A state, plus the worker's end of the queue (keep it alive to let jobs in).
    fn test_state(pending: usize) -> (AppState, mpsc::Receiver<Job>) {
        let (queue, jobs) = mpsc::channel();
        let state = AppState {
            queue,
            pending: Arc::new(AtomicUsize::new(pending)),
            model_name: "test".to_string(),
            ocr: None,
            ocr_slot: Arc::new(tokio::sync::Semaphore::new(1)),
            voice: None,
            accounts: Arc::new(Accounts::in_memory().unwrap()),
            secure_cookies: false,
            limits: Arc::new(Limits::new(20, 100)),
            trust_proxy: false,
            last_number: Arc::new(AtomicU64::new(0)),
            now_serving: Arc::new(AtomicU64::new(0)),
        };
        (state, jobs)
    }

    fn state_with_pending(n: usize) -> AppState {
        test_state(n).0
    }

    const PEER: ([u8; 4], u16) = ([127, 0, 0, 1], 5000);

    async fn ask(state: &AppState, request: ChatRequest) -> Response {
        chat_completions(State(state.clone()), ConnectInfo(SocketAddr::from(PEER)), HeaderMap::new(), Json(request)).await
    }

    async fn body_text(response: Response) -> String {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    fn done_stats() -> Piece {
        Piece::Done(Stats {
            prompt_tokens: 3,
            generated_tokens: 1,
            tokens_per_sec: 20.0,
            finish_reason: crate::engine::FinishReason::Stop,
        })
    }

    fn some_pass() -> Pass {
        Arc::new(Limits::new(1, 1)).admit(&Who::user(1)).unwrap()
    }

    #[tokio::test]
    async fn photo_without_ocr_models_answers_503() {
        let response = read_photo(State(state_with_pending(0)), Bytes::from_static(b"x")).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn full_queue_answers_429() {
        let response = ask(&state_with_pending(MAX_PENDING), request("user", "hi")).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn unknown_role_answers_400() {
        let response = ask(&state_with_pending(0), request("admin", "hi")).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn too_long_message_answers_400() {
        let long = "x".repeat(MAX_MESSAGE_CHARS + 1);
        let response = ask(&state_with_pending(0), request("user", &long)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stream_sends_text_then_stats_then_done() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Piece::Text("Hello".to_string())).unwrap();
        tx.send(done_stats()).unwrap();
        drop(tx); // the worker is finished

        let ticket = Ticket { number: 1, now_serving: Arc::new(AtomicU64::new(1)) };
        let body = body_text(stream_reply(rx, "test-model".to_string(), some_pass(), ticket)).await;

        let events: Vec<&str> = body.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert_eq!(events.len(), 3);
        assert!(events[0].contains(r#""content":"Hello""#));
        assert!(events[1].contains(r#""finish_reason":"stop""#));
        assert_eq!(events[2], "data: [DONE]");
    }

    #[tokio::test]
    async fn second_question_while_waiting_answers_429() {
        let (state, jobs) = test_state(0);
        let mut streamed = request("user", "hi");
        streamed.stream = true;
        let first = ask(&state, streamed).await;
        assert_eq!(first.status(), StatusCode::OK);

        // The first is still in the queue, so the same guest can't ask again yet
        let second = ask(&state, request("user", "and another")).await;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(body_text(second).await.contains("Please wait for your current answer to finish first."));

        // Once the first reply is over, they may ask again
        let job = jobs.recv().unwrap();
        job.reply_to.send(done_stats()).unwrap();
        drop(job);
        body_text(first).await; // runs the stream to its end, which gives the pass back
        let mut third = request("user", "now?");
        third.stream = true;
        assert_eq!(ask(&state, third).await.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn queue_positions_come_before_the_text() {
        let now_serving = Arc::new(AtomicU64::new(1));
        let ticket = Ticket { number: 3, now_serving: now_serving.clone() };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let response = stream_reply(rx, "test-model".to_string(), some_pass(), ticket);

        // Pretend to be the worker: two jobs ahead, then one, then ours
        tokio::spawn(async move {
            tokio::time::sleep(QUEUE_UPDATE_EVERY + Duration::from_millis(200)).await;
            now_serving.store(2, Ordering::SeqCst);
            tokio::time::sleep(QUEUE_UPDATE_EVERY).await;
            now_serving.store(3, Ordering::SeqCst);
            tokio::time::sleep(QUEUE_UPDATE_EVERY).await;
            tx.send(Piece::Text("Hi".to_string())).unwrap();
            tx.send(done_stats()).unwrap();
        });

        let body = body_text(response).await;
        let events: Vec<&str> = body.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert!(events[0].contains(r#""queue_position":2"#), "{body}");
        assert!(events[1].contains(r#""queue_position":1"#), "{body}");
        assert!(events[2].contains(r#""queue_position":0"#), "{body}");
        assert!(events[3].contains(r#""content":"Hi""#), "{body}");
        assert_eq!(events.len(), 6, "{body}"); // + stats + [DONE]; no positions after the text
    }

    #[test]
    fn proxy_headers_only_count_when_trusted() {
        let (mut state, _jobs) = test_state(0);
        let peer = SocketAddr::from(PEER);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.9, 10.0.0.1".parse().unwrap());

        assert_eq!(visitor(&state, &headers, peer).key, "ip:127.0.0.1");

        state.trust_proxy = true;
        assert_eq!(visitor(&state, &headers, peer).key, "ip:203.0.113.9");
        headers.insert("cf-connecting-ip", "2001:db8::1".parse().unwrap());
        assert_eq!(visitor(&state, &headers, peer).key, "ip:2001:db8::1");

        // Not an IP address: fall back to the connection's address
        let mut junk = HeaderMap::new();
        junk.insert("x-forwarded-for", "user:1".parse().unwrap());
        let who = visitor(&state, &junk, peer);
        assert_eq!(who.key, "ip:127.0.0.1");
        assert!(who.guest);
    }

    #[tokio::test]
    async fn logged_in_people_are_counted_by_account() {
        let state = state_with_pending(0);
        let request = SignupRequest { name: "A".into(), email: "a@example.com".into(), password: "kinsin-rocks".into() };
        let cookie = cookie_from(&signup(State(state.clone()), Json(request)).await);
        let who = visitor(&state, &cookie, SocketAddr::from(PEER));
        assert!(who.key.starts_with("user:") && !who.guest, "{who:?}");
    }

    fn cookie_from(response: &Response) -> HeaderMap {
        let set = response.headers()[SET_COOKIE].to_str().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(COOKIE, set.split(';').next().unwrap().parse().unwrap());
        headers
    }

    #[tokio::test]
    async fn signup_cookie_chats_and_logout() {
        let state = state_with_pending(0);
        let request = SignupRequest { name: "Gauri".into(), email: "g@example.com".into(), password: "kinsin-rocks".into() };
        let response = signup(State(state.clone()), Json(request)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let set = response.headers()[SET_COOKIE].to_str().unwrap().to_string();
        assert!(set.contains("HttpOnly") && set.contains("SameSite=Lax"), "{set}");
        let cookie = cookie_from(&response);

        assert_eq!(me(State(state.clone()), cookie.clone()).await.status(), StatusCode::OK);
        assert_eq!(me(State(state.clone()), HeaderMap::new()).await.status(), StatusCode::UNAUTHORIZED);

        let chats = json!([{ "id": "1", "title": "hi", "messages": [] }]);
        assert_eq!(save_chats(State(state.clone()), cookie.clone(), Json(chats.clone())).await.status(), StatusCode::OK);
        let body = axum::body::to_bytes(load_chats(State(state.clone()), cookie.clone()).await.into_body(), usize::MAX).await.unwrap();
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap(), chats);
        assert_eq!(save_chats(State(state.clone()), cookie.clone(), Json(json!({"x": 1}))).await.status(), StatusCode::BAD_REQUEST);

        logout(State(state.clone()), cookie.clone()).await;
        assert_eq!(me(State(state.clone()), cookie.clone()).await.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(load_chats(State(state), cookie).await.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn wrong_login_answers_401() {
        let state = state_with_pending(0);
        let request = LoginRequest { email: "nobody@example.com".into(), password: "whatever1".into() };
        assert_eq!(login(State(state), Json(request)).await.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn bad_cookies_are_ignored() {
        let mut headers = HeaderMap::new();
        headers.insert(COOKIE, "theme=dark; kinsin_session=not-hex".parse().unwrap());
        assert_eq!(session_token(&headers), None);
        let good = "a".repeat(64);
        headers.insert(COOKIE, format!("theme=dark; kinsin_session={good}").parse().unwrap());
        assert_eq!(session_token(&headers), Some(good));
    }
}
