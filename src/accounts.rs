//! Accounts: sign up, log in, log out, and each user's saved chats.
//!
//! Everything is kept in one small database file (SQLite), data/kinsin.db by default:
//!   users     who has an account (the password is stored only as an Argon2 hash)
//!   sessions  who is logged in (the browser holds a random token in a cookie;
//!             we store only a SHA-256 of it, so a stolen database can't log anyone in)
//!   chats     each user's chat list, so it follows them to every device
//!   memories  short facts about each user, given to the model in later chats
//!   feedback  👍/👎 and better answers, kept only for users who turned on
//!             "Help improve kinSin" (users.improve), for --export-training
//!
//! Hashing a password with Argon2 takes ~50 ms of CPU on purpose (slow for
//! attackers), so the server calls these functions from a blocking thread.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use argon2::{Argon2, PasswordHasher, PasswordVerifier};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::memory::{MAX_FACT_CHARS, MAX_FACTS};

/// How long a login lasts.
pub const SESSION_DAYS: u64 = 30;
/// Wrong passwords allowed for one email before it is locked for a while.
const MAX_FAILURES: u32 = 8;
const LOCK_TIME: Duration = Duration::from_secs(15 * 60);
/// Longest better answer a user can write.
const MAX_CORRECTION_CHARS: usize = 8000;
/// Limits on the conversation stored with a rating.
const MAX_FEEDBACK_MESSAGES: usize = 200;
const MAX_FEEDBACK_MESSAGE_CHARS: usize = 16_000;
const MAX_CHAT_ID_CHARS: usize = 100;

/// Someone with an account.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct User {
    #[serde(skip)]
    pub id: i64,
    pub name: String,
    pub email: String,
    /// "Help improve kinSin": their ratings may be used to train the model.
    pub improve: bool,
}

/// The columns that make a User, in the order `user_from_row` reads them.
const USER_COLUMNS: &str = "users.id, users.name, users.email, users.improve";

fn user_from_row(row: &rusqlite::Row) -> rusqlite::Result<User> {
    Ok(User { id: row.get(0)?, name: row.get(1)?, email: row.get(2)?, improve: row.get(3)? })
}

/// One remembered fact.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Memory {
    pub id: i64,
    pub text: String,
    pub created_at: i64,
}

/// One message of a rated conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// A rating of one answer (the message at `index` in chat `chat_id`).
#[derive(Debug, Clone)]
pub struct Feedback {
    pub chat_id: String,
    pub index: i64,
    /// 1 = 👍, -1 = 👎, 0 = no rating.
    pub rating: i64,
    /// What the answer should have been (optional).
    pub corrected: Option<String>,
    /// The answer used live Wikipedia data (such answers age, so they are not trained on).
    pub live: bool,
    /// The conversation up to and including the rated answer.
    pub conversation: Vec<Turn>,
}

/// Why signing up or logging in didn't work. The message is safe to show to users.
#[derive(Debug, PartialEq)]
pub enum AuthError {
    BadInput(String),
    EmailTaken,
    WrongLogin,
    TooManyTries,
    Internal(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::BadInput(message) => write!(f, "{message}"),
            AuthError::EmailTaken => write!(f, "That email already has an account. Sign in instead."),
            // Same message for "no such email" and "wrong password", so emails can't be guessed
            AuthError::WrongLogin => write!(f, "Wrong email or password."),
            AuthError::TooManyTries => write!(f, "Too many wrong tries. Please wait 15 minutes."),
            AuthError::Internal(_) => write!(f, "Something went wrong on the server. Please try again."),
        }
    }
}

impl From<rusqlite::Error> for AuthError {
    fn from(e: rusqlite::Error) -> Self {
        AuthError::Internal(e.to_string())
    }
}

pub struct Accounts {
    db: Mutex<Connection>,
    /// Wrong-password counts per email: (count, when counting started).
    failures: Mutex<HashMap<String, (u32, Instant)>>,
    /// A real hash of a random password, checked when an email doesn't exist,
    /// so "no such user" takes as long as "wrong password".
    dummy_hash: String,
}

impl Accounts {
    /// Open (or create) the database file.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(folder) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(folder)?;
        }
        Self::with_connection(Connection::open(path)?)
    }

    /// A database that lives only in memory (for tests).
    #[cfg(test)]
    pub fn in_memory() -> anyhow::Result<Self> {
        Self::with_connection(Connection::open_in_memory()?)
    }

    fn with_connection(db: Connection) -> anyhow::Result<Self> {
        db.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS users (
                 id            INTEGER PRIMARY KEY,
                 email         TEXT NOT NULL UNIQUE,
                 name          TEXT NOT NULL,
                 password_hash TEXT NOT NULL,
                 created_at    INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS sessions (
                 token_hash TEXT PRIMARY KEY,
                 user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 expires_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS chats (
                 user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                 data       TEXT NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS memories (
                 id         INTEGER PRIMARY KEY,
                 user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 text       TEXT NOT NULL,
                 created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS memories_by_user ON memories(user_id);
             CREATE TABLE IF NOT EXISTS feedback (
                 user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 chat_id       TEXT NOT NULL,
                 message_index INTEGER NOT NULL,
                 rating        INTEGER NOT NULL,
                 corrected     TEXT,
                 live          INTEGER NOT NULL,
                 conversation  TEXT NOT NULL,
                 updated_at    INTEGER NOT NULL,
                 PRIMARY KEY (user_id, chat_id, message_index)
             );",
        )?;
        // Databases made before "Help improve kinSin" existed: add the column (off for everyone)
        let has_improve = db
            .prepare("SELECT 1 FROM pragma_table_info('users') WHERE name = 'improve'")?
            .exists([])?;
        if !has_improve {
            db.execute_batch("ALTER TABLE users ADD COLUMN improve INTEGER NOT NULL DEFAULT 0;")?;
        }
        let dummy_hash = hash_password(&random_token())?;
        Ok(Accounts { db: Mutex::new(db), failures: Mutex::new(HashMap::new()), dummy_hash })
    }

    /// Make a new account and log it in. Returns the user and their session token.
    pub fn signup(&self, name: &str, email: &str, password: &str) -> Result<(User, String), AuthError> {
        let name = name.trim();
        let email = clean_email(email);
        if name.is_empty() || name.chars().count() > 40 {
            return Err(AuthError::BadInput("Please type your name (up to 40 letters).".into()));
        }
        if !looks_like_email(&email) {
            return Err(AuthError::BadInput("That email doesn't look right.".into()));
        }
        check_password(password)?;

        let hash = hash_password(password).map_err(|e| AuthError::Internal(e.to_string()))?;
        let db = self.db.lock().unwrap();
        let inserted = db.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![email, name, hash, now()],
        );
        match inserted {
            Err(rusqlite::Error::SqliteFailure(e, _)) if e.code == rusqlite::ErrorCode::ConstraintViolation => {
                return Err(AuthError::EmailTaken);
            }
            other => {
                other?;
            }
        }
        let user = User { id: db.last_insert_rowid(), name: name.to_string(), email, improve: false };
        let token = new_session(&db, user.id)?;
        Ok((user, token))
    }

    /// Check an email and password. Returns the user and a new session token.
    pub fn login(&self, email: &str, password: &str) -> Result<(User, String), AuthError> {
        let email = clean_email(email);
        if self.is_locked(&email) {
            return Err(AuthError::TooManyTries);
        }
        let found: Option<(User, String)> = self
            .db
            .lock()
            .unwrap()
            .query_row(
                &format!("SELECT {USER_COLUMNS}, users.password_hash FROM users WHERE email = ?1"),
                params![email],
                |row| Ok((user_from_row(row)?, row.get(4)?)),
            )
            .optional()?;

        // Always check a hash (a dummy one if the email is unknown), so both cases take the same time
        let hash = found.as_ref().map(|(_, h)| h.as_str()).unwrap_or(&self.dummy_hash);
        let correct = Argon2::default().verify_password(password.as_bytes(), hash).is_ok();
        match found {
            Some((user, _)) if correct => {
                self.failures.lock().unwrap().remove(&email);
                let db = self.db.lock().unwrap();
                db.execute("DELETE FROM sessions WHERE expires_at < ?1", params![now()])?; // tidy up old logins
                let token = new_session(&db, user.id)?;
                Ok((user, token))
            }
            _ => {
                self.count_failure(&email);
                Err(AuthError::WrongLogin)
            }
        }
    }

    /// Who owns this session token (None if it's unknown or expired).
    pub fn user_for_token(&self, token: &str) -> Option<User> {
        self.db
            .lock()
            .unwrap()
            .query_row(
                &format!(
                    "SELECT {USER_COLUMNS} FROM sessions JOIN users ON users.id = sessions.user_id
                     WHERE sessions.token_hash = ?1 AND sessions.expires_at > ?2"
                ),
                params![sha256_hex(token), now()],
                user_from_row,
            )
            .ok()
    }

    /// End a session.
    pub fn logout(&self, token: &str) {
        let _ = self.db.lock().unwrap().execute("DELETE FROM sessions WHERE token_hash = ?1", params![sha256_hex(token)]);
    }

    /// A user's saved chats, as a JSON array ("[]" if none yet).
    pub fn load_chats(&self, user_id: i64) -> Result<String, AuthError> {
        let data: Option<String> = self
            .db
            .lock()
            .unwrap()
            .query_row("SELECT data FROM chats WHERE user_id = ?1", params![user_id], |row| row.get(0))
            .optional()?;
        Ok(data.unwrap_or_else(|| "[]".into()))
    }

    /// Replace a user's saved chats.
    pub fn save_chats(&self, user_id: i64, json: &str) -> Result<(), AuthError> {
        self.db.lock().unwrap().execute(
            "INSERT INTO chats (user_id, data, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            params![user_id, json, now()],
        )?;
        Ok(())
    }

    // ---------- Memory ----------

    /// Everything remembered about a user, newest first.
    pub fn memories(&self, user_id: i64) -> Result<Vec<Memory>, AuthError> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare("SELECT id, text, created_at FROM memories WHERE user_id = ?1 ORDER BY id DESC")?;
        let rows = query.query_map(params![user_id], |row| {
            Ok(Memory { id: row.get(0)?, text: row.get(1)?, created_at: row.get(2)? })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Remember a fact. Ok(false) if it is already remembered (ignoring case).
    pub fn remember(&self, user_id: i64, text: &str) -> Result<bool, AuthError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(AuthError::BadInput("Type something to remember.".into()));
        }
        if text.chars().count() > MAX_FACT_CHARS {
            return Err(AuthError::BadInput(format!("A memory can be at most {MAX_FACT_CHARS} characters.")));
        }
        let db = self.db.lock().unwrap();
        let known: Vec<String> = db
            .prepare("SELECT text FROM memories WHERE user_id = ?1")?
            .query_map(params![user_id], |row| row.get(0))?
            .collect::<Result<_, _>>()?;
        let lower = text.to_lowercase();
        if known.iter().any(|k| k.to_lowercase() == lower) {
            return Ok(false);
        }
        if known.len() >= MAX_FACTS {
            return Err(AuthError::BadInput(format!(
                "Memory is full ({MAX_FACTS} things). Delete some to make room."
            )));
        }
        db.execute(
            "INSERT INTO memories (user_id, text, created_at) VALUES (?1, ?2, ?3)",
            params![user_id, text, now()],
        )?;
        Ok(true)
    }

    /// Forget one fact. Only the user's own; Ok(false) if there was no such fact.
    pub fn forget(&self, user_id: i64, memory_id: i64) -> Result<bool, AuthError> {
        let deleted = self
            .db
            .lock()
            .unwrap()
            .execute("DELETE FROM memories WHERE id = ?1 AND user_id = ?2", params![memory_id, user_id])?;
        Ok(deleted > 0)
    }

    // ---------- Help improve kinSin ----------

    /// Turn "Help improve kinSin" on or off. Turning it off deletes every rating the user gave.
    pub fn set_improve(&self, user_id: i64, on: bool) -> Result<(), AuthError> {
        let mut db = self.db.lock().unwrap();
        let tx = db.transaction()?;
        tx.execute("UPDATE users SET improve = ?1 WHERE id = ?2", params![on, user_id])?;
        if !on {
            tx.execute("DELETE FROM feedback WHERE user_id = ?1", params![user_id])?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Save (or replace) a rating. A rating of 0 with no better answer removes it.
    pub fn save_feedback(&self, user: &User, feedback: &Feedback) -> Result<(), AuthError> {
        if !user.improve {
            return Err(AuthError::BadInput("Turn on \"Help improve kinSin\" (🧠 Memory) to rate answers.".into()));
        }
        let chat_id = feedback.chat_id.trim();
        if chat_id.is_empty() || chat_id.chars().count() > MAX_CHAT_ID_CHARS {
            return Err(AuthError::BadInput("Unknown chat.".into()));
        }
        if !(-1..=1).contains(&feedback.rating) {
            return Err(AuthError::BadInput("A rating is 1, 0 or -1.".into()));
        }
        if feedback.index < 0 {
            return Err(AuthError::BadInput("Unknown message.".into()));
        }
        let corrected = feedback.corrected.as_deref().map(str::trim).filter(|c| !c.is_empty());
        if corrected.is_some_and(|c| c.chars().count() > MAX_CORRECTION_CHARS) {
            return Err(AuthError::BadInput(format!(
                "A better answer can be at most {MAX_CORRECTION_CHARS} characters."
            )));
        }
        let turns = &feedback.conversation;
        if turns.is_empty() || turns.len() > MAX_FEEDBACK_MESSAGES {
            return Err(AuthError::BadInput("The conversation is empty or too long.".into()));
        }
        if turns.iter().any(|t| {
            !matches!(t.role.as_str(), "system" | "user" | "assistant")
                || t.content.chars().count() > MAX_FEEDBACK_MESSAGE_CHARS
        }) {
            return Err(AuthError::BadInput("A message in the conversation isn't valid.".into()));
        }
        if turns.last().map(|t| t.role.as_str()) != Some("assistant") {
            return Err(AuthError::BadInput("Only kinSin's answers can be rated.".into()));
        }

        let db = self.db.lock().unwrap();
        if feedback.rating == 0 && corrected.is_none() {
            db.execute(
                "DELETE FROM feedback WHERE user_id = ?1 AND chat_id = ?2 AND message_index = ?3",
                params![user.id, chat_id, feedback.index],
            )?;
            return Ok(());
        }
        let conversation = serde_json::to_string(turns).map_err(|e| AuthError::Internal(e.to_string()))?;
        db.execute(
            "INSERT INTO feedback (user_id, chat_id, message_index, rating, corrected, live, conversation, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(user_id, chat_id, message_index) DO UPDATE SET
                 rating = excluded.rating, corrected = excluded.corrected, live = excluded.live,
                 conversation = excluded.conversation, updated_at = excluded.updated_at",
            params![user.id, chat_id, feedback.index, feedback.rating, corrected, feedback.live, conversation, now()],
        )?;
        Ok(())
    }

    /// Write the training set: one {"messages": [...]} per line (JSONL), from users who
    /// turned on "Help improve kinSin". Only good answers (👍) and better answers that
    /// users wrote; answers built on live Wikipedia data are left out. Returns how many.
    pub fn export_training(&self, system_prompt: &str, out: &mut impl Write) -> anyhow::Result<usize> {
        let rows: Vec<(String, Option<String>)> = {
            let db = self.db.lock().unwrap();
            let mut query = db.prepare(
                "SELECT feedback.conversation, feedback.corrected FROM feedback
                 JOIN users ON users.id = feedback.user_id
                 WHERE users.improve = 1 AND feedback.live = 0
                   AND (feedback.rating = 1 OR feedback.corrected IS NOT NULL)
                 ORDER BY feedback.updated_at, feedback.rowid",
            )?;
            query.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<Result<_, _>>()?
        };
        let mut count = 0;
        for (conversation, corrected) in rows {
            let Ok(mut turns) = serde_json::from_str::<Vec<Turn>>(&conversation) else { continue };
            // The user's better answer replaces kinSin's
            if let (Some(better), Some(last)) = (corrected, turns.iter_mut().rev().find(|t| t.role == "assistant")) {
                last.content = better;
            }
            if turns.first().map(|t| t.role.as_str()) != Some("system") {
                turns.insert(0, Turn { role: "system".into(), content: system_prompt.into() });
            }
            serde_json::to_writer(&mut *out, &serde_json::json!({ "messages": turns }))?;
            out.write_all(b"\n")?;
            count += 1;
        }
        Ok(count)
    }

    fn is_locked(&self, email: &str) -> bool {
        let mut failures = self.failures.lock().unwrap();
        match failures.get(email) {
            Some((_, since)) if since.elapsed() > LOCK_TIME => {
                failures.remove(email);
                false
            }
            Some((count, _)) => *count >= MAX_FAILURES,
            None => false,
        }
    }

    fn count_failure(&self, email: &str) {
        let mut failures = self.failures.lock().unwrap();
        // Don't let a flood of random emails fill the memory
        if failures.len() > 10_000 {
            failures.retain(|_, (_, since)| since.elapsed() <= LOCK_TIME);
        }
        failures.entry(email.to_string()).or_insert((0, Instant::now())).0 += 1;
    }
}

// ---------- Small helpers ----------

fn new_session(db: &Connection, user_id: i64) -> Result<String, AuthError> {
    let token = random_token();
    db.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?1, ?2, ?3)",
        params![sha256_hex(&token), user_id, now() + SESSION_DAYS as i64 * 24 * 60 * 60],
    )?;
    Ok(token)
}

fn hash_password(password: &str) -> anyhow::Result<String> {
    Argon2::default()
        .hash_password(password.as_bytes())
        .map(|hash| hash.to_string())
        .map_err(|e| anyhow::anyhow!("could not hash the password: {e}"))
}

fn check_password(password: &str) -> Result<(), AuthError> {
    let length = password.chars().count();
    if length < 8 {
        return Err(AuthError::BadInput("Use at least 8 characters for the password.".into()));
    }
    if length > 200 {
        return Err(AuthError::BadInput("That password is too long (200 characters at most).".into()));
    }
    Ok(())
}

fn clean_email(email: &str) -> String {
    email.trim().to_lowercase()
}

fn looks_like_email(email: &str) -> bool {
    let Some((name, domain)) = email.split_once('@') else { return false };
    email.len() <= 120
        && !name.is_empty()
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
        && !email.chars().any(|c| c.is_whitespace() || c == '<' || c == '>')
}

/// 32 random bytes as 64 hex letters.
fn random_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("the operating system has no random numbers");
    hex(&bytes)
}

fn sha256_hex(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signup_then_login_and_logout() {
        let accounts = Accounts::in_memory().unwrap();
        let (user, token) = accounts.signup(" Gauri ", "Gauri@Example.com", "kinsin-rocks").unwrap();
        assert_eq!((user.name.as_str(), user.email.as_str()), ("Gauri", "gauri@example.com"));
        assert_eq!(accounts.user_for_token(&token), Some(user.clone()));

        let (again, token2) = accounts.login("GAURI@example.com ", "kinsin-rocks").unwrap();
        assert_eq!(again, user);
        assert_ne!(token, token2, "every login gets its own token");

        accounts.logout(&token);
        assert_eq!(accounts.user_for_token(&token), None);
        assert!(accounts.user_for_token(&token2).is_some(), "other devices stay logged in");
    }

    #[test]
    fn passwords_are_never_stored() {
        let accounts = Accounts::in_memory().unwrap();
        accounts.signup("Gauri", "g@example.com", "kinsin-rocks").unwrap();
        let stored: String = accounts
            .db
            .lock()
            .unwrap()
            .query_row("SELECT password_hash FROM users", [], |row| row.get(0))
            .unwrap();
        assert!(stored.starts_with("$argon2id$"), "{stored}");
        assert!(!stored.contains("kinsin-rocks"));
    }

    #[test]
    fn bad_signups_are_refused() {
        let accounts = Accounts::in_memory().unwrap();
        assert!(matches!(accounts.signup("", "a@b.com", "longenough"), Err(AuthError::BadInput(_))));
        assert!(matches!(accounts.signup("A", "not-an-email", "longenough"), Err(AuthError::BadInput(_))));
        assert!(matches!(accounts.signup("A", "a@b.com", "short"), Err(AuthError::BadInput(_))));
        accounts.signup("A", "a@b.com", "longenough").unwrap();
        assert_eq!(accounts.signup("B", "A@B.com", "longenough").unwrap_err(), AuthError::EmailTaken);
    }

    #[test]
    fn wrong_password_and_unknown_email_look_the_same() {
        let accounts = Accounts::in_memory().unwrap();
        accounts.signup("A", "a@b.com", "longenough").unwrap();
        let wrong = accounts.login("a@b.com", "wrong-pass").unwrap_err();
        let unknown = accounts.login("nobody@b.com", "longenough").unwrap_err();
        assert_eq!(wrong, AuthError::WrongLogin);
        assert_eq!(wrong.to_string(), unknown.to_string());
    }

    #[test]
    fn too_many_wrong_passwords_lock_the_email() {
        let accounts = Accounts::in_memory().unwrap();
        accounts.signup("A", "a@b.com", "longenough").unwrap();
        for _ in 0..MAX_FAILURES {
            assert_eq!(accounts.login("a@b.com", "nope-nope").unwrap_err(), AuthError::WrongLogin);
        }
        // Even the right password waits now
        assert_eq!(accounts.login("a@b.com", "longenough").unwrap_err(), AuthError::TooManyTries);
    }

    #[test]
    fn expired_sessions_dont_work() {
        let accounts = Accounts::in_memory().unwrap();
        let (_, token) = accounts.signup("A", "a@b.com", "longenough").unwrap();
        accounts.db.lock().unwrap().execute("UPDATE sessions SET expires_at = 0", []).unwrap();
        assert_eq!(accounts.user_for_token(&token), None);
    }

    #[test]
    fn chats_are_saved_per_user() {
        let accounts = Accounts::in_memory().unwrap();
        let (a, _) = accounts.signup("A", "a@b.com", "longenough").unwrap();
        let (b, _) = accounts.signup("B", "b@b.com", "longenough").unwrap();
        assert_eq!(accounts.load_chats(a.id).unwrap(), "[]");
        accounts.save_chats(a.id, r#"[{"id":"1"}]"#).unwrap();
        accounts.save_chats(a.id, r#"[{"id":"2"}]"#).unwrap();
        assert_eq!(accounts.load_chats(a.id).unwrap(), r#"[{"id":"2"}]"#);
        assert_eq!(accounts.load_chats(b.id).unwrap(), "[]");
    }

    fn turn(role: &str, content: &str) -> Turn {
        Turn { role: role.into(), content: content.into() }
    }

    fn rating(chat: &str, index: i64, rating: i64, corrected: Option<&str>, live: bool) -> Feedback {
        Feedback {
            chat_id: chat.into(),
            index,
            rating,
            corrected: corrected.map(String::from),
            live,
            conversation: vec![turn("user", "What is 2+2?"), turn("assistant", "5")],
        }
    }

    /// A user who turned on "Help improve kinSin".
    fn helper(accounts: &Accounts, email: &str) -> User {
        let (mut user, _) = accounts.signup("H", email, "longenough").unwrap();
        accounts.set_improve(user.id, true).unwrap();
        user.improve = true;
        user
    }

    fn export(accounts: &Accounts) -> (usize, String) {
        let mut out = Vec::new();
        let count = accounts.export_training("SYS", &mut out).unwrap();
        (count, String::from_utf8(out).unwrap())
    }

    #[test]
    fn memories_belong_to_their_user() {
        let accounts = Accounts::in_memory().unwrap();
        let (a, _) = accounts.signup("A", "a@b.com", "longenough").unwrap();
        let (b, _) = accounts.signup("B", "b@b.com", "longenough").unwrap();
        assert!(accounts.remember(a.id, "The user likes tea.").unwrap());
        assert!(accounts.remember(a.id, "The user lives in Pune.").unwrap());
        assert!(!accounts.remember(a.id, "the USER likes TEA.").unwrap(), "duplicate");
        assert!(matches!(accounts.remember(a.id, "  "), Err(AuthError::BadInput(_))));

        let mine = accounts.memories(a.id).unwrap();
        let texts: Vec<&str> = mine.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, ["The user lives in Pune.", "The user likes tea."], "newest first");
        assert!(accounts.memories(b.id).unwrap().is_empty());

        assert!(!accounts.forget(b.id, mine[0].id).unwrap(), "B can't delete A's memory");
        assert!(accounts.forget(a.id, mine[0].id).unwrap());
        assert_eq!(accounts.memories(a.id).unwrap().len(), 1);
    }

    #[test]
    fn memory_has_a_limit() {
        let accounts = Accounts::in_memory().unwrap();
        let (a, _) = accounts.signup("A", "a@b.com", "longenough").unwrap();
        for i in 0..MAX_FACTS {
            assert!(accounts.remember(a.id, &format!("Fact {i}.")).unwrap());
        }
        assert!(matches!(accounts.remember(a.id, "One more."), Err(AuthError::BadInput(_))));
        assert!(matches!(accounts.remember(a.id, &"x".repeat(MAX_FACT_CHARS + 1)), Err(AuthError::BadInput(_))));
    }

    #[test]
    fn improve_is_off_until_turned_on() {
        let accounts = Accounts::in_memory().unwrap();
        let (user, token) = accounts.signup("A", "a@b.com", "longenough").unwrap();
        assert!(!user.improve);
        assert!(matches!(accounts.save_feedback(&user, &rating("c1", 1, 1, None, false)), Err(AuthError::BadInput(_))));

        accounts.set_improve(user.id, true).unwrap();
        let user = accounts.user_for_token(&token).unwrap();
        assert!(user.improve);
        accounts.save_feedback(&user, &rating("c1", 1, 1, None, false)).unwrap();
        assert_eq!(export(&accounts).0, 1);

        // Turning it off deletes their ratings
        accounts.set_improve(user.id, false).unwrap();
        assert_eq!(export(&accounts).0, 0);
        let count: i64 = accounts.db.lock().unwrap().query_row("SELECT COUNT(*) FROM feedback", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn feedback_is_checked() {
        let accounts = Accounts::in_memory().unwrap();
        let user = helper(&accounts, "a@b.com");
        let bad = |f: Feedback| matches!(accounts.save_feedback(&user, &f), Err(AuthError::BadInput(_)));
        assert!(bad(rating("c", 1, 2, None, false)), "rating out of range");
        assert!(bad(rating("", 1, 1, None, false)), "no chat");
        assert!(bad(rating("c", 1, -1, Some(&"x".repeat(MAX_CORRECTION_CHARS + 1)), false)), "long correction");
        let mut not_an_answer = rating("c", 0, 1, None, false);
        not_an_answer.conversation.pop();
        assert!(bad(not_an_answer), "last message must be kinSin's");
        let mut odd_role = rating("c", 1, 1, None, false);
        odd_role.conversation[0].role = "tool".into();
        assert!(bad(odd_role));
    }

    #[test]
    fn export_keeps_good_and_corrected_answers_only() {
        let accounts = Accounts::in_memory().unwrap();
        let user = helper(&accounts, "a@b.com");
        accounts.save_feedback(&user, &rating("good", 1, 1, None, false)).unwrap();
        accounts.save_feedback(&user, &rating("fixed", 1, -1, Some(" 4 "), false)).unwrap();
        accounts.save_feedback(&user, &rating("bad", 1, -1, None, false)).unwrap(); // 👎 alone: not trained on
        accounts.save_feedback(&user, &rating("live", 1, 1, None, true)).unwrap(); // live data: left out
        // Someone who turned "improve" off later is left out too
        let other = helper(&accounts, "o@b.com");
        accounts.save_feedback(&other, &rating("x", 1, 1, None, false)).unwrap();
        accounts.db.lock().unwrap().execute("UPDATE users SET improve = 0 WHERE id = ?1", params![other.id]).unwrap();

        let (count, jsonl) = export(&accounts);
        assert_eq!(count, 2, "{jsonl}");
        let lines: Vec<serde_json::Value> = jsonl.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(
            lines[0],
            serde_json::json!({ "messages": [
                { "role": "system", "content": "SYS" },
                { "role": "user", "content": "What is 2+2?" },
                { "role": "assistant", "content": "5" }
            ] })
        );
        assert_eq!(lines[1]["messages"][2]["content"], "4", "the correction replaces the answer");
    }

    #[test]
    fn rating_zero_removes_and_ratings_can_change() {
        let accounts = Accounts::in_memory().unwrap();
        let user = helper(&accounts, "a@b.com");
        accounts.save_feedback(&user, &rating("c", 1, 1, None, false)).unwrap();
        accounts.save_feedback(&user, &rating("c", 1, -1, Some("4"), false)).unwrap(); // same answer: replaced
        let (count, jsonl) = export(&accounts);
        assert_eq!(count, 1);
        assert!(jsonl.contains(r#""content":"4""#));
        accounts.save_feedback(&user, &rating("c", 1, 0, None, false)).unwrap();
        assert_eq!(export(&accounts).0, 0);
    }

    #[test]
    fn old_databases_get_the_improve_column() {
        let path = std::env::temp_dir().join(format!("kinsin-migrate-{}.db", random_token()));
        {
            let old = Connection::open(&path).unwrap();
            old.execute_batch(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                                     password_hash TEXT NOT NULL, created_at INTEGER NOT NULL);",
            )
            .unwrap();
        }
        let accounts = Accounts::open(&path).unwrap();
        let (user, _) = accounts.signup("A", "a@b.com", "longenough").unwrap();
        assert!(!user.improve);
        drop(accounts);
        Accounts::open(&path).unwrap(); // opening again doesn't try to add it twice
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}
