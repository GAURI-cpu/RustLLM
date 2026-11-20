//! Accounts: sign up, log in, log out, and each user's saved chats.
//!
//! Everything is kept in one small database file (SQLite), data/kinsin.db by default:
//!   users     who has an account (the password is stored only as an Argon2 hash)
//!   sessions  who is logged in (the browser holds a random token in a cookie;
//!             we store only a SHA-256 of it, so a stolen database can't log anyone in)
//!   chats     each user's chat list, so it follows them to every device
//!
//! Hashing a password with Argon2 takes ~50 ms of CPU on purpose (slow for
//! attackers), so the server calls these functions from a blocking thread.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use argon2::{Argon2, PasswordHasher, PasswordVerifier};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

/// How long a login lasts.
pub const SESSION_DAYS: u64 = 30;
/// Wrong passwords allowed for one email before it is locked for a while.
const MAX_FAILURES: u32 = 8;
const LOCK_TIME: Duration = Duration::from_secs(15 * 60);

/// Someone with an account.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct User {
    #[serde(skip)]
    pub id: i64,
    pub name: String,
    pub email: String,
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
             );",
        )?;
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
        let user = User { id: db.last_insert_rowid(), name: name.to_string(), email };
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
                "SELECT id, name, email, password_hash FROM users WHERE email = ?1",
                params![email],
                |row| Ok((User { id: row.get(0)?, name: row.get(1)?, email: row.get(2)? }, row.get(3)?)),
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
                "SELECT users.id, users.name, users.email FROM sessions JOIN users ON users.id = sessions.user_id
                 WHERE sessions.token_hash = ?1 AND sessions.expires_at > ?2",
                params![sha256_hex(token), now()],
                |row| Ok(User { id: row.get(0)?, name: row.get(1)?, email: row.get(2)? }),
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
}
