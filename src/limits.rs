//! Fair use: everyone gets a turn.
//!
//! Two rules, checked before a question joins the queue:
//!   1. One question at a time per person: a second one while the first is
//!      still being answered is refused.
//!   2. A cap per hour (a sliding window), lower for guests than for people
//!      who are logged in.
//!
//! A person is a logged-in user ("user:<id>") or, for guests, an IP address ("ip:<addr>").

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The window the hourly limits count over.
const WINDOW: Duration = Duration::from_secs(60 * 60);
/// Above this many remembered people, forget the ones with nothing left in the window.
const PRUNE_ABOVE: usize = 10_000;

/// Who is asking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Who {
    /// "user:<id>" or "ip:<addr>".
    pub key: String,
    /// Not logged in: gets the guest limit.
    pub guest: bool,
}

impl Who {
    pub fn user(id: i64) -> Self {
        Who { key: format!("user:{id}"), guest: false }
    }

    pub fn guest(ip: std::net::IpAddr) -> Self {
        Who { key: format!("ip:{ip}"), guest: true }
    }
}

/// Why a question was turned away.
#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    /// This person's previous question is still being answered.
    AlreadyAsking,
    /// Too many questions in the last hour.
    TooMany { limit: usize, wait_minutes: u64, guest: bool },
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::AlreadyAsking => write!(f, "Please wait for your current answer to finish first."),
            Refused::TooMany { limit, wait_minutes, guest: true } => write!(
                f,
                "Guests can send {limit} messages an hour. Log in for more, or try again in {wait_minutes} min."
            ),
            Refused::TooMany { limit, wait_minutes, guest: false } => {
                write!(f, "You've sent {limit} messages this hour. Please try again in {wait_minutes} min.")
            }
        }
    }
}

pub struct Limits {
    pub guest_per_hour: usize,
    pub user_per_hour: usize,
    /// When each person asked, oldest first, within the last hour.
    asked: Mutex<HashMap<String, VecDeque<Instant>>>,
    /// People whose question is being answered (or waiting in the queue) right now.
    busy: Mutex<HashSet<String>>,
}

/// Permission to ask one question. While it is alive the person counts as busy;
/// dropping it (reply finished, client gone) lets them ask again.
#[must_use]
pub struct Pass {
    limits: Arc<Limits>,
    key: String,
}

impl Drop for Pass {
    fn drop(&mut self) {
        if let Ok(mut busy) = self.limits.busy.lock() {
            busy.remove(&self.key);
        }
    }
}

impl Limits {
    pub fn new(guest_per_hour: usize, user_per_hour: usize) -> Self {
        Limits {
            guest_per_hour,
            user_per_hour,
            asked: Mutex::new(HashMap::new()),
            busy: Mutex::new(HashSet::new()),
        }
    }

    /// May `who` ask a question now? If so, it is counted and they are busy until the Pass is dropped.
    pub fn admit(self: &Arc<Self>, who: &Who) -> Result<Pass, Refused> {
        self.admit_at(who, Instant::now())
    }

    fn admit_at(self: &Arc<Self>, who: &Who, now: Instant) -> Result<Pass, Refused> {
        // Locks are always taken in this order (busy, then asked), so they can't deadlock.
        let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        if busy.contains(&who.key) {
            return Err(Refused::AlreadyAsking);
        }

        let limit = if who.guest { self.guest_per_hour } else { self.user_per_hour };
        let mut asked = self.asked.lock().unwrap_or_else(|e| e.into_inner());
        if asked.len() > PRUNE_ABOVE {
            asked.retain(|_, times| {
                forget_old(times, now);
                !times.is_empty()
            });
        }
        let times = asked.entry(who.key.clone()).or_default();
        forget_old(times, now);
        if times.len() >= limit {
            // Wait until the oldest question leaves the window (rounded up, at least a minute)
            let wait = match times.front() {
                Some(&oldest) => (oldest + WINDOW).saturating_duration_since(now),
                None => WINDOW, // a limit of 0: nothing will ever leave
            };
            let wait_minutes = wait.as_secs().div_ceil(60).max(1);
            return Err(Refused::TooMany { limit, wait_minutes, guest: who.guest });
        }

        times.push_back(now);
        busy.insert(who.key.clone());
        Ok(Pass { limits: self.clone(), key: who.key.clone() })
    }
}

/// Drop the times that are an hour old or more.
fn forget_old(times: &mut VecDeque<Instant>, now: Instant) {
    while times.front().is_some_and(|&t| now.saturating_duration_since(t) >= WINDOW) {
        times.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guest() -> Who {
        Who::guest("10.0.0.1".parse().unwrap())
    }

    #[test]
    fn one_question_at_a_time() {
        let limits = Arc::new(Limits::new(5, 10));
        let pass = limits.admit(&guest()).unwrap();
        assert_eq!(limits.admit(&guest()).err(), Some(Refused::AlreadyAsking));
        // someone else is not affected
        let _other = limits.admit(&Who::user(7)).unwrap();
        drop(pass);
        assert!(limits.admit(&guest()).is_ok());
    }

    #[test]
    fn guests_hit_their_hourly_limit() {
        let limits = Arc::new(Limits::new(2, 10));
        let start = Instant::now();
        drop(limits.admit_at(&guest(), start).unwrap());
        drop(limits.admit_at(&guest(), start + Duration::from_secs(10 * 60)).unwrap());

        // 20 minutes in: the first question leaves the window in 40 minutes
        let refused = limits.admit_at(&guest(), start + Duration::from_secs(20 * 60)).err().unwrap();
        assert_eq!(refused, Refused::TooMany { limit: 2, wait_minutes: 40, guest: true });
        assert_eq!(
            refused.to_string(),
            "Guests can send 2 messages an hour. Log in for more, or try again in 40 min."
        );

        // An hour after the first one, there is room again
        assert!(limits.admit_at(&guest(), start + WINDOW).is_ok());
    }

    #[test]
    fn users_get_their_own_limit_and_message() {
        let limits = Arc::new(Limits::new(1, 3));
        let me = Who::user(42);
        let start = Instant::now();
        for i in 0..3 {
            drop(limits.admit_at(&me, start + Duration::from_secs(i)).unwrap());
        }
        let refused = limits.admit_at(&me, start + Duration::from_secs(59 * 60 + 50)).err().unwrap();
        // 12 seconds left still shows as 1 minute
        assert_eq!(refused, Refused::TooMany { limit: 3, wait_minutes: 1, guest: false });
        assert_eq!(refused.to_string(), "You've sent 3 messages this hour. Please try again in 1 min.");
    }

    #[test]
    fn refused_questions_are_not_counted() {
        let limits = Arc::new(Limits::new(2, 10));
        let pass = limits.admit(&guest()).unwrap();
        for _ in 0..5 {
            assert_eq!(limits.admit(&guest()).err(), Some(Refused::AlreadyAsking));
        }
        drop(pass);
        assert!(limits.admit(&guest()).is_ok()); // second of two
    }

    #[test]
    fn old_entries_are_pruned_when_the_map_grows() {
        let limits = Arc::new(Limits::new(5, 5));
        let start = Instant::now();
        for i in 0..=PRUNE_ABOVE {
            drop(limits.admit_at(&Who::user(i as i64), start).unwrap());
        }
        assert_eq!(limits.asked.lock().unwrap().len(), PRUNE_ABOVE + 1);
        // An hour later, the next question clears out everyone whose window is empty
        drop(limits.admit_at(&guest(), start + WINDOW).unwrap());
        assert_eq!(limits.asked.lock().unwrap().len(), 1);
    }
}
