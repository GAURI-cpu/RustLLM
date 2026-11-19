//! The conversation and the ChatML prompt format.

use anyhow::{Result, bail};

/// Sets the assistant's role and identity for the whole chat.
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// Who wrote a message.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    /// Read a role name like "user". Returns None for anything unknown.
    pub fn parse(name: &str) -> Option<Role> {
        match name {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// One message in the conversation.
pub struct Message {
    pub role: Role,
    pub text: String,
}

/// The ChatML text for one message.
fn turn(role: Role, text: &str) -> String {
    format!("<|im_start|>{}\n{}<|im_end|>\n", role.name(), text)
}

/// Like `build_prompt`, but drops the OLDEST messages until the prompt is at most
/// `max_tokens` long. The system message and the newest message are always kept.
/// `count_tokens` measures text (the engine's tokenizer in real use).
pub fn build_prompt_within(
    history: &[Message],
    max_tokens: usize,
    count_tokens: impl Fn(&str) -> Result<usize>,
) -> Result<String> {
    if history.is_empty() {
        bail!("the conversation is empty");
    }

    // Split off a leading system message: it is never dropped
    let (system, rest) = match history.first() {
        Some(m) if m.role == Role::System => (&history[..1], &history[1..]),
        _ => (&history[..0], history),
    };

    // Try with all messages, then without the oldest one, and so on
    for start in 0..rest.len() {
        let kept: Vec<&Message> = system.iter().chain(rest[start..].iter()).collect();
        let prompt = build_prompt_refs(&kept);
        if count_tokens(&prompt)? <= max_tokens {
            return Ok(prompt);
        }
    }
    bail!("your message is too long: it doesn't fit in {max_tokens} tokens, even on its own")
}

/// Turn messages into ChatML text, ending with an open assistant turn.
/// If they don't start with their own system message, ours is added.
fn build_prompt_refs(messages: &[&Message]) -> String {
    let mut prompt = String::new();
    if messages.first().map(|m| m.role) != Some(Role::System) {
        prompt.push_str(&turn(Role::System, SYSTEM_PROMPT));
    }
    for message in messages {
        prompt.push_str(&turn(message.role, &message.text));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, text: &str) -> Message {
        Message { role, text: text.to_string() }
    }

    // The whole conversation as a prompt, nothing dropped
    fn build_prompt(history: &[Message]) -> String {
        build_prompt_refs(&history.iter().collect::<Vec<_>>())
    }

    // A fake token counter for tests: 1 "token" per character
    fn chars(text: &str) -> Result<usize> {
        Ok(text.chars().count())
    }

    #[test]
    fn prompt_has_system_user_and_open_assistant_turn() {
        let prompt = build_prompt(&[msg(Role::User, "Hi")]);
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn custom_system_message_replaces_the_default() {
        let prompt = build_prompt(&[msg(Role::System, "Be brief."), msg(Role::User, "Hi")]);
        assert_eq!(prompt.matches("<|im_start|>system").count(), 1);
        assert!(prompt.contains("Be brief."));
    }

    #[test]
    fn short_conversation_is_kept_whole() {
        let history = [msg(Role::User, "one"), msg(Role::Assistant, "two"), msg(Role::User, "three")];
        let prompt = build_prompt_within(&history, 10_000, chars).unwrap();
        assert_eq!(prompt, build_prompt(&history));
    }

    #[test]
    fn oldest_messages_are_dropped_first() {
        let history = [
            msg(Role::User, &"a".repeat(200)),
            msg(Role::Assistant, &"b".repeat(200)),
            msg(Role::User, "newest question"),
        ];
        // Room for the system prompt + the newest message, but not the long old ones
        let budget = build_prompt(&history[2..]).chars().count() + 5;
        let prompt = build_prompt_within(&history, budget, chars).unwrap();
        assert!(prompt.contains("newest question"));
        assert!(!prompt.contains("aaaa"));
        assert!(!prompt.contains("bbbb"));
    }

    #[test]
    fn custom_system_message_is_never_dropped() {
        let history = [
            msg(Role::System, "Be brief."),
            msg(Role::User, &"x".repeat(500)),
            msg(Role::User, "latest"),
        ];
        let budget = build_prompt(&[msg(Role::System, "Be brief."), msg(Role::User, "latest")]).chars().count();
        let prompt = build_prompt_within(&history, budget, chars).unwrap();
        assert!(prompt.contains("Be brief."));
        assert!(prompt.contains("latest"));
        assert!(!prompt.contains("xxxx"));
    }

    #[test]
    fn a_single_huge_message_is_an_error() {
        let history = [msg(Role::User, &"z".repeat(1000))];
        assert!(build_prompt_within(&history, 100, chars).is_err());
    }
}
