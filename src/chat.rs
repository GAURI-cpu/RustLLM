//! The conversation and the ChatML prompt format.

/// Sets the assistant's role and identity for the whole chat.
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// Who wrote a message.
#[derive(Clone, Copy, PartialEq)]
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

/// Turn the conversation into ChatML text, ending with an open assistant turn.
/// If the conversation doesn't start with its own system message, ours is added.
pub fn build_prompt(history: &[Message]) -> String {
    let mut prompt = String::new();
    if history.first().map(|m| m.role) != Some(Role::System) {
        prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", SYSTEM_PROMPT));
    }
    for message in history {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role.name(),
            message.text
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_has_system_user_and_open_assistant_turn() {
        let history = vec![Message { role: Role::User, text: "Hi".to_string() }];
        let prompt = build_prompt(&history);
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn custom_system_message_replaces_the_default() {
        let history = vec![
            Message { role: Role::System, text: "Be brief.".to_string() },
            Message { role: Role::User, text: "Hi".to_string() },
        ];
        let prompt = build_prompt(&history);
        assert_eq!(prompt.matches("<|im_start|>system").count(), 1);
        assert!(prompt.contains("Be brief."));
    }
}