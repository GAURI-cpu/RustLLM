//! The conversation and the ChatML prompt format.

/// Sets the assistant's role and identity for the whole chat.
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// Who wrote a message.
#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One message in the conversation.
pub struct Message {
    pub role: Role,
    pub text: String,
}

/// Turn the conversation into ChatML text, ending with an open assistant turn.
pub fn build_prompt(history: &[Message]) -> String {
    let mut prompt = format!("<|im_start|>system\n{}<|im_end|>\n", SYSTEM_PROMPT);
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
}