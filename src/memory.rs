//! Memory: small facts about a logged-in user, picked out of what they say,
//! and given to the model in later chats.
//!
//!   "remember that my sister is Riya"   -> "My sister is Riya."
//!   "Hi, my name is Gauri."             -> "The user's name is Gauri."
//!   "I live in Pune and I love chess"   -> "The user lives in Pune and I love chess."
//!
//! Only plain statements count: questions, and sentences about "this"/"that"/"it"
//! ("I like that answer") are ignored, so chatting about the chat isn't saved.

/// Longest fact, in characters.
pub const MAX_FACT_CHARS: usize = 300;
/// Most facts kept per user.
pub const MAX_FACTS: usize = 100;
/// How many (newest) facts go into the system message.
const FACTS_IN_PROMPT: usize = 20;

/// "I like X" etc.: (how the sentence starts, how the fact starts).
const RULES: &[(&str, &str)] = &[
    ("my name is ", "The user's name is "),
    ("call me ", "The user wants to be called "),
    ("i am studying ", "The user is studying "),
    ("i'm studying ", "The user is studying "),
    ("i am a ", "The user is a "),
    ("i am an ", "The user is an "),
    ("i'm a ", "The user is a "),
    ("i'm an ", "The user is an "),
    ("i live in ", "The user lives in "),
    ("i am from ", "The user is from "),
    ("i'm from ", "The user is from "),
    ("i work as ", "The user works as "),
    ("i work at ", "The user works at "),
    ("i study ", "The user studies "),
    ("i prefer ", "The user prefers "),
    ("i like ", "The user likes "),
    ("i love ", "The user loves "),
    ("my favourite ", "The user's favourite "),
    ("my favorite ", "The user's favorite "),
];
/// A fact can't be about the conversation itself ("I love it", "I like that answer").
const NOT_A_FACT: &[&str] = &["that", "this", "it", "your", "you", "the", "these", "those", "how", "when", "to"];
/// Names are short: "my name is Gauri and I code" keeps "Gauri and I code" at most 4 words.
const MAX_NAME_WORDS: usize = 4;

/// The facts worth remembering in one message (usually none).
pub fn facts_in(message: &str) -> Vec<String> {
    let message = message.trim().replace('’', "'");

    // Asked outright: "remember that X", "remember: X", "please remember X"
    let asked = strip_ci(&message, "please ")
        .and_then(|rest| strip_ci(rest, "remember that ").or_else(|| strip_ci(rest, "remember")))
        .or_else(|| strip_ci(&message, "remember that "))
        .or_else(|| strip_ci(&message, "remember:"));
    if let Some(fact) = asked {
        let fact = fact.trim_start_matches([':', ' ']);
        return finish("", fact).into_iter().collect();
    }

    let mut facts: Vec<String> = Vec::new();
    for sentence in message.split(['.', '!', '\n', ';']) {
        if sentence.contains('?') {
            continue;
        }
        let sentence = drop_greeting(sentence.trim());
        for (start, fact_start) in RULES {
            let Some(rest) = strip_ci(sentence, start) else { continue };
            let rest = if *start == "my name is " || *start == "call me " {
                // Just the name: up to "and"/"but"/a comma, and 4 words at most
                let name = rest.split(',').next().unwrap_or("");
                name.split_whitespace()
                    .take_while(|w| !w.eq_ignore_ascii_case("and") && !w.eq_ignore_ascii_case("but"))
                    .take(MAX_NAME_WORDS)
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                rest.to_string()
            };
            if let Some(fact) = finish(fact_start, &rest) {
                if !facts.iter().any(|f| f.eq_ignore_ascii_case(&fact)) {
                    facts.push(fact);
                }
            }
            break;
        }
    }
    facts
}

/// The system message plus what is remembered about the user (newest first).
pub fn system_with_facts(system: &str, facts: &[String]) -> String {
    if facts.is_empty() {
        return system.to_string();
    }
    let mut text = format!("{system}\n\nWhat you remember about this user from earlier chats (use it only when it helps):");
    for fact in facts.iter().take(FACTS_IN_PROMPT) {
        text.push_str("\n- ");
        text.push_str(fact);
    }
    text
}

/// `text` without `prefix` at its start, ignoring ASCII case.
fn strip_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then(|| &text[prefix.len()..])
}

/// "Hi, my name is..." -> "my name is..."
fn drop_greeting(sentence: &str) -> &str {
    for greeting in ["hello", "hey", "hi"] {
        if let Some(rest) = strip_ci(sentence, greeting) {
            // Only a whole word: "hiking" is not a greeting
            if rest.is_empty() || rest.starts_with(|c: char| !c.is_alphanumeric()) {
                return rest.trim_start_matches(|c: char| !c.is_alphanumeric());
            }
        }
    }
    sentence
}

/// Turn the rest of a sentence into a fact: tidy it, refuse non-facts, end with a full stop.
fn finish(fact_start: &str, rest: &str) -> Option<String> {
    let rest = rest.trim().trim_end_matches(|c: char| c.is_whitespace() || ".,!?;:".contains(c));
    let first_word = rest.split_whitespace().next()?.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    // Asked outright ("remember that this weekend...") is always kept; found facts are checked
    if first_word.is_empty() || (!fact_start.is_empty() && NOT_A_FACT.contains(&first_word.as_str())) {
        return None;
    }
    let mut fact = if fact_start.is_empty() {
        // Asked to remember: keep it as the user wrote it, with a capital first letter
        let mut chars = rest.chars();
        let first = chars.next()?;
        first.to_uppercase().chain(chars).collect()
    } else {
        format!("{fact_start}{rest}")
    };
    if fact.chars().count() >= MAX_FACT_CHARS {
        fact = fact.chars().take(MAX_FACT_CHARS - 1).collect::<String>().trim_end().to_string();
    }
    fact.push('.');
    Some(fact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asked_to_remember() {
        assert_eq!(facts_in("remember that my sister is Riya"), ["My sister is Riya."]);
        assert_eq!(facts_in("Remember: I use NeoVim!"), ["I use NeoVim."]);
        assert_eq!(facts_in("please remember my exam is on 3 March"), ["My exam is on 3 March."]);
        assert_eq!(facts_in("Please remember that I'm vegetarian."), ["I'm vegetarian."]);
        assert_eq!(facts_in("remember that this weekend I have exams"), ["This weekend I have exams."]);
        assert!(facts_in("remember that?").is_empty());
        assert!(facts_in("remember").is_empty());
    }

    #[test]
    fn plain_statements_become_facts() {
        assert_eq!(facts_in("Hi, my name is Gauri."), ["The user's name is Gauri."]);
        assert_eq!(facts_in("hey my name is Gauri Singla and I build things"), ["The user's name is Gauri Singla."]);
        assert_eq!(facts_in("my name is Ana Maria de la Cruz"), ["The user's name is Ana Maria de la."]);
        assert_eq!(facts_in("call me G"), ["The user wants to be called G."]);
        assert_eq!(facts_in("I’m a student in Delhi"), ["The user is a student in Delhi."]);
        assert_eq!(facts_in("I am an engineer"), ["The user is an engineer."]);
        assert_eq!(facts_in("I live in Pune. I work at Infosys!"), ["The user lives in Pune.", "The user works at Infosys."]);
        assert_eq!(facts_in("i'm from Punjab"), ["The user is from Punjab."]);
        assert_eq!(facts_in("I work as a nurse"), ["The user works as a nurse."]);
        assert_eq!(facts_in("I study computer science"), ["The user studies computer science."]);
        assert_eq!(facts_in("I am studying for GATE"), ["The user is studying for GATE."]);
        assert_eq!(facts_in("I prefer short answers"), ["The user prefers short answers."]);
        assert_eq!(facts_in("I like Rust; I love chess"), ["The user likes Rust.", "The user loves chess."]);
        assert_eq!(facts_in("My favourite colour is blue"), ["The user's favourite colour is blue."]);
        assert_eq!(facts_in("my favorite food is dosa"), ["The user's favorite food is dosa."]);
    }

    #[test]
    fn chat_about_the_chat_is_not_saved() {
        for message in [
            "I love it, thanks",
            "I like that answer",
            "I am tired",
            "Am I a good coder?",
            "I like this!",
            "I prefer your first version",
            "I love how you explained it",
            "I like to code", // "to ..." is too vague
            "Do you know that my name is Gauri?",
            "explain recursion",
            "hiking is fun",
            "",
        ] {
            assert!(facts_in(message).is_empty(), "{message:?} -> {:?}", facts_in(message));
        }
    }

    #[test]
    fn questions_are_skipped_but_other_sentences_count() {
        assert_eq!(facts_in("My name is Gauri. What's yours?"), ["The user's name is Gauri."]);
    }

    #[test]
    fn facts_are_capped() {
        let long = format!("I like {}", "a".repeat(500));
        let facts = facts_in(&long);
        assert_eq!(facts[0].chars().count(), MAX_FACT_CHARS);
        assert!(facts[0].ends_with('.'));
    }

    #[test]
    fn system_message_lists_the_newest_facts() {
        assert_eq!(system_with_facts("Be nice.", &[]), "Be nice.");
        let facts: Vec<String> = (0..30).map(|i| format!("Fact {i}.")).collect();
        let system = system_with_facts("Be nice.", &facts);
        assert!(system.starts_with(
            "Be nice.\n\nWhat you remember about this user from earlier chats (use it only when it helps):\n- Fact 0.\n- Fact 1."
        ));
        assert!(system.contains("- Fact 19.") && !system.contains("Fact 20."));
    }
}
