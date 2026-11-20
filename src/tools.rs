//! Live knowledge from Wikipedia.
//!
//! The model only knows what it learned in training. For questions about recent
//! things ("who won...", "latest news", "search X"), the server looks the question up
//! on Wikipedia first and hands what it found to the model with the question.
//!
//! Wikipedia is the ONLY outside website this server ever contacts.

use std::time::Duration;

use chrono::{Days, NaiveDate, Utc};
use serde::Serialize;
use serde_json::Value;

/// How long one request to Wikipedia may take.
const TIMEOUT: Duration = Duration::from_secs(6);
const USER_AGENT: &str = "kinSin/0.1 (https://github.com/GAURI-cpu)";
/// Most characters of looked-up text given to the model.
const MAX_TEXT_CHARS: usize = 1800;
const MAX_NEWS_CHARS: usize = 900;
const MAX_ARTICLE_CHARS: usize = 600;
/// A day's current-events page shorter than this is probably not filled in yet.
const MIN_NEWS_CHARS: usize = 200;

/// Words that make a question about recent things.
const TRIGGER_WORDS: &[&str] = &[
    "latest", "news", "current", "currently", "recent", "recently", "score", "headlines", "trending", "president",
    "2025", "2026", "2027",
];
const TRIGGER_PHRASES: &[&str] = &[
    "right now",
    "this week",
    "this year",
    "who won",
    "who is the",
    "who is the current",
    "what happened",
    "release date",
    "price of",
    "prime minister",
    "ceo of",
];
/// Words that ask for today's news (these also read Wikipedia's Current events page).
const NEWS_WORDS: &[&str] = &["news", "latest", "headlines", "happened", "today"];
const NEWS_PHRASES: &[&str] = &["this week", "current events"];
/// Question words that only confuse Wikipedia's search ("who is the current prime minister
/// of india" finds the PM's *office*; "prime minister of india" finds the PM).
const QUESTION_WORDS: &[&str] = &[
    "who", "what", "when", "where", "which", "why", "how", "is", "are", "was", "were", "the", "a", "an", "current",
    "currently", "latest", "recent", "recently", "right", "now", "today",
];
/// Words left out of the search.
const FILLER_WORDS: &[&str] = &[
    "hello", "hi", "hey", "hii", "please", "can", "could", "you", "u", "tell", "me", "search", "wiki", "wikipedia",
    "look", "up", "for", "kinsin", "bro", "pls", "plz",
];

/// Where a piece of looked-up information came from (shown under the answer).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Source {
    /// "news" (Current events) or "search" (an article).
    pub kind: &'static str,
    pub name: String,
    pub url: String,
}

/// What was looked up for one question.
#[derive(Debug)]
pub struct LiveData {
    pub text: String,
    pub sources: Vec<Source>,
}

impl LiveData {
    /// The message the model sees instead of the bare question.
    pub fn wrap(&self, question: &str) -> String {
        format!(
            "Information looked up on Wikipedia just now for this question:\n{}\n\n\
             Answer using the information above. If it doesn't answer the question, say so honestly. \
             Don't invent facts.\n\nQuestion: {question}",
            self.text
        )
    }
}

/// The websites used. Only Wikipedia; tests point it at a fake local server.
#[derive(Clone, Debug)]
pub struct Urls {
    pub wikipedia: String,
}

impl Default for Urls {
    fn default() -> Self {
        Urls { wikipedia: "https://en.wikipedia.org".to_string() }
    }
}

pub struct Tools {
    client: reqwest::Client,
    urls: Urls,
}

/// Lowercase, spell out "what's"/"who's", and turn punctuation into spaces. Returns the words.
fn normalize(question: &str) -> Vec<String> {
    let lower = question.to_lowercase().replace('’', "'");
    let expanded = lower.replace("what's", "what is").replace("who's", "who is");
    let cleaned: String = expanded.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
    cleaned.split_whitespace().map(String::from).collect()
}

/// Does the (normalized) text contain this phrase as whole words?
fn has_phrase(joined: &str, phrase: &str) -> bool {
    format!(" {joined} ").contains(&format!(" {phrase} "))
}

/// Should this question be looked up? If so, what to search for.
pub fn detect(question: &str) -> Option<String> {
    let words = normalize(question);
    let joined = words.join(" ");
    let first = words.first().map(String::as_str);
    let wanted = matches!(first, Some("search" | "wiki" | "wikipedia"))
        || has_phrase(&joined, "look up")
        || has_phrase(&joined, "search for")
        || words.iter().any(|w| TRIGGER_WORDS.contains(&w.as_str()))
        || TRIGGER_PHRASES.iter().any(|p| has_phrase(&joined, p));
    if !wanted {
        return None;
    }
    let query: Vec<&str> =
        words.iter().map(String::as_str).filter(|w| !FILLER_WORDS.contains(w)).collect();
    if query.is_empty() { None } else { Some(query.join(" ")) }
}

fn wants_news(query: &str) -> bool {
    let words: Vec<&str> = query.split_whitespace().collect();
    words.iter().any(|w| NEWS_WORDS.contains(w)) || NEWS_PHRASES.iter().any(|p| has_phrase(query, p))
}

/// What to type into Wikipedia's search box: the query without question words
/// (or the whole query, if that would leave nothing).
fn search_terms(query: &str) -> String {
    let terms: Vec<&str> = query.split_whitespace().filter(|w| !QUESTION_WORDS.contains(w)).collect();
    if terms.is_empty() { query.to_string() } else { terms.join(" ") }
}

/// Wikipedia's page HTML as plain lines of text: no tags, styles, scripts or edit links.
fn html_to_text(html: &str) -> String {
    let mut text = String::new();
    let mut rest = html;
    while let Some(start) = rest.find('<') {
        text.push_str(&rest[..start]);
        let Some(len) = rest[start..].find('>') else { break };
        let tag = rest[start + 1..start + len].to_ascii_lowercase();
        rest = &rest[start + len + 1..];
        let name: String = tag.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '/').collect();
        match name.as_str() {
            // Skip everything up to the closing tag
            "style" | "script" => match rest.to_ascii_lowercase().find(&format!("</{name}")) {
                Some(end) => rest = &rest[end..],
                None => rest = "",
            },
            "br" | "/p" | "/li" | "/div" | "/dt" | "/dd" | "/h1" | "/h2" | "/h3" | "/h4" | "/h5" | "/h6" | "/tr" => {
                text.push('\n')
            }
            _ => {}
        }
    }
    text.push_str(rest);
    decode_entities(&text)
        .replace('\u{a0}', " ")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !matches!(*line, "edit" | "history" | "watch"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// &amp; &lt; &#39; &#x2013; and friends.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let decoded = rest[1..].find(';').filter(|&end| end <= 10).and_then(|end| {
            let entity = &rest[1..=end];
            let c = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                _ => match entity.strip_prefix("#x").or_else(|| entity.strip_prefix("#X")) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok().and_then(char::from_u32),
                    None => entity.strip_prefix('#').and_then(|n| n.parse().ok()).and_then(char::from_u32),
                },
            };
            c.map(|c| (c, end + 2))
        });
        match decoded {
            Some((c, used)) => {
                out.push(c);
                rest = &rest[used..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// At most `max` characters, with "…" if anything was cut.
fn clip(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    format!("{}…", cut.trim_end())
}

impl Tools {
    pub fn new(urls: Urls) -> Self {
        let client = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .user_agent(USER_AGENT)
            .build()
            .expect("the HTTP client can always be built with these settings");
        Tools { client, urls }
    }

    /// Look the question up if it needs it. None = an ordinary question, answer it as usual.
    pub async fn gather(&self, question: &str) -> Option<LiveData> {
        let query = detect(question)?;
        let live = match self.search(&query).await {
            Ok(live) => live,
            Err(e) => {
                eprintln!("Wikipedia lookup failed: {e}");
                LiveData {
                    text: format!("Wikipedia could not be reached right now ({e}). Tell the user."),
                    sources: Vec::new(),
                }
            }
        };
        Some(LiveData { text: clip(&live.text, MAX_TEXT_CHARS), sources: live.sources })
    }

    /// Search Wikipedia: today's Current events for news questions, then the top two articles.
    pub async fn search(&self, query: &str) -> Result<LiveData, reqwest::Error> {
        let mut text = format!("From Wikipedia, for \"{query}\" (it may not have the last few hours yet):");
        let mut sources = Vec::new();

        if wants_news(query) {
            // Not fatal: the articles below may still help
            match self.current_events(Utc::now().date_naive()).await {
                Ok(Some((extract, source))) => {
                    text.push_str(&format!("\n\n{}:\n{}", source.name, clip(&extract, MAX_NEWS_CHARS)));
                    sources.push(source);
                }
                Ok(None) => {}
                Err(e) => eprintln!("Wikipedia current events failed: {e}"),
            }
        }

        let base = &self.urls.wikipedia;
        let found = self
            .get_json(&format!(
                "{base}/w/api.php?action=query&list=search&srlimit=2&format=json&utf8=1&srsearch={}",
                urlencoding::encode(&search_terms(query))
            ))
            .await?;
        let titles: Vec<String> = found["query"]["search"]
            .as_array()
            .map(|hits| hits.iter().filter_map(|h| h["title"].as_str().map(String::from)).take(2).collect())
            .unwrap_or_default();

        let mut articles = 0;
        for title in titles {
            let path = urlencoding::encode(&title.replace(' ', "_")).into_owned();
            let summary = match self.get_json(&format!("{base}/api/rest_v1/page/summary/{path}")).await {
                Ok(summary) => summary,
                Err(e) => {
                    eprintln!("Wikipedia summary of {title:?} failed: {e}");
                    continue;
                }
            };
            let extract = summary["extract"].as_str().unwrap_or("").trim();
            if extract.is_empty() {
                continue;
            }
            let url = summary["content_urls"]["desktop"]["page"]
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| format!("{base}/wiki/{path}"));
            text.push_str(&format!("\n\n{title}:\n{}", clip(extract, MAX_ARTICLE_CHARS)));
            sources.push(Source { kind: "search", name: format!("Wikipedia: {title}"), url });
            articles += 1;
        }

        if articles == 0 && sources.is_empty() {
            text.push_str(&format!(
                "\n\nWikipedia has nothing about \"{query}\". Tell the user you couldn't find it there."
            ));
        }
        Ok(LiveData { text, sources })
    }

    /// The day's "Portal:Current_events" page (or yesterday's, if today's is still nearly empty).
    /// These pages are built from templates, which Wikipedia's plain-text extracts leave out
    /// (they come back empty), so the rendered page is fetched and its HTML turned into text.
    async fn current_events(&self, today: NaiveDate) -> Result<Option<(String, Source)>, reqwest::Error> {
        let yesterday = today.checked_sub_days(Days::new(1)).unwrap_or(today);
        for day in [today, yesterday] {
            let title = format!("Portal:Current_events/{}", day.format("%Y_%B_%-d"));
            let base = &self.urls.wikipedia;
            let page = self
                .get_json(&format!(
                    "{base}/w/api.php?action=parse&prop=text&format=json&formatversion=2\
                     &disablelimitreport=1&disableeditsection=1&page={}",
                    urlencoding::encode(&title)
                ))
                .await?;
            // A day nobody has written yet answers with an error ("missingtitle"): treat it as empty
            let extract = html_to_text(page["parse"]["text"].as_str().unwrap_or(""));
            if extract.chars().count() >= MIN_NEWS_CHARS {
                let source = Source {
                    kind: "news",
                    name: format!("Wikipedia: Current events {}", day.format("%-d %b")),
                    url: format!("{base}/wiki/{title}"),
                };
                return Ok(Some((extract, source)));
            }
        }
        Ok(None)
    }

    async fn get_json(&self, url: &str) -> Result<Value, reqwest::Error> {
        self.client.get(url).send().await?.error_for_status()?.json().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use axum::extract::{Path, Query};
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::json;

    #[test]
    fn recent_things_are_looked_up() {
        let cases = [
            ("Who is the current prime minister of India?", "who is the current prime minister of india"),
            ("latest news today", "latest news today"),
            ("search Rust programming language", "rust programming language"),
            ("hey kinsin can you look up the Eiffel Tower pls", "the eiffel tower"),
            ("Wikipedia: Ada Lovelace", "ada lovelace"),
            ("What's the release date of GTA 6?", "what is the release date of gta 6"),
            ("who won the world cup in 2026", "who won the world cup in 2026"),
            ("Who’s the CEO of OpenAI", "who is the ceo of openai"),
        ];
        for (question, query) in cases {
            assert_eq!(detect(question).as_deref(), Some(query), "{question}");
        }
    }

    #[test]
    fn normal_questions_are_not_looked_up() {
        for question in [
            "explain recursion",
            "write a haiku about rain",
            "what is the time complexity of quicksort?",
            "who are you",
            "weather in bangalore",
            "hello",
            "researcher tips", // "search" inside a word doesn't count
            "search",          // nothing left to search for
        ] {
            assert_eq!(detect(question), None, "{question}");
        }
    }

    #[test]
    fn news_words_read_current_events() {
        assert!(wants_news("latest news today"));
        assert!(wants_news("what happened this week"));
        assert!(!wants_news("rust programming language"));
    }

    #[test]
    fn wrap_puts_the_question_last() {
        let live = LiveData { text: "Paris is in France.".into(), sources: vec![] };
        let wrapped = live.wrap("where is Paris?");
        assert!(wrapped.starts_with("Information looked up on Wikipedia just now for this question:\nParis is in France."));
        assert!(wrapped.contains("Don't invent facts."));
        assert!(wrapped.ends_with("\n\nQuestion: where is Paris?"));
    }

    #[test]
    fn question_words_stay_out_of_the_search_box() {
        assert_eq!(search_terms("who is the current prime minister of india"), "prime minister of india");
        assert_eq!(search_terms("rust programming language"), "rust programming language");
        assert_eq!(search_terms("who won the world cup in 2026"), "won world cup in 2026");
        assert_eq!(search_terms("what is the latest"), "what is the latest"); // nothing would be left
    }

    #[test]
    fn html_becomes_plain_lines() {
        let html = "<h2>Sports</h2><style>p{x:1}</style><SCRIPT>alert(1)</SCRIPT>\
                    <ul><li><a href=\"/wiki/A\">A&nbsp;team</a> wins &amp; <b>celebrates</b> &#8211; again&#x21;</li>\
                    <li>edit</li><li>Fish &chips; 5 &lt; 6</li></ul>";
        assert_eq!(html_to_text(html), "Sports\nA team wins & celebrates – again!\nFish &chips; 5 < 6");
    }

    #[test]
    fn clip_cuts_on_characters() {
        assert_eq!(clip("héllo", 10), "héllo");
        assert_eq!(clip("héllo wörld", 5), "héllo…");
    }

    /// A tiny stand-in for Wikipedia on a free local port.
    async fn fake_wikipedia() -> String {
        async fn api(Query(q): Query<HashMap<String, String>>) -> Json<Value> {
            if q.get("list").map(String::as_str) == Some("search") {
                assert_eq!(q.get("srlimit").map(String::as_str), Some("2"));
                return Json(json!({ "query": { "search": [
                    { "title": "Rust (programming language)" },
                    { "title": "Cargo (package manager)" }
                ] } }));
            }
            assert_eq!(q.get("action").map(String::as_str), Some("parse"));
            let title = q.get("page").cloned().unwrap_or_default();
            assert!(title.starts_with("Portal:Current_events/"), "{title}");
            let html = format!(
                "<div class=\"p-current-events\"><style>.x{{color:red}}</style><span>edit</span>\
                 <ul>{}</ul></div>",
                "<li><a href=\"/wiki/X\">Big event</a> happened &amp; more.</li>".repeat(20)
            );
            Json(json!({ "parse": { "title": title, "text": html } }))
        }
        async fn summary(Path(title): Path<String>) -> Json<Value> {
            Json(json!({
                "title": title,
                "extract": format!("All about {title}."),
                "content_urls": { "desktop": { "page": format!("https://en.wikipedia.org/wiki/{title}") } }
            }))
        }
        let app = Router::new().route("/w/api.php", get(api)).route("/api/rest_v1/page/summary/{title}", get(summary));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn reads_the_top_two_articles() {
        let tools = Tools::new(Urls { wikipedia: fake_wikipedia().await });
        let live = tools.gather("search Rust programming language").await.unwrap();
        assert!(live.text.starts_with("From Wikipedia, for \"rust programming language\""), "{}", live.text);
        assert!(live.text.contains("All about Rust_(programming_language)."), "{}", live.text);
        assert!(live.text.contains("All about Cargo_(package_manager)."), "{}", live.text);
        assert!(!live.text.contains("Big event"), "not a news question: {}", live.text);
        let names: Vec<&str> = live.sources.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["Wikipedia: Rust (programming language)", "Wikipedia: Cargo (package manager)"]);
        assert_eq!(live.sources[0].url, "https://en.wikipedia.org/wiki/Rust_(programming_language)");
        assert!(live.sources.iter().all(|s| s.kind == "search"));
    }

    #[tokio::test]
    async fn news_questions_also_read_current_events() {
        let tools = Tools::new(Urls { wikipedia: fake_wikipedia().await });
        let live = tools.gather("latest news today").await.unwrap();
        assert!(live.text.contains("Big event happened & more.\nBig event"), "{}", live.text);
        assert!(!live.text.contains("color:red") && !live.text.contains("<"), "{}", live.text);
        assert!(live.text.chars().count() <= MAX_TEXT_CHARS + 1, "capped");
        assert_eq!(live.sources[0].kind, "news");
        assert!(live.sources[0].name.starts_with("Wikipedia: Current events "), "{:?}", live.sources[0]);
        assert!(live.sources[0].url.contains("/wiki/Portal:Current_events/"), "{:?}", live.sources[0]);
        assert_eq!(live.sources.len(), 3);
    }

    #[tokio::test]
    async fn unreachable_wikipedia_gives_an_honest_note() {
        // A port that was free a moment ago: nothing is listening there now
        let address = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap();
        let tools = Tools::new(Urls { wikipedia: format!("http://{address}") });
        let live = tools.gather("latest news today").await.unwrap();
        assert!(live.text.starts_with("Wikipedia could not be reached right now ("), "{}", live.text);
        assert!(live.text.ends_with("Tell the user."), "{}", live.text);
        assert!(live.sources.is_empty());
    }

    #[tokio::test]
    async fn ordinary_questions_never_touch_the_network() {
        let tools = Tools::new(Urls { wikipedia: "http://127.0.0.1:1".into() });
        assert!(tools.gather("explain recursion").await.is_none());
    }
}
