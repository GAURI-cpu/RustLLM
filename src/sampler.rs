//! Choosing the next token from the model's scores.

/// Settings that control how the next token is picked.
pub struct SamplingOptions {
    /// 0.0 = always the most likely token; higher = more creative.
    pub temperature: f32,
    /// Only the best `top_k` tokens can be picked.
    pub top_k: usize,
    /// Above 1.0 makes recently used tokens less likely.
    pub repeat_penalty: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, repeat_penalty: 1.1 }
    }
}

/// A tiny random number generator (xorshift).
pub struct Rng(u64);

impl Rng {
    pub fn from_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Rng(nanos | 1)
    }

    /// A random number between 0.0 and 1.0.
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Pick the next token: repeat penalty -> temperature -> top-k -> weighted random pick.
pub fn sample(logits: &[f32], history: &[u32], opts: &SamplingOptions, rng: &mut Rng) -> u32 {
    let mut scores = logits.to_vec();

    // 1. Repeat penalty: make recently used tokens less likely
    let start = history.len().saturating_sub(64);
    for &token in &history[start..] {
        let s = &mut scores[token as usize];
        if *s > 0.0 { *s /= opts.repeat_penalty; } else { *s *= opts.repeat_penalty; }
    }

    // temperature 0 = no randomness, just take the best
    if opts.temperature <= 0.0 {
        return argmax(&scores);
    }

    // 2. Temperature, then 3. keep only the top-k candidates
    let mut candidates: Vec<(usize, f32)> =
        scores.iter().map(|s| s / opts.temperature).enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    candidates.truncate(opts.top_k.max(1));

    // 4. Softmax weights
    let max = candidates[0].1;
    let weights: Vec<f32> = candidates.iter().map(|(_, s)| (s - max).exp()).collect();
    let total: f32 = weights.iter().sum();

    // 5. Spin the wheel
    let mut r = rng.next_f32() * total;
    for (i, w) in weights.iter().enumerate() {
        if r < *w {
            return candidates[i].0 as u32;
        }
        r -= w;
    }
    candidates[0].0 as u32
}

/// The position of the highest score = the most likely token id.
pub fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_finds_the_biggest() {
        assert_eq!(argmax(&[0.1, 5.0, 2.0]), 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let opts = SamplingOptions { temperature: 0.0, top_k: 40, repeat_penalty: 1.0 };
        let mut rng = Rng::from_time();
        for _ in 0..10 {
            assert_eq!(sample(&[1.0, 3.0, 2.0], &[], &opts, &mut rng), 1);
        }
    }

    #[test]
    fn top_k_one_always_picks_the_best() {
        let opts = SamplingOptions { temperature: 1.0, top_k: 1, repeat_penalty: 1.0 };
        let mut rng = Rng::from_time();
        for _ in 0..10 {
            assert_eq!(sample(&[1.0, 3.0, 2.0], &[], &opts, &mut rng), 1);
        }
    }
}