//! The Qwen2 transformer, put together from the building blocks in ops.rs.

use std::time::{Duration, Instant};

use anyhow::Context;

use crate::gguf::Gguf;
use crate::ops::{rms_norm, rope, softmax, swiglu};
use crate::quant::QMatrix;

/// The model's shape, read from the GGUF file.
#[derive(Debug, Clone)]
pub struct Config {
    pub layers: usize,
    pub hidden: usize,   // numbers per word (1536)
    pub heads: usize,    // query heads (12)
    pub kv_heads: usize, // key/value heads (2)
    pub head_dim: usize, // numbers per head (128)
    pub eps: f32,
    pub rope_base: f32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Config> {
        let arch = g.get_str("general.architecture").unwrap_or("qwen2").to_string();
        let int = |k: &str| g.get_u32(&format!("{arch}.{k}")).map(|v| v as usize).with_context(|| format!("missing {arch}.{k}"));
        let hidden = int("embedding_length")?;
        let heads = int("attention.head_count")?;
        Ok(Config {
            layers: int("block_count")?,
            hidden,
            heads,
            kv_heads: int("attention.head_count_kv")?,
            head_dim: hidden / heads,
            eps: g.get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            rope_base: g.get_f32(&format!("{arch}.rope.freq_base")).unwrap_or(10_000.0),
        })
    }
}

/// The weights of one transformer layer (block).
pub struct Layer {
    attn_norm: Vec<f32>,
    wq: QMatrix,
    wk: QMatrix,
    wv: QMatrix,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    wo: QMatrix,
    ffn_norm: Vec<f32>,
    gate: QMatrix,
    up: QMatrix,
    down: QMatrix,
}

/// What one layer remembers about earlier words: their keys and values.
#[derive(Default)]
pub struct LayerCache {
    keys: Vec<f32>,   // position after position, each kv_heads × head_dim numbers
    values: Vec<f32>,
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Layer {
    pub fn load(g: &Gguf, i: usize) -> anyhow::Result<Layer> {
        let m = |name: &str| g.matrix(&format!("blk.{i}.{name}.weight"));
        let v = |name: &str| g.read_tensor_f32(&format!("blk.{i}.{name}"));
        Ok(Layer {
            attn_norm: v("attn_norm.weight")?,
            wq: m("attn_q")?,
            wk: m("attn_k")?,
            wv: m("attn_v")?,
            bq: v("attn_q.bias")?,
            bk: v("attn_k.bias")?,
            bv: v("attn_v.bias")?,
            wo: m("attn_output")?,
            ffn_norm: v("ffn_norm.weight")?,
            gate: m("ffn_gate")?,
            up: m("ffn_up")?,
            down: m("ffn_down")?,
        })
    }

    /// Run one word (at position `pos`) through this layer. Changes x in place.
    pub fn forward(&self, x: &mut [f32], pos: usize, cfg: &Config, cache: &mut LayerCache) {
        let hd = cfg.head_dim;
        let kv_dim = cfg.kv_heads * hd;

        // ----- Attention: look at earlier words -----
        let h = rms_norm(x, &self.attn_norm, cfg.eps);
        let mut q = self.wq.matvec(&h);
        let mut k = self.wk.matvec(&h);
        let mut v = self.wv.matvec(&h);
        add(&mut q, &self.bq);
        add(&mut k, &self.bk);
        add(&mut v, &self.bv);

        // Tell q and k where this word is (each head separately)
        for head in q.chunks_exact_mut(hd) {
            rope(head, pos, cfg.rope_base);
        }
        for head in k.chunks_exact_mut(hd) {
            rope(head, pos, cfg.rope_base);
        }

        // Remember this word's key and value for future words
        cache.keys.extend_from_slice(&k);
        cache.values.extend_from_slice(&v);
        let seen = cache.keys.len() / kv_dim; // words so far, including this one

        // Each query head shares a key/value head with its group (12 / 2 = 6 per group)
        let group = cfg.heads / cfg.kv_heads;
        let mut out = vec![0.0; cfg.heads * hd];
        for head in 0..cfg.heads {
            let kv = head / group;
            let q_head = &q[head * hd..(head + 1) * hd];

            // How well does this word match every earlier word?
            let mut scores: Vec<f32> = (0..seen)
                .map(|t| {
                    let key = &cache.keys[t * kv_dim + kv * hd..][..hd];
                    dot(q_head, key) / (hd as f32).sqrt()
                })
                .collect();
            softmax(&mut scores);

            // Mix the earlier words' values, weighted by those scores
            let out_head = &mut out[head * hd..(head + 1) * hd];
            for (t, &p) in scores.iter().enumerate() {
                let value = &cache.values[t * kv_dim + kv * hd..][..hd];
                for (o, val) in out_head.iter_mut().zip(value) {
                    *o += p * val;
                }
            }
        }
        add(x, &self.wo.matvec(&out)); // residual: add the result to what came in

        // ----- Feed-forward: think about it -----
        let h = rms_norm(x, &self.ffn_norm, cfg.eps);
        let mut gate = self.gate.matvec(&h);
        let up = self.up.matvec(&h);
        swiglu(&mut gate, &up);
        add(x, &self.down.matvec(&gate)); // residual again
    }
}

impl Layer {
    /// Run SEVERAL words (a prompt) through this layer at once. Word t sits at position
    /// start + t. Each weight matrix is read once for all the words (matmul).
    pub fn forward_batch(&self, xs: &mut [Vec<f32>], start: usize, cfg: &Config, cache: &mut LayerCache) {
        use rayon::prelude::*;
        let hd = cfg.head_dim;
        let kv_dim = cfg.kv_heads * hd;
        let n = xs.len();

        // ----- Attention -----
        let hs: Vec<Vec<f32>> = xs.iter().map(|x| rms_norm(x, &self.attn_norm, cfg.eps)).collect();
        let mut qs = self.wq.matmul(&hs);
        let mut ks = self.wk.matmul(&hs);
        let vs = self.wv.matmul(&hs);
        for t in 0..n {
            add(&mut qs[t], &self.bq);
            add(&mut ks[t], &self.bk);
            for head in qs[t].chunks_exact_mut(hd) {
                rope(head, start + t, cfg.rope_base);
            }
            for head in ks[t].chunks_exact_mut(hd) {
                rope(head, start + t, cfg.rope_base);
            }
            cache.keys.extend_from_slice(&ks[t]);
            let mut v = vs[t].clone();
            add(&mut v, &self.bv);
            cache.values.extend_from_slice(&v);
        }

        // Each word looks only at itself and the words BEFORE it (the causal rule)
        let group = cfg.heads / cfg.kv_heads;
        let cache = &*cache;
        let outs: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|t| {
                let seen = start + t + 1; // words this one may look at
                let q = &qs[t];
                let mut out = vec![0.0; cfg.heads * hd];
                for head in 0..cfg.heads {
                    let kv = head / group;
                    let q_head = &q[head * hd..(head + 1) * hd];
                    let mut scores: Vec<f32> = (0..seen)
                        .map(|p| dot(q_head, &cache.keys[p * kv_dim + kv * hd..][..hd]) / (hd as f32).sqrt())
                        .collect();
                    softmax(&mut scores);
                    let out_head = &mut out[head * hd..(head + 1) * hd];
                    for (p, &w) in scores.iter().enumerate() {
                        let value = &cache.values[p * kv_dim + kv * hd..][..hd];
                        for (o, val) in out_head.iter_mut().zip(value) {
                            *o += w * val;
                        }
                    }
                }
                out
            })
            .collect();
        for (x, o) in xs.iter_mut().zip(self.wo.matmul(&outs)) {
            add(x, &o);
        }

        // ----- Feed-forward -----
        let hs: Vec<Vec<f32>> = xs.iter().map(|x| rms_norm(x, &self.ffn_norm, cfg.eps)).collect();
        let mut gates = self.gate.matmul(&hs);
        let ups = self.up.matmul(&hs);
        for (g, u) in gates.iter_mut().zip(&ups) {
            swiglu(g, u);
        }
        for (x, d) in xs.iter_mut().zip(self.down.matmul(&gates)) {
            add(x, &d);
        }
    }
}

/// The whole model.
pub struct Model {
    pub cfg: Config,
    embed: QMatrix, // token_embd: one row per word in the vocabulary
    layers: Vec<Layer>,
    norm: Vec<f32>,
    output: QMatrix, // output.weight, or token_embd again (tied embeddings)
    cache: Vec<LayerCache>,
    pub time_layers: Duration, // total time spent in the 28 layers
    pub time_output: Duration, // total time spent turning numbers into word scores
}

impl Model {
    pub fn load(path: &str) -> anyhow::Result<Model> {
        let g = Gguf::open(path)?;
        let cfg = Config::from_gguf(&g)?;
        let layers = (0..cfg.layers).map(|i| Layer::load(&g, i)).collect::<anyhow::Result<Vec<_>>>()?;
        let embed = g.matrix("token_embd.weight")?;
        let output = if g.tensors.iter().any(|t| t.name == "output.weight") {
            g.matrix("output.weight")?
        } else {
            g.matrix("token_embd.weight")? // tied: the same table turns numbers back into words
        };
        Ok(Model {
            cache: (0..cfg.layers).map(|_| LayerCache::default()).collect(),
            norm: g.read_tensor_f32("output_norm.weight")?,
            cfg,
            embed,
            layers,
            output,
            time_layers: Duration::ZERO,
            time_output: Duration::ZERO,
        })
    }

    /// Forget the conversation (empty the KV cache).
    pub fn reset(&mut self) {
        for c in &mut self.cache {
            *c = LayerCache::default();
        }
    }

    /// Feed a whole prompt at once (positions start, start+1, ...). Much faster than
    /// one word at a time. Returns the scores for the word AFTER the last prompt word.
    pub fn forward_prompt(&mut self, tokens: &[u32], start: usize) -> Vec<f32> {
        let mut xs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                let mut x = Vec::with_capacity(self.cfg.hidden);
                self.embed.row(t as usize, &mut x);
                x
            })
            .collect();

        let started = Instant::now();
        for (layer, cache) in self.layers.iter().zip(&mut self.cache) {
            layer.forward_batch(&mut xs, start, &self.cfg, cache);
        }
        self.time_layers += started.elapsed();

        // Only the LAST word's next-word scores are needed
        let started = Instant::now();
        let last = rms_norm(xs.last().expect("empty prompt"), &self.norm, self.cfg.eps);
        let scores = self.output.matvec(&last);
        self.time_output += started.elapsed();
        scores
    }

    /// Feed one token at position `pos`; returns a score for every word in the vocabulary.
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let mut x = Vec::with_capacity(self.cfg.hidden);
        self.embed.row(token as usize, &mut x); // word number -> 1536 numbers

        let started = Instant::now();
        for (layer, cache) in self.layers.iter().zip(&mut self.cache) {
            layer.forward(&mut x, pos, &self.cfg, cache);
        }
        self.time_layers += started.elapsed();

        let started = Instant::now();
        let x = rms_norm(&x, &self.norm, self.cfg.eps);
        let scores = self.output.matvec(&x); // 1536 numbers -> 151936 word scores
        self.time_output += started.elapsed();
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reading a prompt in one batch must give the same scores as reading it word by word.
    /// Uses the real model, so it's slow: run it with
    ///   cargo test --release -p kinsin-engine -- --ignored
    #[test]
    #[ignore]
    fn batched_prompt_matches_word_by_word() {
        let path = "../models/kinsin-1.5b-v2-q4_0.gguf"; // tests run inside engine/
        if !std::path::Path::new(path).exists() {
            eprintln!("skipped: {path} not found");
            return;
        }
        let prompt = [3u32, 17, 5, 22, 9, 30, 1];
        let mut one_by_one = Model::load(path).unwrap();
        let mut expected = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            expected = one_by_one.forward(t, pos);
        }
        let mut batched = Model::load(path).unwrap();
        let got = batched.forward_prompt(&prompt, 0);
        let diff = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(diff < 1e-3, "max difference {diff}");
        // And the conversation continues identically afterwards
        let a = one_by_one.forward(11, prompt.len());
        let b = batched.forward(11, prompt.len());
        let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(diff < 1e-3, "after the prompt: max difference {diff}");
    }
}