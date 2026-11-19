//! Speech to text with OpenAI's open Whisper model, run on the CPU with candle.
//!
//! Needs three files from huggingface.co/openai/whisper-<size> in one folder
//! (see scripts/download_whisper.*): config.json, tokenizer.json, model.safetensors.

use std::collections::HashSet;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as w, Config, audio, model::Whisper};
use tokenizers::Tokenizer;

const FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];
/// Longest recording accepted, in seconds.
pub const MAX_SECONDS: usize = 120;
/// Quieter than this (RMS) counts as silence: Whisper invents text ("Thank you.") for silence.
const SILENCE_RMS: f32 = 0.003;
/// Most tokens written for one 30-second piece.
const MAX_STEPS: usize = 200;
const NOT_A_RECORDING: &str = "this is not a WAV recording kinSin can read";

/// Whisper's languages, in Whisper's order (the order matters to the model's token ids).
pub const LANGUAGES: [&str; 99] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

pub struct Voice {
    model: Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    sot: u32,
    transcribe: u32,
    no_timestamps: u32,
    eot: u32,
    /// (language code, token id) for every language the tokenizer knows
    languages: Vec<(&'static str, u32)>,
    suppress: HashSet<u32>,
}

impl Voice {
    /// Are all three Whisper files in `dir`?
    pub fn files_present(dir: &Path) -> bool {
        FILES.iter().all(|f| dir.join(f).is_file())
    }

    pub fn load(dir: &Path) -> Result<Voice> {
        let config_path = dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("could not read the Whisper config {}", config_path.display()))?;
        let config: Config = serde_json::from_str(&config_text)
            .with_context(|| format!("{} is not a Whisper config", config_path.display()))?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("could not load the Whisper tokenizer {}: {e}", tokenizer_path.display()))?;

        let weights_path = dir.join("model.safetensors");
        // Safety: the weights file is memory-mapped; it must not change while the server runs.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], w::DTYPE, &Device::Cpu) }
            .with_context(|| format!("could not open the Whisper weights {}", weights_path.display()))?;
        let model = Whisper::load(&vb, config.clone()).context("could not load the Whisper model")?;

        let token = |name: &str| {
            tokenizer.token_to_id(name).ok_or_else(|| anyhow!("the Whisper tokenizer has no {name} token"))
        };
        let languages = LANGUAGES
            .iter()
            .filter_map(|&code| tokenizer.token_to_id(&format!("<|{code}|>")).map(|id| (code, id)))
            .collect::<Vec<_>>();
        if languages.is_empty() {
            bail!("the Whisper tokenizer has no language tokens (is this an English-only model?)");
        }

        Ok(Voice {
            sot: token(w::SOT_TOKEN)?,
            transcribe: token(w::TRANSCRIBE_TOKEN)?,
            no_timestamps: token(w::NO_TIMESTAMPS_TOKEN)?,
            eot: token(w::EOT_TOKEN)?,
            suppress: config.suppress_tokens.iter().copied().collect(),
            mel_filters: mel_filters(w::SAMPLE_RATE, w::N_FFT, config.num_mel_bins),
            languages,
            tokenizer,
            config,
            model,
        })
    }

    /// Turn speech (16 kHz mono samples) into text. Returns (text, language code).
    /// `language`: None = detect it.
    pub fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<(String, String)> {
        if samples.len() > MAX_SECONDS * w::SAMPLE_RATE {
            bail!("the recording is too long (at most {MAX_SECONDS} seconds)");
        }
        if is_silent(samples) {
            return Ok((String::new(), language.unwrap_or("").to_string()));
        }
        let lang_token = match language {
            Some(code) => self.language_token(code)?,
            None => 0, // detected on the first piece
        };

        let n_mels = self.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.config, samples, &self.mel_filters);
        let total_frames = mel.len() / n_mels;
        let mel = Tensor::from_vec(mel, (1, n_mels, total_frames), &Device::Cpu)?;

        let mut lang_token = lang_token;
        let mut pieces = Vec::new();
        let content_frames = samples.len() / w::HOP_LENGTH;
        let mut seek = 0;
        // Whisper hears 30 seconds at a time
        while seek < content_frames {
            let segment = mel.narrow(2, seek, w::N_FRAMES.min(total_frames - seek))?;
            let features = self.model.encoder.forward(&segment, true)?;
            if lang_token == 0 {
                lang_token = self.detect_language(&features)?;
            }
            let text = self.decode(&features, lang_token)?;
            if !text.is_empty() {
                pieces.push(text);
            }
            seek += w::N_FRAMES;
        }

        let code = self.languages.iter().find(|(_, id)| *id == lang_token).map(|(c, _)| *c).unwrap_or("");
        Ok((pieces.join(" "), code.to_string()))
    }

    fn language_token(&self, code: &str) -> Result<u32> {
        self.languages
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, id)| *id)
            .ok_or_else(|| anyhow!("unknown language '{code}'"))
    }

    /// Which language is spoken: the language token Whisper rates highest right after <|startoftranscript|>.
    fn detect_language(&mut self, features: &Tensor) -> Result<u32> {
        let tokens = Tensor::new(&[[self.sot]], &Device::Cpu)?;
        let hidden = self.model.decoder.forward(&tokens, features, true)?;
        let logits = self.model.decoder.final_linear(&hidden.i((.., 0..1))?)?.i(0)?.i(0)?;
        let logits: Vec<f32> = logits.to_vec1()?;
        let best = self
            .languages
            .iter()
            .max_by(|a, b| logits[a.1 as usize].total_cmp(&logits[b.1 as usize]))
            .map(|(_, id)| *id)
            .ok_or_else(|| anyhow!("no languages"))?;
        Ok(best)
    }

    /// Greedy decoding of one 30-second piece.
    fn decode(&mut self, features: &Tensor, lang_token: u32) -> Result<String> {
        let mut tokens = vec![self.sot, lang_token, self.transcribe, self.no_timestamps];
        let prompt_len = tokens.len();
        for step in 0..MAX_STEPS {
            let input = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let hidden = self.model.decoder.forward(&input, features, step == 0)?;
            let (_, len, _) = hidden.dims3()?;
            let logits: Vec<f32> = self.model.decoder.final_linear(&hidden.i((.., len - 1..))?)?.i(0)?.i(0)?.to_vec1()?;

            // Best text token (ids below <|endoftext|>, minus the suppressed ones), or the end
            let mut next = self.eot;
            let mut best = logits[self.eot as usize];
            for (id, &score) in logits.iter().enumerate().take(self.eot as usize) {
                if score > best && !self.suppress.contains(&(id as u32)) {
                    best = score;
                    next = id as u32;
                }
            }
            if next == self.eot {
                break;
            }
            // Stuck repeating itself: stop before a 4th copy in a row
            let n = tokens.len();
            if n >= prompt_len + 3 && tokens[n - 3..].iter().all(|&t| t == next) {
                break;
            }
            tokens.push(next);
        }
        let text = self
            .tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|e| anyhow!("could not decode the transcript: {e}"))?;
        Ok(text.trim().to_string())
    }
}

/// Whisper's mel filter bank (Slaney scale, Slaney normalisation), computed instead of downloaded.
fn mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp };
    let mel_to_hz = |m: f64| if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m };
    let n_freqs = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs).map(|k| k as f64 * sample_rate as f64 / n_fft as f64).collect();
    let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(sample_rate as f64 / 2.0));
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut filters = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let (left, center, right) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
        let norm = 2.0 / (right - left);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let rising = (f - left) / (center - left);
            let falling = (right - f) / (right - center);
            filters[m * n_freqs + k] = (rising.min(falling).max(0.0) * norm) as f32;
        }
    }
    filters
}

/// Root-mean-square loudness of the samples (0 = silence, 1 = full scale).
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn is_silent(samples: &[f32]) -> bool {
    rms(samples) < SILENCE_RMS
}

/// Read a WAV file into 16 kHz mono samples in -1..1 (what Whisper expects).
pub fn wav_to_samples(bytes: &[u8]) -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|_| anyhow!(NOT_A_RECORDING))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| anyhow!(NOT_A_RECORDING))?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| anyhow!(NOT_A_RECORDING))?,
    };
    // Mix all channels into one
    let mono: Vec<f32> = interleaved.chunks(channels).map(|frame| frame.iter().sum::<f32>() / frame.len() as f32).collect();
    Ok(resample(&mono, spec.sample_rate as usize, w::SAMPLE_RATE))
}

/// Linear-interpolation resampling (good enough for speech).
fn resample(samples: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || samples.is_empty() || from == 0 {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to as u64 / from as u64) as usize;
    let step = from as f64 / to as f64;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * step;
            let left = pos.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (pos - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(sample_rate: u32, channels: u16, seconds: f32, amplitude: f32) -> Vec<u8> {
        let spec = hound::WavSpec { channels, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut bytes = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut bytes, spec).unwrap();
        let frames = (sample_rate as f32 * seconds) as usize;
        for i in 0..frames {
            let v = (amplitude * (i as f32 * 440.0 * std::f32::consts::TAU / sample_rate as f32).sin() * 32767.0) as i16;
            for _ in 0..channels {
                writer.write_sample(v).unwrap();
            }
        }
        writer.finalize().unwrap();
        bytes.into_inner()
    }

    #[test]
    fn mel_filters_match_whisper() {
        let filters = mel_filters(w::SAMPLE_RATE, w::N_FFT, 80);
        assert_eq!(filters.len(), 80 * 201);
        let max = filters.iter().cloned().fold(0f32, f32::max);
        assert!((max - 0.0259).abs() < 0.0005, "max was {max}");
    }

    #[test]
    fn stereo_48k_wav_becomes_16k_mono() {
        let samples = wav_to_samples(&wav(48_000, 2, 1.0, 0.5)).unwrap();
        assert_eq!(samples.len(), 16_000);
        let peak = samples.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
        assert!((0.45..=0.51).contains(&peak), "peak was {peak}");
    }

    #[test]
    fn silence_is_detected() {
        assert!(is_silent(&vec![0.0; 16_000]));
        assert!(is_silent(&vec![0.001; 16_000]));
        assert!(!is_silent(&wav_to_samples(&wav(16_000, 1, 1.0, 0.2)).unwrap()));
        assert!(is_silent(&[]));
    }

    #[test]
    fn non_wav_bytes_are_rejected() {
        let err = wav_to_samples(b"definitely not audio").unwrap_err();
        assert!(err.to_string().contains("not a WAV"));
        assert!(wav_to_samples(&[]).is_err());
    }

    #[test]
    fn missing_files_are_detected() {
        assert!(!Voice::files_present(Path::new("no/such/folder")));
    }

    #[test]
    fn language_list_is_whispers() {
        assert_eq!(LANGUAGES.len(), 99);
        assert_eq!(LANGUAGES[0], "en");
        assert_eq!(LANGUAGES[17], "hi");
        assert_eq!(LANGUAGES[98], "su");
    }
}
