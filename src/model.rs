//! Whisper encoder-decoder in plain burn ops: `matmul`, `module::layer_norm`, `module::attention`,
//! `activation::softmax`, `conv1d` (the ops Partii's optional WGSL kernels replace). Weights: the Hugging Face
//! `openai/whisper-*` tensor names. The architecture follows OpenAI's Whisper (MIT); see NOTICE.
use crate::decode_attn::{decode_attn, Flags, KvCache};
use crate::mel;
use crate::op;
use crate::tokenizer::Tokenizer;
use crate::weights::Weights;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{activation, module, Device, Int, Tensor, TensorData};

/// The model's `config.json` (Hugging Face `WhisperConfig`); only the fields this crate reads.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    /// Width of the model (384 tiny, 512 base).
    pub d_model: usize,
    /// Encoder blocks.
    pub encoder_layers: usize,
    /// Decoder blocks.
    pub decoder_layers: usize,
    /// Encoder attention heads.
    pub encoder_attention_heads: usize,
    /// Decoder attention heads.
    pub decoder_attention_heads: usize,
    /// Mel bins of the input (80 for tiny and base).
    pub num_mel_bins: usize,
    /// Vocabulary size, special tokens included.
    pub vocab_size: usize,
    /// Most decoder positions (448).
    pub max_target_positions: usize,
    /// Most encoder positions after the stride-2 conv (1500 = 30 s).
    pub max_source_positions: usize,
    /// Tokens never sampled (Whisper's non-speech list).
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
}

/// Decoding options. `Opts::default()` is tuned for short spoken commands.
#[derive(Debug, Clone, Copy)]
pub struct Opts {
    /// Feed the encoder the clip's own length instead of the 30 s pad.
    pub trim: bool,
    /// Greedy steps between token readbacks (the ids stay on the device in between).
    pub tokens_per_readback: usize,
    /// Most tokens per utterance (game commands are short; Whisper's own cap is 224).
    pub max_tokens: usize,
    /// Silence (ms) appended to a trimmed clip. Whisper learned end-of-text against the 30 s pad's silence; with none
    /// it tends not to stop (the right text, then repeats up to the cap). [`PAD_AUTO`] (the default) picks it per
    /// model at load: 1000 ms for tiny, 1500 ms for base and up. Measured on 100 LibriSpeech clips: tiny 5.5 / 5.1 /
    /// 5.4 / 4.9 % WER at 500 / 1000 / 1500 / 2000 ms (flat within noise); base 12.3 / 5.3 / 3.9 / 4.5 %.
    pub pad_ms: usize,
    /// Debug: log the f32 matmuls of the first readback window of decode to stderr. Only with the
    /// `partii-kernels` feature; otherwise ignored.
    pub mm_log: bool,
    /// Token cap from the clip length: tokens <= per_sec * seconds + 10 (speech is ~3-4 tokens/s).
    pub tokens_per_sec: f32,
}

/// `Opts::pad_ms` sentinel: choose the trailing pad from the model (see `Opts::pad_ms`).
pub const PAD_AUTO: usize = usize::MAX;

impl Default for Opts {
    fn default() -> Self {
        Opts { trim: true, tokens_per_readback: 8, max_tokens: 96, pad_ms: PAD_AUTO, mm_log: false, tokens_per_sec: 6.0 }
    }
}

/// One utterance's result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Transcript {
    /// The text, trimmed.
    pub text: String,
    /// The decoded token ids (no special tokens, no end-of-text).
    pub tokens: Vec<u32>,
    /// Tensor ops issued by this crate for the call (before burn's fusion).
    pub ops: u64,
    /// Device-to-host readbacks of token ids.
    pub readbacks: u32,
    /// Encoder positions used (50 per second of audio plus pad, with `Opts::trim`; 1500 without).
    pub encoder_positions: usize,
    /// Time in the CPU mel front end, ms (0 on wasm32-unknown-unknown, where the caller times calls).
    pub mel_ms: f64,
}

struct Lin {
    w: Tensor<2>, // [in, out]
    b: Option<Tensor<1>>,
}

impl Lin {
    fn load(w: &Weights, p: &str, bias: bool) -> Result<Lin, String> {
        Ok(Lin { w: w.t::<2>(&format!("{p}.weight"))?.transpose(), b: if bias { Some(w.t::<1>(&format!("{p}.bias"))?) } else { None } })
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let y = x.reshape([b * n, self.w.dims()[0]]).matmul(self.w.clone());
        op(1);
        let y = match &self.b {
            Some(bias) => {
                op(1);
                y + bias.clone().unsqueeze_dim::<2>(0)
            }
            None => y,
        };
        y.reshape([b, n, self.w.dims()[1]])
    }
}

struct Ln {
    g: Tensor<1>,
    b: Tensor<1>,
}

impl Ln {
    fn load(w: &Weights, p: &str) -> Result<Ln, String> {
        Ok(Ln { g: w.t::<1>(&format!("{p}.weight"))?, b: w.t::<1>(&format!("{p}.bias"))? })
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        op(1);
        module::layer_norm(x, self.g.clone(), Some(self.b.clone()), 1e-5)
    }
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    heads: usize,
}

impl Attn {
    fn load(w: &Weights, p: &str, heads: usize) -> Result<Attn, String> {
        Ok(Attn { q: Lin::load(w, &format!("{p}.q_proj"), true)?, k: Lin::load(w, &format!("{p}.k_proj"), false)?, v: Lin::load(w, &format!("{p}.v_proj"), true)?, o: Lin::load(w, &format!("{p}.out_proj"), true)?, heads })
    }
    /// [b, n, d] -> [b, heads, n, d/heads]
    fn split(&self, x: Tensor<3>) -> Tensor<4> {
        let [b, n, d] = x.dims();
        x.reshape([b, n, self.heads, d / self.heads]).swap_dims(1, 2)
    }
    fn kv(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>) {
        (self.split(self.k.f(x.clone())), self.split(self.v.f(x)))
    }
    /// [1, n, d] -> [n, heads, d/heads] (the decode kernel's layout)
    fn rows(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, n, d] = x.dims();
        x.reshape([n, self.heads, d / self.heads])
    }
    /// Decode step through a KV cache: self-attention appends this step's K/V; cross-attention reads a fixed KV.
    fn attend_cached(&self, x: Tensor<3>, cache: &mut KvCache, append: bool, causal: bool) -> Result<Tensor<3>, String> {
        let [b, n, d] = x.dims();
        let q = self.rows(self.q.f(x.clone()));
        let kv = append.then(|| (self.rows(self.k.f(x.clone())), self.rows(self.v.f(x))));
        let scale = 1.0 / ((d / self.heads) as f64).sqrt();
        let o = decode_attn(q, cache, kv, &Flags { causal, append, scale })?;
        Ok(self.o.f(o.reshape([b, n, d])))
    }
    fn attend(&self, x: Tensor<3>, k: Tensor<4>, v: Tensor<4>, causal: bool) -> Tensor<3> {
        let [b, n, d] = x.dims();
        let q = self.split(self.q.f(x));
        let scale = 1.0 / ((d / self.heads) as f64).sqrt();
        op(1);
        let o = module::attention(q, k, v, None, None, AttentionModuleOptions { scale: Some(scale), softcap: None, is_causal: causal });
        self.o.f(o.swap_dims(1, 2).reshape([b, n, d]))
    }
}

struct Mlp {
    fc1: Lin,
    fc2: Lin,
}

impl Mlp {
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        op(1);
        self.fc2.f(activation::gelu(self.fc1.f(x)))
    }
}

struct EncBlock {
    ln1: Ln,
    attn: Attn,
    ln2: Ln,
    mlp: Mlp,
}

struct DecBlock {
    ln1: Ln,
    self_attn: Attn,
    ln_x: Ln,
    cross: Attn,
    ln2: Ln,
    mlp: Mlp,
}

/// A loaded Whisper model on one burn [`Device`]. Load once, then call [`Whisper::transcribe`] per utterance.
pub struct Whisper {
    /// The model's config.
    pub cfg: Config,
    dev: Device,
    conv1: (Tensor<3>, Tensor<1>),
    conv2: (Tensor<3>, Tensor<1>),
    enc_pos: Tensor<2>,
    enc: Vec<EncBlock>,
    enc_ln: Ln,
    tok_emb: Tensor<2>,   // [vocab, d]
    tok_emb_t: Tensor<2>, // [d, vocab] for the logits
    dec_pos: Tensor<2>,
    dec: Vec<DecBlock>,
    dec_ln: Ln,
    suppress: Tensor<2>, // [1, vocab]: 0 or -inf
    filters: Vec<f32>,
    tok: Tokenizer,
    sot: u32,
    eot: u32,
    no_ts: u32,
    /// The options in use (with `pad_ms` resolved from [`PAD_AUTO`]).
    pub opts: Opts,
}

fn mlp(w: &Weights, p: &str) -> Result<Mlp, String> {
    Ok(Mlp { fc1: Lin::load(w, &format!("{p}.fc1"), true)?, fc2: Lin::load(w, &format!("{p}.fc2"), true)? })
}

impl Whisper {
    /// Load a model from the bytes of the Hugging Face `openai/whisper-*` files: `config.json`, `model.safetensors`
    /// (f32, f16 or bf16; kept as f32 on the device) and `tokenizer.json`. The weights are copied to `dev`, so the
    /// bytes can be dropped afterwards.
    ///
    /// `dev` is any burn device: `Device::flex()` for the CPU, or a wgpu device (feature `gpu`).
    pub fn load(dev: &Device, config_json: &str, safetensors: &[u8], tokenizer_json: &str, opts: Opts) -> Result<Whisper, String> {
        // partii-kernels: Partii's WGSL f32 matmul (fmm) is no faster than burn's default for this decode on Metal,
        // so it stays off natively; LISTEN_FMM=1 turns it on for A/B tests. In the browser the Worker is the process:
        // web.rs sets the switch from its `fmm` argument before this runs.
        #[cfg(all(feature = "partii-kernels", not(all(target_arch = "wasm32", target_os = "unknown"))))]
        burn_cubecl::kernel::fmm_wgsl::set_enabled(std::env::var("LISTEN_FMM").as_deref() == Ok("1"));
        let cfg: Config = serde_json::from_str(config_json).map_err(|e| e.to_string())?;
        let mut opts = opts;
        if opts.pad_ms == PAD_AUTO {
            opts.pad_ms = if cfg.d_model <= 384 { 1000 } else { 1500 };
        }
        let w = Weights::new(safetensors, dev)?;
        let pre = if w.has("model.encoder.conv1.weight") { "model." } else { "" };
        let e = |s: &str| format!("{pre}encoder.{s}");
        let d = |s: &str| format!("{pre}decoder.{s}");
        let mut enc = Vec::new();
        for i in 0..cfg.encoder_layers {
            let p = e(&format!("layers.{i}"));
            enc.push(EncBlock { ln1: Ln::load(&w, &format!("{p}.self_attn_layer_norm"))?, attn: Attn::load(&w, &format!("{p}.self_attn"), cfg.encoder_attention_heads)?, ln2: Ln::load(&w, &format!("{p}.final_layer_norm"))?, mlp: mlp(&w, &p)? });
        }
        let mut dec = Vec::new();
        for i in 0..cfg.decoder_layers {
            let p = d(&format!("layers.{i}"));
            dec.push(DecBlock {
                ln1: Ln::load(&w, &format!("{p}.self_attn_layer_norm"))?,
                self_attn: Attn::load(&w, &format!("{p}.self_attn"), cfg.decoder_attention_heads)?,
                ln_x: Ln::load(&w, &format!("{p}.encoder_attn_layer_norm"))?,
                cross: Attn::load(&w, &format!("{p}.encoder_attn"), cfg.decoder_attention_heads)?,
                ln2: Ln::load(&w, &format!("{p}.final_layer_norm"))?,
                mlp: mlp(&w, &p)?,
            });
        }
        let tok = Tokenizer::from_json(tokenizer_json)?;
        let sot = tok.special("<|startoftranscript|>").ok_or("no <|startoftranscript|>")?;
        let eot = tok.special("<|endoftext|>").ok_or("no <|endoftext|>")?;
        let no_ts = tok.special("<|notimestamps|>").ok_or("no <|notimestamps|>")?;
        // suppress: the config's list, plus every special token except EOT (no timestamps, no task tokens mid-text)
        let mut sup = vec![0f32; cfg.vocab_size];
        for &i in &cfg.suppress_tokens {
            if (i as usize) < sup.len() {
                sup[i as usize] = f32::NEG_INFINITY;
            }
        }
        for i in tok.first_special as usize..cfg.vocab_size {
            if i as u32 != eot {
                sup[i] = f32::NEG_INFINITY;
            }
        }
        let tok_emb = w.t::<2>(&d("embed_tokens.weight"))?;
        Ok(Whisper {
            conv1: (w.t::<3>(&e("conv1.weight"))?, w.t::<1>(&e("conv1.bias"))?),
            conv2: (w.t::<3>(&e("conv2.weight"))?, w.t::<1>(&e("conv2.bias"))?),
            enc_pos: w.t::<2>(&e("embed_positions.weight"))?,
            enc,
            enc_ln: Ln::load(&w, &e("layer_norm"))?,
            tok_emb_t: tok_emb.clone().transpose(),
            tok_emb,
            dec_pos: w.t::<2>(&d("embed_positions.weight"))?,
            dec,
            dec_ln: Ln::load(&w, &d("layer_norm"))?,
            suppress: Tensor::<2>::from_data(TensorData::new(sup, [1, cfg.vocab_size]), dev),
            filters: mel::mel_filters(cfg.num_mel_bins),
            tok,
            sot,
            eot,
            no_ts,
            dev: dev.clone(),
            cfg,
            opts,
        })
    }

    fn encode(&self, mel: Tensor<3>) -> Tensor<3> {
        let cv = |x, (w, b): &(Tensor<3>, Tensor<1>), stride| {
            op(2);
            activation::gelu(module::conv1d(x, w.clone(), Some(b.clone()), burn::tensor::ops::ConvOptions::new([stride], [1], [1], 1)))
        };
        let x = cv(mel, &self.conv1, 1);
        let x = cv(x, &self.conv2, 2); // [1, d, T/2]
        let x = x.swap_dims(1, 2); // [1, T/2, d]
        let n = x.dims()[1];
        op(1);
        let mut x = x + self.enc_pos.clone().narrow(0, 0, n).unsqueeze_dim::<3>(0);
        for b in &self.enc {
            let h = b.ln1.f(x.clone());
            let (k, v) = b.attn.kv(h.clone());
            op(2);
            x = x + b.attn.attend(h, k, v, false);
            x = x.clone() + b.mlp.f(b.ln2.f(x));
        }
        self.enc_ln.f(x)
    }

    /// Speech to text for one utterance: 16 kHz mono f32 samples in -1..1, up to 30 s.
    pub async fn transcribe(&self, pcm: &[f32]) -> Result<Transcript, String> {
        self.transcribe_with(pcm, None).await
    }

    /// As [`Whisper::transcribe`], stopping early with `Err("cancelled")` once `cancel` is set; it is checked at
    /// every token readback.
    pub async fn transcribe_with(&self, pcm: &[f32], cancel: Option<&crate::stream::Cancel>) -> Result<Transcript, String> {
        let ops0 = crate::ops_total();
        let t0 = now_ms();
        let pad = if self.opts.trim { pcm.len() + self.opts.pad_ms * mel::SAMPLE_RATE / 1000 } else { mel::N_SAMPLES_30S };
        let (m, frames) = mel::log_mel(pcm, self.cfg.num_mel_bins, &self.filters, pad);
        // conv2 halves the frames; keep an even count within the model's positions (2 * max_source_positions)
        let frames_used = (frames - frames % 2).clamp(2, 2 * self.cfg.max_source_positions);
        let mut mm = vec![0f32; self.cfg.num_mel_bins * frames_used];
        for r in 0..self.cfg.num_mel_bins {
            mm[r * frames_used..(r + 1) * frames_used].copy_from_slice(&m[r * frames..r * frames + frames_used]);
        }
        let mel_ms = now_ms() - t0;
        let mel = Tensor::<3>::from_data(TensorData::new(mm, [1, self.cfg.num_mel_bins, frames_used]), &self.dev);
        let enc = self.encode(mel);
        let positions = enc.dims()[1];
        // cross-attention K/V once per utterance, bound read-only in the decode kernel's layout [heads, seq, hd]
        let mut cross: Vec<KvCache> = self.dec.iter().map(|b| {
            let (k, v) = b.cross.kv(enc.clone());
            KvCache::fixed(k.squeeze_dim::<3>(0), v.squeeze_dim::<3>(0))
        }).collect();
        let hd = self.cfg.d_model / self.cfg.decoder_attention_heads;
        let cap = self.opts.max_tokens + 2;
        let mut cache: Vec<KvCache> = self.dec.iter().map(|_| KvCache::new(self.cfg.decoder_attention_heads, cap, hd, &self.dev)).collect();
        let mut ids: Tensor<1, Int> = Tensor::<1, Int>::from_data(TensorData::new(vec![self.sot as i64, self.no_ts as i64], [2]), &self.dev);
        let mut pos = 0usize;
        let mut pending: Vec<Tensor<1, Int>> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        let mut readbacks = 0u32;
        let secs = pcm.len() as f32 / mel::SAMPLE_RATE as f32;
        let max = self.opts.max_tokens.min(self.cfg.max_target_positions - 2).min((self.opts.tokens_per_sec * secs) as usize + 10);
        #[cfg(feature = "partii-kernels")]
        if self.opts.mm_log {
            burn_cubecl::kernel::fmm_wgsl::mm_log_enable(true);
        }
        'steps: for step in 0..max {
            let n = ids.dims()[0];
            op(3);
            let mut x = self.tok_emb.clone().select(0, ids.clone()).unsqueeze_dim::<3>(0) + self.dec_pos.clone().narrow(0, pos, n).unsqueeze_dim::<3>(0);
            for (i, b) in self.dec.iter().enumerate() {
                op(1);
                x = x.clone() + b.self_attn.attend_cached(b.ln1.f(x), &mut cache[i], true, true)?;
                op(1);
                x = x.clone() + b.cross.attend_cached(b.ln_x.f(x), &mut cross[i], false, false)?;
                op(1);
                x = x.clone() + b.mlp.f(b.ln2.f(x));
            }
            pos += n;
            let last = self.dec_ln.f(x.narrow(1, n - 1, 1)).reshape([1, self.cfg.d_model]);
            op(3);
            let logits = last.matmul(self.tok_emb_t.clone()) + self.suppress.clone();
            let next = logits.argmax(1).reshape([1]); // stays on the device
            pending.push(next.clone());
            ids = next;
            if pending.len() == self.opts.tokens_per_readback || step + 1 == max {
                readbacks += 1;
                if cancel.is_some_and(|c| c.is_set()) {
                    return Err("cancelled".into());
                }
                let got = Tensor::cat(std::mem::take(&mut pending), 0).into_data_async().await.map_err(|e| format!("{e:?}"))?;
                #[cfg(feature = "partii-kernels")]
                if self.opts.mm_log && readbacks == 1 {
                    for l in burn_cubecl::kernel::fmm_wgsl::mm_log_take() {
                        eprintln!("mmlog decode steps 0..={step}: {l}");
                    }
                    burn_cubecl::kernel::fmm_wgsl::mm_log_enable(false);
                }
                for t in got.iter::<i64>() {
                    if t as u32 == self.eot {
                        break 'steps;
                    }
                    out.push(t as u32);
                }
                if cut_repeats(&mut out) {
                    break 'steps;
                }
            }
        }
        Ok(Transcript { text: self.tok.decode(&out).trim().to_string(), tokens: out, ops: crate::ops_total() - ops0, readbacks, encoder_positions: positions, mel_ms })
    }
}

/// Greedy decoding loops: if the tail is one span of 2..=16 tokens repeated 3 times or more, or a span of 6..=64
/// tokens repeated back to back (a partial last copy counts), keep one copy and stop.
pub(crate) fn cut_repeats(out: &mut Vec<u32>) -> bool {
    let n = out.len();
    for k in 2..=16usize {
        if n < 3 * k {
            break;
        }
        let tail = &out[n - k..];
        let mut reps = 1;
        while n >= (reps + 1) * k && &out[n - (reps + 1) * k..n - reps * k] == tail {
            reps += 1;
        }
        if reps >= 3 {
            out.truncate(n - (reps - 1) * k);
            return true;
        }
    }
    // sentence-level loops (seen with base.en on LibriSpeech): the tail has a period p of 6..=64 tokens for at least
    // one whole period ("A. A." or "A. A. A" with a partial copy). Keep the first copy.
    for p in 6..=64usize {
        if n < 2 * p {
            break;
        }
        let mut r = 0;
        while r < n - p && out[n - 1 - r] == out[n - 1 - r - p] {
            r += 1;
        }
        if r >= p {
            out.truncate(n - r);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    #[test]
    fn repeats() {
        let mut v = vec![1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5];
        assert!(super::cut_repeats(&mut v));
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
        let mut w = vec![1, 2, 1, 2, 9];
        assert!(!super::cut_repeats(&mut w));
        assert_eq!(w.len(), 5);
        // a whole sentence twice, then a partial third copy
        let s: Vec<u32> = (10..20).collect();
        let mut x = vec![1, 2];
        x.extend(&s); x.extend(&s); x.extend(&s[..4]);
        assert!(super::cut_repeats(&mut x));
        let mut want = vec![1, 2]; want.extend(&s);
        assert_eq!(x, want);
        // a 6+ token phrase that recurs later, not back to back, is kept
        let mut y: Vec<u32> = (10..20).collect();
        y.extend([90, 91, 92]); y.extend(10..16);
        assert!(!super::cut_repeats(&mut y));
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn now_ms() -> f64 {
    0.0 // the JS side times calls (performance.now()); mel_ms is native-only
}
