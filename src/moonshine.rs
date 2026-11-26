//! Moonshine (Useful Sensors, English `moonshine-tiny` / `moonshine-base`) encoder-decoder in plain burn ops.
//! Weights: the Hugging Face `UsefulSensors/moonshine-*` tensor names. The architecture follows Hugging Face
//! Transformers' `modeling_moonshine.py` (v4.57); no Transformers code is included. See NOTICE.
//!
//! - Front end: raw 16 kHz audio, no mel. conv1d(1 -> d, k 127, s 64, no bias) + tanh, GroupNorm(1 group),
//!   conv1d(d -> 2d, k 7, s 3) + GELU, conv1d(2d -> d, k 3, s 2) + GELU: ~41.7 encoder positions per second.
//! - Encoder: pre-norm blocks (LayerNorm without bias), self-attention with partial rotary embeddings (the first
//!   `partial_rotary_factor * head_dim` dims, GPT-J style interleaved pairs), GELU MLP, final LayerNorm.
//! - Decoder: token embedding (no scale, no learned positions), pre-norm blocks of causal self-attention (rotary),
//!   cross-attention (no rotary), gated SiLU MLP (`fc1` gives 2 * intermediate: value, gate; `silu(gate) * value`),
//!   final LayerNorm, `proj_out` (tied to the embedding when absent).
//! - Greedy decoding from `decoder_start_token_id` (1, `<s>`) to `eos_token_id` (2, `</s>`).
use crate::decode_attn::{decode_attn, Flags, KvCache};
use crate::model::{cut_repeats, Transcript};
use crate::op;
use crate::sp_tokenizer::SpTokenizer;
use crate::stream::Cancel;
use crate::weights::Weights;
use burn::tensor::ops::{AttentionModuleOptions, ConvOptions};
use burn::tensor::{activation, module, Device, Int, Tensor, TensorData};

/// The model's `config.json` (Hugging Face `MoonshineConfig`); only the fields this crate reads.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MoonshineConfig {
    /// Width of the model (288 tiny, 416 base).
    pub hidden_size: usize,
    /// MLP width (the decoder's `fc1` gives twice this: value and gate).
    pub intermediate_size: usize,
    /// Encoder blocks.
    pub encoder_num_hidden_layers: usize,
    /// Decoder blocks.
    pub decoder_num_hidden_layers: usize,
    /// Encoder attention heads.
    pub encoder_num_attention_heads: usize,
    /// Decoder attention heads.
    pub decoder_num_attention_heads: usize,
    /// Encoder key/value heads (`None`: as many as query heads; only that is supported).
    #[serde(default)]
    pub encoder_num_key_value_heads: Option<usize>,
    /// Decoder key/value heads (`None`: as many as query heads; only that is supported).
    #[serde(default)]
    pub decoder_num_key_value_heads: Option<usize>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Most decoder positions.
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    /// Rotary base.
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    /// Share of each head's dims that get rotary embeddings (0.9).
    #[serde(default = "d_prf")]
    pub partial_rotary_factor: f64,
    /// Encoder MLP activation (`gelu`).
    #[serde(default = "d_gelu")]
    pub encoder_hidden_act: String,
    /// Decoder MLP activation (`silu`, gated).
    #[serde(default = "d_silu")]
    pub decoder_hidden_act: String,
    /// First decoder token (`<s>` = 1).
    #[serde(default = "d_start")]
    pub decoder_start_token_id: u32,
    /// End of text (`</s>` = 2); an int or a list in config.json.
    #[serde(default)]
    pub eos_token_id: serde_json::Value,
}

fn d_max_pos() -> usize { 194 }
fn d_theta() -> f64 { 10_000.0 }
fn d_prf() -> f64 { 0.9 }
fn d_gelu() -> String { "gelu".into() }
fn d_silu() -> String { "silu".into() }
fn d_start() -> u32 { 1 }

/// Decoding options for [`Moonshine`]. The default is tuned for short spoken commands.
#[derive(Debug, Clone, Copy)]
pub struct MoonshineOpts {
    /// Greedy steps between token readbacks (the ids stay on the device in between).
    pub tokens_per_readback: usize,
    /// Most tokens per utterance, whatever the clip length.
    pub max_tokens: usize,
    /// Token cap from the clip length: `max(min_tokens, floor(tokens_per_sec * seconds))`. 6.5 is Useful Sensors'
    /// own limit (speech is ~3-4 tokens/s); it stops hallucination loops early on short clips.
    pub tokens_per_sec: f32,
    /// The length cap never goes below this (a one-word command still gets room).
    pub min_tokens: usize,
    /// Cut greedy decoding loops (a span repeated 3+ times, or a long span twice) and stop.
    pub repeat_guard: bool,
}

impl Default for MoonshineOpts {
    fn default() -> Self {
        MoonshineOpts { tokens_per_readback: 8, max_tokens: 192, tokens_per_sec: 6.5, min_tokens: 12, repeat_guard: true }
    }
}

/// Fewest samples the conv stem takes (shorter clips are padded with silence up to this).
const MIN_SAMPLES: usize = 1600;

struct Lin {
    w: Tensor<2>, // [in, out]
    b: Option<Tensor<1>>,
}

impl Lin {
    /// The bias is used when the checkpoint has one.
    fn load(w: &Weights, p: &str) -> Result<Lin, String> {
        let b = format!("{p}.bias");
        Ok(Lin { w: w.t::<2>(&format!("{p}.weight"))?.transpose(), b: if w.has(&b) { Some(w.t::<1>(&b)?) } else { None } })
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        op(1);
        let y = x.reshape([b * n, self.w.dims()[0]]).matmul(self.w.clone());
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

/// LayerNorm without bias (eps 1e-5).
struct Ln(Tensor<1>);

impl Ln {
    fn load(w: &Weights, p: &str) -> Result<Ln, String> {
        Ok(Ln(w.t::<1>(&format!("{p}.weight"))?))
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        op(1);
        module::layer_norm(x, self.0.clone(), None, 1e-5)
    }
}

/// Rotary tables for positions `start..start + n`: (cos, sin), each `[n, 1, rot_dim / 2]`.
fn rope_table(inv_freq: &[f32], start: usize, n: usize, dev: &Device) -> (Tensor<3>, Tensor<3>) {
    let h = inv_freq.len();
    let mut c = Vec::with_capacity(n * h);
    let mut s = Vec::with_capacity(n * h);
    for p in start..start + n {
        for &f in inv_freq {
            let a = p as f32 * f;
            c.push(a.cos());
            s.push(a.sin());
        }
    }
    (Tensor::<3>::from_data(TensorData::new(c, [n, 1, h]), dev), Tensor::<3>::from_data(TensorData::new(s, [n, 1, h]), dev))
}

/// Interleaved (GPT-J style) rotary on the first `2 * cos.dims()[2]` dims of x (`[n, heads, head_dim]`): each pair
/// (x0, x1) becomes (x0 cos - x1 sin, x1 cos + x0 sin); the other dims pass through.
fn rope(x: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
    let [n, h, hd] = x.dims();
    let half = cos.dims()[2];
    let rd = 2 * half;
    let r = x.clone().narrow(2, 0, rd).reshape([n, h, half, 2]);
    let x0 = r.clone().narrow(3, 0, 1).reshape([n, h, half]);
    let x1 = r.narrow(3, 1, 1).reshape([n, h, half]);
    op(6);
    let o0 = x0.clone() * cos.clone() - x1.clone() * sin.clone();
    let o1 = x1 * cos.clone() + x0 * sin.clone();
    let o = Tensor::cat(vec![o0.reshape([n, h, half, 1]), o1.reshape([n, h, half, 1])], 3).reshape([n, h, rd]);
    if rd < hd { Tensor::cat(vec![o, x.narrow(2, rd, hd - rd)], 2) } else { o }
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
        Ok(Attn { q: Lin::load(w, &format!("{p}.q_proj"))?, k: Lin::load(w, &format!("{p}.k_proj"))?, v: Lin::load(w, &format!("{p}.v_proj"))?, o: Lin::load(w, &format!("{p}.o_proj"))?, heads })
    }
    /// [1, n, d] -> [n, heads, d / heads]
    fn rows(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, n, d] = x.dims();
        x.reshape([n, self.heads, d / self.heads])
    }
    fn scale(&self, d: usize) -> f64 {
        1.0 / ((d / self.heads) as f64).sqrt()
    }
    /// Full bidirectional self-attention with rotary (the encoder). x: [1, n, d].
    fn attend_full(&self, x: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
        let [_, n, d] = x.dims();
        let q = rope(self.rows(self.q.f(x.clone())), cos, sin);
        let k = rope(self.rows(self.k.f(x.clone())), cos, sin);
        let v = self.rows(self.v.f(x));
        let to4 = |t: Tensor<3>| t.swap_dims(0, 1).unsqueeze_dim::<4>(0);
        op(1);
        let o = module::attention(to4(q), to4(k), to4(v), None, None, AttentionModuleOptions { scale: Some(self.scale(d)), softcap: None, is_causal: false });
        self.o.f(o.squeeze_dim::<3>(0).swap_dims(0, 1).reshape([1, n, d]))
    }
    /// K/V of the encoder output for cross-attention, in the decode layout `[heads, seq, head_dim]`.
    fn cross_kv(&self, enc: Tensor<3>) -> KvCache {
        let k = self.rows(self.k.f(enc.clone())).swap_dims(0, 1);
        let v = self.rows(self.v.f(enc)).swap_dims(0, 1);
        KvCache::fixed(k, v)
    }
    /// A decode step: causal self-attention with rotary that appends to `cache` (`rot` given), or cross-attention
    /// over a fixed KV (`rot` None).
    fn attend_cached(&self, x: Tensor<3>, cache: &mut KvCache, rot: Option<(&Tensor<3>, &Tensor<3>)>) -> Result<Tensor<3>, String> {
        let [_, n, d] = x.dims();
        let mut q = self.rows(self.q.f(x.clone()));
        let kv = match rot {
            Some((c, s)) => {
                q = rope(q, c, s);
                Some((rope(self.rows(self.k.f(x.clone())), c, s), self.rows(self.v.f(x))))
            }
            None => None,
        };
        let append = kv.is_some();
        let o = decode_attn(q, cache, kv, &Flags { causal: append, append, scale: self.scale(d) })?;
        Ok(self.o.f(o.reshape([1, n, d])))
    }
}

struct Mlp {
    fc1: Lin,
    fc2: Lin,
    gated: bool,
    silu: bool,
}

impl Mlp {
    fn load(w: &Weights, p: &str, act: &str, gated: bool) -> Result<Mlp, String> {
        let silu = match act {
            "silu" | "swish" => true,
            "gelu" => false,
            other => return Err(format!("{p}: activation {other} not supported")),
        };
        Ok(Mlp { fc1: Lin::load(w, &format!("{p}.fc1"))?, fc2: Lin::load(w, &format!("{p}.fc2"))?, gated, silu })
    }
    fn act(&self, x: Tensor<3>) -> Tensor<3> {
        op(1);
        if self.silu { activation::silu(x) } else { activation::gelu(x) }
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        let h = self.fc1.f(x);
        let h = if self.gated {
            let i = h.dims()[2] / 2;
            op(1);
            self.act(h.clone().narrow(2, i, i)) * h.narrow(2, 0, i)
        } else {
            self.act(h)
        };
        self.fc2.f(h)
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

/// A loaded Moonshine model (`UsefulSensors/moonshine-tiny` or `moonshine-base`, English) on one burn [`Device`].
/// Load once, then call [`Moonshine::transcribe`] per utterance.
///
/// Moonshine reads raw 16 kHz audio of any length (no mel, no 30 s window), so the encoder cost follows the clip.
pub struct Moonshine {
    /// The model's config.
    pub cfg: MoonshineConfig,
    /// The options in use.
    pub opts: MoonshineOpts,
    dev: Device,
    conv1: Tensor<3>,
    conv2: (Tensor<3>, Tensor<1>),
    conv3: (Tensor<3>, Tensor<1>),
    gn: (Tensor<1>, Tensor<1>),
    enc: Vec<EncBlock>,
    enc_ln: Ln,
    tok_emb: Tensor<2>,
    proj_t: Tensor<2>, // [d, vocab]
    dec: Vec<DecBlock>,
    dec_ln: Ln,
    enc_inv_freq: Vec<f32>,
    dec_cos: Tensor<3>,
    dec_sin: Tensor<3>,
    tok: SpTokenizer,
    start: u32,
    eos: u32,
}

/// Rotary inverse frequencies for `rot_dim = floor(head_dim * factor)` (Hugging Face's default rope init).
fn inv_freq(head_dim: usize, factor: f64, theta: f64) -> Vec<f32> {
    let rd = (head_dim as f64 * factor) as usize;
    (0..rd / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / rd as f64)) as f32).collect()
}

impl Moonshine {
    /// Load a model from the bytes of the Hugging Face `UsefulSensors/moonshine-tiny` or `moonshine-base` files:
    /// `config.json`, `model.safetensors` (f32, f16 or bf16; kept as f32 on the device) and `tokenizer.json`. The
    /// weights are copied to `dev`, so the bytes can be dropped afterwards.
    ///
    /// `dev` is any burn device: `Device::flex()` for the CPU, or a wgpu device (feature `gpu`).
    pub fn load(dev: &Device, config_json: &str, safetensors: &[u8], tokenizer_json: &str, opts: MoonshineOpts) -> Result<Moonshine, String> {
        let cfg: MoonshineConfig = serde_json::from_str(config_json).map_err(|e| e.to_string())?;
        for (q, kv) in [(cfg.encoder_num_attention_heads, cfg.encoder_num_key_value_heads), (cfg.decoder_num_attention_heads, cfg.decoder_num_key_value_heads)] {
            if kv.is_some_and(|kv| kv != q) {
                return Err(format!("grouped-query attention ({q} query heads, {kv:?} kv heads) is not supported"));
            }
        }
        let w = Weights::new(safetensors, dev)?;
        let pre = if w.has("model.encoder.conv1.weight") { "model." } else { "" };
        let e = |s: &str| format!("{pre}encoder.{s}");
        let d = |s: &str| format!("{pre}decoder.{s}");
        let mut enc = Vec::new();
        for i in 0..cfg.encoder_num_hidden_layers {
            let p = e(&format!("layers.{i}"));
            enc.push(EncBlock {
                ln1: Ln::load(&w, &format!("{p}.input_layernorm"))?,
                attn: Attn::load(&w, &format!("{p}.self_attn"), cfg.encoder_num_attention_heads)?,
                ln2: Ln::load(&w, &format!("{p}.post_attention_layernorm"))?,
                mlp: Mlp::load(&w, &format!("{p}.mlp"), &cfg.encoder_hidden_act, false)?,
            });
        }
        let mut dec = Vec::new();
        for i in 0..cfg.decoder_num_hidden_layers {
            let p = d(&format!("layers.{i}"));
            dec.push(DecBlock {
                ln1: Ln::load(&w, &format!("{p}.input_layernorm"))?,
                self_attn: Attn::load(&w, &format!("{p}.self_attn"), cfg.decoder_num_attention_heads)?,
                ln_x: Ln::load(&w, &format!("{p}.post_attention_layernorm"))?,
                cross: Attn::load(&w, &format!("{p}.encoder_attn"), cfg.decoder_num_attention_heads)?,
                ln2: Ln::load(&w, &format!("{p}.final_layernorm"))?,
                mlp: Mlp::load(&w, &format!("{p}.mlp"), &cfg.decoder_hidden_act, true)?,
            });
        }
        let tok = SpTokenizer::from_json(tokenizer_json)?;
        let eos = match &cfg.eos_token_id {
            serde_json::Value::Number(n) => n.as_u64().map(|v| v as u32),
            serde_json::Value::Array(a) => a.first().and_then(|v| v.as_u64()).map(|v| v as u32),
            _ => None,
        }
        .or_else(|| tok.special("</s>"))
        .ok_or("no eos_token_id in config.json and no </s>")?;
        let tok_emb = w.t::<2>(&d("embed_tokens.weight"))?;
        let proj_t = if w.has("proj_out.weight") { w.t::<2>("proj_out.weight")?.transpose() } else { tok_emb.clone().transpose() };
        let dec_hd = cfg.hidden_size / cfg.decoder_num_attention_heads;
        let enc_hd = cfg.hidden_size / cfg.encoder_num_attention_heads;
        let dec_if = inv_freq(dec_hd, cfg.partial_rotary_factor, cfg.rope_theta);
        let (dec_cos, dec_sin) = rope_table(&dec_if, 0, cfg.max_position_embeddings, dev);
        Ok(Moonshine {
            conv1: w.t::<3>(&e("conv1.weight"))?,
            conv2: (w.t::<3>(&e("conv2.weight"))?, w.t::<1>(&e("conv2.bias"))?),
            conv3: (w.t::<3>(&e("conv3.weight"))?, w.t::<1>(&e("conv3.bias"))?),
            gn: (w.t::<1>(&e("groupnorm.weight"))?, w.t::<1>(&e("groupnorm.bias"))?),
            enc,
            enc_ln: Ln::load(&w, &e("layer_norm"))?,
            tok_emb,
            proj_t,
            dec,
            dec_ln: Ln::load(&w, &d("norm"))?,
            enc_inv_freq: inv_freq(enc_hd, cfg.partial_rotary_factor, cfg.rope_theta),
            dec_cos,
            dec_sin,
            start: cfg.decoder_start_token_id,
            eos,
            tok,
            dev: dev.clone(),
            cfg,
            opts,
        })
    }

    /// The greedy token cap for a clip of `samples` 16 kHz samples (see [`MoonshineOpts::tokens_per_sec`]).
    pub fn max_new_tokens(&self, samples: usize) -> usize {
        let by_len = (self.opts.tokens_per_sec as f64 * samples as f64 / crate::SAMPLE_RATE as f64) as usize;
        by_len.max(self.opts.min_tokens).min(self.opts.max_tokens).min(self.cfg.max_position_embeddings - 1)
    }

    /// The encoder: [1, positions, d].
    fn encode(&self, pcm: &[f32]) -> Tensor<3> {
        let mut a = pcm.to_vec();
        if a.len() < MIN_SAMPLES {
            a.resize(MIN_SAMPLES, 0.0);
        }
        let n = a.len();
        let x = Tensor::<3>::from_data(TensorData::new(a, [1, 1, n]), &self.dev);
        op(2);
        let x = module::conv1d(x, self.conv1.clone(), None, ConvOptions::new([64], [0], [1], 1)).tanh();
        // GroupNorm with one group: normalize over (channels, time), then a per-channel affine
        let [b, c, t] = x.dims();
        let flat = x.reshape([b, c * t]);
        op(8);
        let xc = flat.clone() - flat.mean_dim(1);
        let var = (xc.clone() * xc.clone()).mean_dim(1);
        let x = (xc / var.add_scalar(1e-5).sqrt()).reshape([b, c, t]);
        let x = x * self.gn.0.clone().reshape([1, c, 1]) + self.gn.1.clone().reshape([1, c, 1]);
        let cv = |x, (w, bias): &(Tensor<3>, Tensor<1>), stride| {
            op(2);
            activation::gelu(module::conv1d(x, w.clone(), Some(bias.clone()), ConvOptions::new([stride], [0], [1], 1)))
        };
        let x = cv(x, &self.conv2, 3);
        let x = cv(x, &self.conv3, 2);
        let mut x = x.swap_dims(1, 2); // [1, positions, d]
        let (cos, sin) = rope_table(&self.enc_inv_freq, 0, x.dims()[1], &self.dev);
        for blk in &self.enc {
            op(2);
            x = x.clone() + blk.attn.attend_full(blk.ln1.f(x), &cos, &sin);
            x = x.clone() + blk.mlp.f(blk.ln2.f(x));
        }
        self.enc_ln.f(x)
    }

    /// The encoder's output for a clip, `(values, positions)`: row-major `[positions, hidden_size]`. For parity
    /// checks against a reference implementation.
    pub async fn encoder_output(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize), String> {
        let enc = self.encode(pcm);
        let n = enc.dims()[1];
        let v = enc.into_data_async().await.map_err(|e| format!("{e:?}"))?.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
        Ok((v, n))
    }

    /// Speech to text for one utterance: 16 kHz mono f32 samples in -1..1, any length (tens of seconds at most;
    /// the model was trained on clips up to ~30 s).
    pub async fn transcribe(&self, pcm: &[f32]) -> Result<Transcript, String> {
        self.transcribe_with(pcm, None).await
    }

    /// As [`Moonshine::transcribe`], stopping early with `Err("cancelled")` once `cancel` is set; it is checked at
    /// every token readback.
    pub async fn transcribe_with(&self, pcm: &[f32], cancel: Option<&Cancel>) -> Result<Transcript, String> {
        let ops0 = crate::ops_total();
        let enc = self.encode(pcm);
        let positions = enc.dims()[1];
        let mut cross: Vec<KvCache> = self.dec.iter().map(|b| b.cross.cross_kv(enc.clone())).collect();
        let heads = self.cfg.decoder_num_attention_heads;
        let hd = self.cfg.hidden_size / heads;
        let max = self.max_new_tokens(pcm.len());
        let mut cache: Vec<KvCache> = self.dec.iter().map(|_| KvCache::new(heads, max + 1, hd, &self.dev)).collect();
        let mut ids: Tensor<1, Int> = Tensor::<1, Int>::from_data(TensorData::new(vec![self.start as i64], [1]), &self.dev);
        let mut pending: Vec<Tensor<1, Int>> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        let mut readbacks = 0u32;
        'steps: for pos in 0..max {
            let cos = self.dec_cos.clone().narrow(0, pos, 1);
            let sin = self.dec_sin.clone().narrow(0, pos, 1);
            op(1);
            let mut x = self.tok_emb.clone().select(0, ids.clone()).unsqueeze_dim::<3>(0); // [1, 1, d]
            for (i, b) in self.dec.iter().enumerate() {
                op(3);
                x = x.clone() + b.self_attn.attend_cached(b.ln1.f(x), &mut cache[i], Some((&cos, &sin)))?;
                x = x.clone() + b.cross.attend_cached(b.ln_x.f(x), &mut cross[i], None)?;
                x = x.clone() + b.mlp.f(b.ln2.f(x));
            }
            let last = self.dec_ln.f(x).reshape([1, self.cfg.hidden_size]);
            op(2);
            let next = last.matmul(self.proj_t.clone()).argmax(1).reshape([1]); // stays on the device
            pending.push(next.clone());
            ids = next;
            if pending.len() == self.opts.tokens_per_readback.max(1) || pos + 1 == max {
                readbacks += 1;
                if cancel.is_some_and(|c| c.is_set()) {
                    return Err("cancelled".into());
                }
                let got = Tensor::cat(std::mem::take(&mut pending), 0).into_data_async().await.map_err(|e| format!("{e:?}"))?;
                for t in got.iter::<i64>() {
                    if t as u32 == self.eos {
                        break 'steps;
                    }
                    out.push(t as u32);
                }
                if self.opts.repeat_guard && cut_repeats(&mut out) {
                    break 'steps;
                }
            }
        }
        Ok(Transcript { text: self.tok.decode(&out).trim().to_string(), tokens: out, ops: crate::ops_total() - ops0, readbacks, encoder_positions: positions, mel_ms: 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn max_abs(t: Tensor<3>) -> f32 {
        t.abs().max().into_data().to_vec::<f32>().unwrap()[0]
    }

    #[test]
    fn rope_identity_at_zero_and_norm_kept() {
        let d = Device::flex();
        let f = inv_freq(36, 0.9, 10_000.0);
        assert_eq!(f.len(), 16); // tiny: rot_dim 32 of 36
        assert_eq!(inv_freq(52, 0.9, 10_000.0).len(), 23); // base: rot_dim 46 of 52
        let x = Tensor::<3>::random([3, 2, 36], Distribution::Normal(0.0, 1.0), &d);
        let (c, s) = rope_table(&f, 0, 3, &d);
        let y = rope(x.clone(), &c, &s);
        // position 0 is the identity
        assert!(max_abs(y.clone().narrow(0, 0, 1) - x.clone().narrow(0, 0, 1)) < 1e-6);
        // pass-through dims untouched
        assert!(max_abs(y.clone().narrow(2, 32, 4) - x.clone().narrow(2, 32, 4)) < 1e-6);
        // a rotation keeps each pair's norm
        let n2 = |t: Tensor<3>| (t.clone() * t).sum_dim(2);
        assert!(max_abs(n2(y) - n2(x)) < 1e-4);
    }

    #[test]
    fn rope_pair_rotation() {
        let d = Device::flex();
        // one head of 2 dims, all rotary, frequency 1: position 1 rotates (1, 0) by 1 radian
        let x = Tensor::<3>::from_data(TensorData::new(vec![1.0f32, 0.0, 1.0, 0.0], [2, 1, 2]), &d);
        let (c, s) = rope_table(&[1.0], 0, 2, &d);
        let y = rope(x, &c, &s).into_data().to_vec::<f32>().unwrap();
        assert!((y[2] - 1f32.cos()).abs() < 1e-6 && (y[3] - 1f32.sin()).abs() < 1e-6, "{y:?}");
    }
}
