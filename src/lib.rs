//! On-device speech to text: OpenAI's Whisper (`tiny.en`, `base.en`) on [burn](https://burn.dev) 0.22.1.
//!
//! The same code runs on the CPU (burn's `flex` backend, natively and in wasm) and on the GPU through wgpu (WebGPU in
//! the browser, Metal / Vulkan / D3D12 natively). Everything is async, so it also runs in a single-threaded browser
//! Worker without blocking.
//!
//! The crate never downloads anything: you pass it the bytes of the three Hugging Face files
//! (`config.json`, `model.safetensors`, `tokenizer.json` from `openai/whisper-tiny.en` or `openai/whisper-base.en`).
//! The `transcribe` example shows a download pinned by revision and sha256.
//!
//! # Example
//!
//! ```no_run
//! use burn::tensor::Device;
//! use speech_to_text::{Opts, Whisper};
//!
//! # fn main() -> Result<(), String> {
//! let dir = "whisper-tiny.en";
//! let read = |f: &str| std::fs::read(format!("{dir}/{f}")).map_err(|e| e.to_string());
//! let config = String::from_utf8(read("config.json")?).map_err(|e| e.to_string())?;
//! let tokenizer = String::from_utf8(read("tokenizer.json")?).map_err(|e| e.to_string())?;
//! let weights = read("model.safetensors")?;
//!
//! let device = Device::flex(); // the CPU; a wgpu device works the same way
//! let whisper = Whisper::load(&device, &config, &weights, &tokenizer, Opts::default())?;
//!
//! let pcm: Vec<f32> = vec![0.0; 16_000]; // 16 kHz mono f32, -1..1
//! let t = pollster::block_on(whisper.transcribe(&pcm))?;
//! println!("{}", t.text);
//! # Ok(())
//! # }
//! ```
//!
//! For push-to-talk with partial results while the user is still speaking, see [`Utterance`]; to stop a decode
//! early, [`Cancel`].
//!
//! # How it is fast on short commands
//! - **Trimmed mel** ([`Opts::trim`], default on): the encoder sees the clip's own length plus a short silence pad,
//!   not the 30 s window. A 3 s command is ~200 encoder positions instead of 1500.
//! - **Tokens stay on the device**: greedy argmax feeds the next step's embedding lookup directly; the ids are read
//!   back every [`Opts::tokens_per_readback`] steps (default 8), not every token.
//! - **Fixed-capacity KV caches** for the decoder, allocated once per utterance.
//! - **A repeat cutter** stops greedy decoding loops.
//!
//! # Features
//! - `gpu` (default): the wgpu backend.
//! - `flex-simd` (default), `flex-rayon` (default): SIMD and threads for the CPU backend.
//! - `web`: the wasm-bindgen browser entry, `web::Listen` (build for `wasm32-unknown-unknown`).
//! - `partii-kernels`: switches for Partii's hand-written WGSL kernels. Needs Partii's fork of `burn-cubecl` through
//!   `[patch.crates-io]`; it does not build against stock burn.
//! - `bench`: the `listen_bench` and `mm_probe` binaries.
//!
//! Model weights are not part of this crate. See the NOTICE file for the weights' license (MIT) and the code this
//! crate ports.
#![warn(missing_docs)]

mod decode_attn;
mod mel;
mod model;
mod stream;
mod tokenizer;
mod weights;
#[cfg(feature = "web")]
pub mod web;

pub use model::{Config, Opts, PAD_AUTO, Transcript, Whisper};
pub use stream::{Cancel, PartialEvery, Utterance};

/// Audio sample rate the model takes (16 kHz).
pub const SAMPLE_RATE: usize = mel::SAMPLE_RATE;

use std::sync::atomic::{AtomicU64, Ordering};
static OPS: AtomicU64 = AtomicU64::new(0);
pub(crate) fn op(n: u64) {
    OPS.fetch_add(n, Ordering::Relaxed);
}

/// Tensor ops this crate has issued in this process, all models together (before burn's fusion). A cheap proxy for
/// GPU dispatches; [`Transcript::ops`] is the per-call count.
pub fn ops_total() -> u64 {
    OPS.load(Ordering::Relaxed)
}
