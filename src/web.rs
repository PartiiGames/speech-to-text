//! Browser entry (feature `web`, `wasm32-unknown-unknown`, then `wasm-bindgen --target web`), meant to run in a
//! dedicated Web Worker:
//! ```js
//! const l = await Listen.load(configJson, weights, tokenizerJson, "webgpu", false, undefined, warmWav);
//! const r = JSON.parse(await l.transcribe(pcm16k));        // {text, tokens, ops, readbacks, positions}
//! const r = JSON.parse(await l.transcribeWav(wavBytes));   // the same, from a 16 kHz 16-bit WAV file
//! ```
//! `fmm` turns on Partii's WGSL f32 matmul (feature `partii-kernels` only). The switch is process-wide, so give the
//! model a Worker of its own when you use it.
use crate::{Opts, Whisper};
use burn::tensor::{Device, IntDType};
use wasm_bindgen::prelude::*;

/// A loaded model in the browser.
#[wasm_bindgen]
pub struct Listen {
    w: Whisper,
}

fn err(e: impl std::fmt::Debug) -> JsError {
    JsError::new(&format!("{e:?}"))
}

/// 16-bit PCM WAV -> mono f32 (the clip set is 16 kHz mono already; channels are averaged otherwise).
fn wav_pcm(bytes: &[u8]) -> Result<Vec<f32>, JsError> {
    let mut r = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(err)?;
    let spec = r.spec();
    if spec.sample_rate != 16000 || spec.bits_per_sample != 16 {
        return Err(JsError::new(&format!("wav must be 16 kHz 16-bit, got {} Hz {} bit", spec.sample_rate, spec.bits_per_sample)));
    }
    let ch = spec.channels as usize;
    let s: Vec<f32> = r.samples::<i16>().map(|x| x.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>().map_err(err)?;
    Ok(if ch == 1 { s } else { s.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect() })
}

#[wasm_bindgen]
impl Listen {
    /// Load from the bytes of `config.json`, `model.safetensors` and `tokenizer.json`.
    /// `backend`: "webgpu" or "cpu" (flex on the Worker thread); `fmm`: Partii's WGSL f32 matmul (webgpu, feature
    /// `partii-kernels`; an error without it); `pad_ms`: trailing silence added before the mel (`Opts::pad_ms`; omit
    /// it for the per-model default); `warm_wav`: an optional clip run once before returning (compiles the pipelines).
    pub async fn load(config_json: String, weights: Vec<u8>, tokenizer_json: String, backend: String, fmm: bool, pad_ms: Option<u32>, warm_wav: Option<Vec<u8>>) -> Result<Listen, JsError> {
        let device = match backend.as_str() {
            "webgpu" => {
                // the switch before the first op
                #[cfg(feature = "partii-kernels")]
                burn_cubecl::kernel::fmm_wgsl::set_enabled(fmm);
                #[cfg(not(feature = "partii-kernels"))]
                if fmm {
                    return Err(JsError::new("fmm needs the partii-kernels feature"));
                }
                use burn_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};
                let mut o = RuntimeOptions::default();
                o.memory_config = MemoryConfiguration::ExclusivePages;
                let wd = WgpuDevice::new(Default::default());
                init_setup_async::<AutoGraphicsApi>(&wd, o).await;
                let mut d = Device::from(wd);
                d.configure(IntDType::I32).map_err(err)?;
                d
            }
            "cpu" => Device::flex(),
            o => return Err(JsError::new(&format!("backend {o}: webgpu or cpu"))),
        };
        let opts = Opts { pad_ms: pad_ms.map_or(crate::PAD_AUTO, |p| p as usize), ..Opts::default() };
        let w = Whisper::load(&device, &config_json, &weights, &tokenizer_json, opts).map_err(|e| JsError::new(&e))?;
        drop(weights);
        let l = Listen { w };
        if let Some(b) = warm_wav {
            l.w.transcribe(&wav_pcm(&b)?).await.map_err(|e| JsError::new(&e))?;
        }
        Ok(l)
    }

    /// One utterance, 16 kHz mono f32 -> JSON {text, tokens, ops, readbacks, positions}.
    pub async fn transcribe(&self, pcm: Vec<f32>) -> Result<String, JsError> {
        let t = self.w.transcribe(&pcm).await.map_err(|e| JsError::new(&e))?;
        Ok(serde_json::json!({"text": t.text, "tokens": t.tokens, "ops": t.ops, "readbacks": t.readbacks, "positions": t.encoder_positions}).to_string())
    }

    /// One utterance from a 16 kHz 16-bit WAV file's bytes.
    #[wasm_bindgen(js_name = transcribeWav)]
    pub async fn transcribe_wav(&self, wav: Vec<u8>) -> Result<String, JsError> {
        self.transcribe(wav_pcm(&wav)?).await
    }

    /// Whether Partii's fmm is on in this Worker, and its small-M and total launches so far (shows which path ran).
    /// `[0, 0, 0]` without the `partii-kernels` feature.
    pub fn fmm() -> Vec<u64> {
        #[cfg(feature = "partii-kernels")]
        return vec![burn_cubecl::kernel::fmm_wgsl::wanted() as u64, burn_cubecl::kernel::fmm_wgsl::small_launches(), burn_cubecl::kernel::fmm_wgsl::launches()];
        #[cfg(not(feature = "partii-kernels"))]
        vec![0, 0, 0]
    }
}
