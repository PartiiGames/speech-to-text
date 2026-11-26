//! One type for either model, picked from `config.json`'s `model_type`.
use crate::{Cancel, Moonshine, MoonshineOpts, Opts, Transcript, Whisper};
use burn::tensor::Device;

/// Either speech-to-text model behind one API. [`Stt::load`] reads `model_type` from `config.json` (`"moonshine"`
/// or `"whisper"`) and loads that model with its default options; build a variant yourself for other options.
///
/// ```no_run
/// # async fn run(config: &str, weights: &[u8], tokenizer: &str, pcm: &[f32]) -> Result<(), String> {
/// use burn::tensor::Device;
/// use speech_to_text::Stt;
/// let stt = Stt::load(&Device::flex(), config, weights, tokenizer)?; // Moonshine or Whisper files
/// println!("{}", stt.transcribe(pcm).await?.text);
/// # Ok(())
/// # }
/// ```
pub enum Stt {
    /// OpenAI Whisper (`tiny.en`, `base.en`).
    Whisper(Whisper),
    /// Useful Sensors Moonshine (`moonshine-tiny`, `moonshine-base`, English).
    Moonshine(Moonshine),
}

impl Stt {
    /// Load the model the files describe, with default options. Errors on an unknown `model_type`.
    pub fn load(dev: &Device, config_json: &str, safetensors: &[u8], tokenizer_json: &str) -> Result<Stt, String> {
        let v: serde_json::Value = serde_json::from_str(config_json).map_err(|e| e.to_string())?;
        match v["model_type"].as_str() {
            Some("moonshine") => Ok(Stt::Moonshine(Moonshine::load(dev, config_json, safetensors, tokenizer_json, MoonshineOpts::default())?)),
            Some("whisper") => Ok(Stt::Whisper(Whisper::load(dev, config_json, safetensors, tokenizer_json, Opts::default())?)),
            other => Err(format!("config.json: model_type {other:?}, expected \"moonshine\" or \"whisper\"")),
        }
    }

    /// `"moonshine"` or `"whisper"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Stt::Whisper(_) => "whisper",
            Stt::Moonshine(_) => "moonshine",
        }
    }

    /// Speech to text for one utterance: 16 kHz mono f32 samples in -1..1.
    pub async fn transcribe(&self, pcm: &[f32]) -> Result<Transcript, String> {
        self.transcribe_with(pcm, None).await
    }

    /// As [`Stt::transcribe`], stopping early with `Err("cancelled")` once `cancel` is set.
    pub async fn transcribe_with(&self, pcm: &[f32], cancel: Option<&Cancel>) -> Result<Transcript, String> {
        match self {
            Stt::Whisper(w) => w.transcribe_with(pcm, cancel).await,
            Stt::Moonshine(m) => m.transcribe_with(pcm, cancel).await,
        }
    }
}

impl From<Whisper> for Stt {
    fn from(w: Whisper) -> Stt {
        Stt::Whisper(w)
    }
}

impl From<Moonshine> for Stt {
    fn from(m: Moonshine) -> Stt {
        Stt::Moonshine(m)
    }
}
