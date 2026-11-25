//! Transcribe a 16 kHz WAV file on the CPU with Whisper tiny.en.
//!
//! ```text
//! cargo run --release --example transcribe -- clip.wav
//! ```
//!
//! The first run downloads the three model files (~152 MB), checks each one's sha256 and keeps them in a cache
//! directory: `$SPEECH_TO_TEXT_CACHE`, else `~/.cache/speech-to-text`. Later runs read the cache (and check the
//! sha256 again). Sources, in order: Partii's model mirror (`https://api.partii.games/v1/ai/speech-to-text/<sha256>/
//! <file>`, flat per sha256), then Hugging Face at the pinned revision. `SPEECH_TO_TEXT_SOURCE=hf` skips the mirror. Convert other audio first, e.g.
//! `ffmpeg -i in.mp3 -ar 16000 -ac 1 clip.wav`.
use burn::tensor::Device;
use sha2::{Digest, Sha256};
use speech_to_text::{Opts, SAMPLE_RATE, Whisper};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// openai/whisper-tiny.en, the upstream source, pinned to one Hugging Face commit.
const REPO: &str = "openai/whisper-tiny.en";
const REVISION: &str = "87c7102498dcde7456f24cfd30239ca606ed9063";
/// Partii's mirror of the same files, addressed by sha256.
const MIRROR: &str = "https://api.partii.games/v1/ai/speech-to-text";
/// (file, sha256, bytes)
const FILES: [(&str, &str, usize); 3] = [
    ("config.json", "b5366317df280f79e6cc366f259f379674f81a8721d458acf2e00680ccb10b1d", 1937),
    ("model.safetensors", "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad", 151_060_136),
    ("tokenizer.json", "5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0", 2_405_679),
];

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// One download, checked against its length, then its sha256.
fn download(url: &str, sha256: &str, len: usize) -> Res<Vec<u8>> {
    eprintln!("downloading {url}");
    let mut b = Vec::with_capacity(len);
    ureq::get(url).call()?.into_reader().take(len as u64 + 1).read_to_end(&mut b)?;
    if b.len() != len {
        return Err(format!("{} bytes, expected {len}", b.len()).into());
    }
    let got = sha256_hex(&b);
    if got != sha256 {
        return Err(format!("sha256 {got}, expected {sha256}").into());
    }
    Ok(b)
}

/// The file's bytes from the cache, downloading it first if it is missing (mirror, then Hugging Face).
fn fetch(dir: &Path, name: &str, sha256: &str, len: usize) -> Res<Vec<u8>> {
    let path = dir.join(name);
    if let Ok(b) = std::fs::read(&path) {
        if b.len() == len && sha256_hex(&b) == sha256 {
            return Ok(b);
        }
        eprintln!("{name}: cached copy has the wrong sha256, downloading again");
    }
    let mut urls = vec![format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{name}")];
    if std::env::var("SPEECH_TO_TEXT_SOURCE").as_deref() != Ok("hf") {
        urls.insert(0, format!("{MIRROR}/{sha256}/{name}"));
    }
    for url in urls {
        match download(&url, sha256, len) {
            Ok(b) => {
                let part = dir.join(format!("{name}.part"));
                std::fs::write(&part, &b)?;
                std::fs::rename(&part, &path)?;
                return Ok(b);
            }
            Err(e) => eprintln!("{url}: {e}"),
        }
    }
    Err(format!("{name}: no source gave a file with sha256 {sha256}").into())
}

fn cache_dir() -> PathBuf {
    let base = std::env::var_os("SPEECH_TO_TEXT_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")).unwrap_or_else(|| PathBuf::from(".cache")).join("speech-to-text")
    });
    base.join(format!("{}@{}", REPO.replace('/', "--"), &REVISION[..12]))
}

/// A 16 kHz WAV (16-bit int or 32-bit float, any channel count) as mono f32.
fn read_wav(path: &str) -> Res<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    if spec.sample_rate as usize != SAMPLE_RATE {
        return Err(format!("{path}: {} Hz; the model takes 16000 Hz (ffmpeg -i in -ar 16000 -ac 1 out.wav)", spec.sample_rate).into());
    }
    let s: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => r.samples::<i16>().map(|x| x.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => r.samples::<f32>().collect::<Result<_, _>>()?,
        (f, b) => return Err(format!("{path}: {b}-bit {f:?} samples; use 16-bit PCM or 32-bit float").into()),
    };
    let ch = spec.channels as usize;
    Ok(if ch == 1 { s } else { s.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect() })
}

fn main() -> Res<()> {
    let wav = std::env::args().nth(1).ok_or("usage: transcribe <clip.wav>")?;
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let [config, weights, tokenizer] = FILES.map(|(name, sha, len)| fetch(&dir, name, sha, len));
    let (config, weights, tokenizer) = (String::from_utf8(config?)?, weights?, String::from_utf8(tokenizer?)?);

    let device = Device::flex(); // the CPU backend
    let t = Instant::now();
    let whisper = Whisper::load(&device, &config, &weights, &tokenizer, Opts::default())?;
    drop(weights);
    eprintln!("loaded {REPO} in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);

    let pcm = read_wav(&wav)?;
    let t = Instant::now();
    let out = pollster::block_on(whisper.transcribe(&pcm))?;
    eprintln!("{:.1} s of audio in {:.0} ms", pcm.len() as f64 / SAMPLE_RATE as f64, t.elapsed().as_secs_f64() * 1e3);
    println!("{}", out.text);
    Ok(())
}
