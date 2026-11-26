//! Transcribe a 16 kHz WAV file on the CPU with Moonshine (tiny by default, `--base` for base).
//!
//! ```text
//! cargo run --release --example moonshine -- clip.wav
//! cargo run --release --example moonshine -- --base clip.wav
//! ```
//!
//! The first run downloads the three model files (tiny ~110 MB, base ~248 MB), checks each one's length and
//! sha256 and keeps them in a cache directory: `$SPEECH_TO_TEXT_CACHE`, else `~/.cache/speech-to-text`. Later runs
//! read the cache (and check the sha256 again). Sources, in order: Partii's model mirror
//! (`https://api.partii.games/v1/ai/speech-to-text/<sha256>/<file>`, flat per sha256), then Hugging Face at the
//! pinned revision. `SPEECH_TO_TEXT_SOURCE=hf` skips the mirror. Convert other audio first, e.g.
//! `ffmpeg -i in.mp3 -ar 16000 -ac 1 clip.wav`.
use burn::tensor::Device;
use sha2::{Digest, Sha256};
use speech_to_text::{Moonshine, MoonshineOpts, SAMPLE_RATE};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One model: the upstream Hugging Face repo pinned to one commit, and (file, sha256, bytes) of each file it reads.
struct Model {
    repo: &'static str,
    revision: &'static str,
    files: [(&'static str, &'static str, usize); 3],
}

const TINY: Model = Model {
    repo: "UsefulSensors/moonshine-tiny",
    revision: "390624ed33d594443aa4aa221f5b9f283b545b5a",
    files: [
        ("config.json", "47a43777a14e17b1ffd5f533e021d4d18c3c475cbb96de0947ce409e16444ded", 897),
        ("model.safetensors", "867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736", 108_389_192),
        ("tokenizer.json", "6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb", 1_985_530),
    ],
};

const BASE: Model = Model {
    repo: "UsefulSensors/moonshine-base",
    revision: "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
    files: [
        ("config.json", "f2073049d85ebf3d02825a35e6355004d59310568a1098779e4421b01333a2b1", 976),
        ("model.safetensors", "e020c79d0a979a7ec099f718ff1cd2f19e92aead230d69654bca5975a8e1b862", 246_079_928),
        ("tokenizer.json", "6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb", 1_985_530),
    ],
};

/// Partii's mirror of the same files, addressed by sha256.
const MIRROR: &str = "https://api.partii.games/v1/ai/speech-to-text";

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
fn fetch(m: &Model, dir: &Path, name: &str, sha256: &str, len: usize) -> Res<Vec<u8>> {
    let path = dir.join(name);
    if let Ok(b) = std::fs::read(&path) {
        if b.len() == len && sha256_hex(&b) == sha256 {
            return Ok(b);
        }
        eprintln!("{name}: cached copy has the wrong length or sha256, downloading again");
    }
    let mut urls = vec![format!("https://huggingface.co/{}/resolve/{}/{name}", m.repo, m.revision)];
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

fn cache_dir(m: &Model) -> PathBuf {
    let base = std::env::var_os("SPEECH_TO_TEXT_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")).unwrap_or_else(|| PathBuf::from(".cache")).join("speech-to-text")
    });
    base.join(format!("{}@{}", m.repo.replace('/', "--"), &m.revision[..12]))
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m = if args.iter().any(|a| a == "--base") { &BASE } else { &TINY };
    let wav = args.iter().find(|a| !a.starts_with("--")).ok_or("usage: moonshine [--base] <clip.wav>")?;
    let dir = cache_dir(m);
    std::fs::create_dir_all(&dir)?;
    let [config, weights, tokenizer] = m.files.map(|(name, sha, len)| fetch(m, &dir, name, sha, len));
    let (config, weights, tokenizer) = (String::from_utf8(config?)?, weights?, String::from_utf8(tokenizer?)?);

    let device = Device::flex(); // the CPU backend
    let t = Instant::now();
    let model = Moonshine::load(&device, &config, &weights, &tokenizer, MoonshineOpts::default())?;
    drop(weights);
    eprintln!("loaded {} in {:.0} ms", m.repo, t.elapsed().as_secs_f64() * 1e3);

    let pcm = read_wav(wav)?;
    let t = Instant::now();
    let out = pollster::block_on(model.transcribe(&pcm))?;
    eprintln!("{:.1} s of audio in {:.0} ms", pcm.len() as f64 / SAMPLE_RATE as f64, t.elapsed().as_secs_f64() * 1e3);
    println!("{}", out.text);
    Ok(())
}
