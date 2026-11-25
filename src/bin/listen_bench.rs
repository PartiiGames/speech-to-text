//! Bench (feature `bench`): speech-to-text Whisper on a backend over a clip set (clips.json: id, set, path, text, sec). Prints a progress line per clip.
//! Usage: listen_bench <hf_dir> <clips.json> <cpu|wgpu> <label> <out.json> [max_clips]
use burn::tensor::Device;
use speech_to_text::{Opts, Whisper};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Clip { id: String, set: String, path: String, text: String, sec: f64 }

fn norm(s: &str) -> Vec<String> {
    s.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c == '\'' || c.is_whitespace() { c } else { ' ' }).collect::<String>()
        .split_whitespace().map(|w| w.to_string()).collect()
}
fn edits(a: &[String], b: &[String]) -> usize {
    let mut d: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut prev = d[0];
        d[0] = i;
        for j in 1..=b.len() {
            let t = d[j];
            d[j] = (d[j] + 1).min(d[j - 1] + 1).min(prev + if a[i - 1] == b[j - 1] { 0 } else { 1 });
            prev = t;
        }
    }
    d[b.len()]
}
fn log(m: &str) {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("[{t}] {m}");
}
fn read_wav(p: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).expect("wav");
    let ch = r.spec().channels as usize;
    let s: Vec<f32> = r.samples::<i16>().map(|x| x.unwrap() as f32 / 32768.0).collect();
    if ch == 1 { s } else { s.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect() }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, clips, be, label, out) = (&a[1], &a[2], a[3].as_str(), &a[4], &a[5]);
    let max: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let dev: Device = match be {
        "cpu" => Device::flex(),
        #[cfg(feature = "gpu")]
        "wgpu" => Device::from(burn::backend::wgpu::WgpuDevice::default()),
        other => panic!("backend {other} not built in"),
    };
    let rd = |f: &str| std::fs::read(format!("{dir}/{f}")).expect(f);
    let t = Instant::now();
    let mut opts = Opts { trim: std::env::var("LISTEN_PAD30").is_err(), ..Opts::default() };
    if let Some(v) = std::env::var("LISTEN_PAD_MS").ok().and_then(|v| v.parse().ok()) { opts.pad_ms = v; }
    opts.mm_log = std::env::var("LISTEN_MM_LOG").is_ok();
    let w = Whisper::load(&dev, &String::from_utf8(rd("config.json")).unwrap(), &rd("model.safetensors"), &String::from_utf8(rd("tokenizer.json")).unwrap(), opts).expect("load");
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    log(&format!("{label}: loaded in {load_ms:.0} ms (trim {} pad_ms {})", w.opts.trim, w.opts.pad_ms));
    let all: Vec<Clip> = serde_json::from_str(&std::fs::read_to_string(clips).unwrap()).unwrap();
    let sel: Vec<&Clip> = all.iter().filter(|c| !c.text.is_empty() && (c.set == "libri" || c.set == "commands")).take(max).collect();
    let mut rows = Vec::new();
    let (mut e, mut n) = (0usize, 0usize);
    // warm-up (pipeline compiles on wgpu)
    if let Some(c) = sel.first() {
        let t = Instant::now();
        let _ = pollster::block_on(w.transcribe(&read_wav(&c.path)));
        log(&format!("{label}: warm-up {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
    }
    for (i, c) in sel.iter().enumerate() {
        let pcm = read_wav(&c.path);
        let t = Instant::now();
        let r = pollster::block_on(w.transcribe(&pcm)).expect("transcribe");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (rf, hy) = (norm(&c.text), norm(&r.text));
        let ce = edits(&rf, &hy);
        e += ce;
        n += rf.len();
        log(&format!("{label} {}/{} {} {:.0} ms rtf {:.3} ops {} rb {} pos {} wer {:.3} | {}", i + 1, sel.len(), c.id, ms, ms / 1e3 / c.sec, r.ops, r.readbacks, r.encoder_positions, ce as f64 / rf.len().max(1) as f64, r.text));
        rows.push(serde_json::json!({"id": c.id, "set": c.set, "sec": c.sec, "ms": ms, "rtf": ms / 1e3 / c.sec, "ops": r.ops, "readbacks": r.readbacks, "positions": r.encoder_positions, "mel_ms": r.mel_ms, "ref": c.text, "hyp": r.text, "edits": ce, "words": rf.len()}));
    }
    let wer = e as f64 / n.max(1) as f64;
    #[cfg(feature = "partii-kernels")]
    let alias = { let (c, d) = burn_cubecl::kernel::alias::stats(); format!(" | alias guard: {c} inputs copied, {d} launches declined") };
    #[cfg(not(feature = "partii-kernels"))]
    let alias = String::new();
    log(&format!("{label}: WER {wer:.4} over {} clips{alias}", rows.len()));
    std::fs::write(out, serde_json::to_string_pretty(&serde_json::json!({"label": label, "backend": be, "trim": opts.trim, "load_ms": load_ms, "wer": wer, "rows": rows})).unwrap()).unwrap();
}
