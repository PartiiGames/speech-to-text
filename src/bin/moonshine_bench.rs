//! Bench and parity probe (feature `bench`): Moonshine on the CPU (flex) over a clip list written by
//! farm_moonshine.py. Writes one JSON line per clip: tokens, text, ms (median of `reps` runs), and, when the clip has
//! a reference encoder dump (`enc`: raw little-endian f32, [positions, hidden]), the max abs difference.
//! Threads: RAYON_NUM_THREADS. MOONSHINE_RAW=1 turns the repeat guard off (raw greedy, for token parity).
//! Usage: moonshine_bench <hf_dir> <clips.json> <out.jsonl> <label>
use burn::tensor::Device;
use speech_to_text::{Moonshine, MoonshineOpts};
use std::io::Write;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Clip {
    id: String,
    wav: String,
    #[serde(default)]
    enc: Option<String>,
    #[serde(default = "one")]
    reps: usize,
}
fn one() -> usize { 1 }

fn read_wav(p: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).expect("wav");
    assert_eq!(r.spec().sample_rate, 16_000, "{p}");
    let ch = r.spec().channels as usize;
    let s: Vec<f32> = r.samples::<i16>().map(|x| x.unwrap() as f32 / 32768.0).collect();
    if ch == 1 { s } else { s.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect() }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, clips, out, label) = (&a[1], &a[2], &a[3], &a[4]);
    let rd = |f: &str| std::fs::read(format!("{dir}/{f}")).expect(f);
    let opts = MoonshineOpts { repeat_guard: std::env::var("MOONSHINE_RAW").is_err(), ..MoonshineOpts::default() };
    let t = Instant::now();
    let m = Moonshine::load(&Device::flex(), &String::from_utf8(rd("config.json")).unwrap(), &rd("model.safetensors"), &String::from_utf8(rd("tokenizer.json")).unwrap(), opts).expect("load");
    println!("{label}: loaded in {:.0} ms (threads {})", t.elapsed().as_secs_f64() * 1e3, std::env::var("RAYON_NUM_THREADS").unwrap_or("default".into()));
    let all: Vec<Clip> = serde_json::from_str(&std::fs::read_to_string(clips).unwrap()).unwrap();
    let mut f = std::fs::File::create(out).unwrap();
    // warm-up
    if let Some(c) = all.first() {
        pollster::block_on(m.transcribe(&read_wav(&c.wav))).expect("warm-up");
    }
    let (mut audio, mut busy) = (0f64, 0f64);
    for (i, c) in all.iter().enumerate() {
        let pcm = read_wav(&c.wav);
        let sec = pcm.len() as f64 / 16_000.0;
        let mut ms = Vec::new();
        let mut tr = None;
        for _ in 0..c.reps.max(1) {
            let t = Instant::now();
            let r = pollster::block_on(m.transcribe(&pcm)).expect("transcribe");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            tr = Some(r);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        let tr = tr.unwrap();
        let (mut dmax, mut rmax) = (None, None);
        if let Some(p) = &c.enc {
            let b = std::fs::read(p).expect("enc");
            let r: Vec<f32> = b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
            let (ours, _) = pollster::block_on(m.encoder_output(&pcm)).expect("enc");
            if ours.len() == r.len() {
                dmax = Some(ours.iter().zip(&r).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
                rmax = Some(r.iter().map(|x| x.abs()).fold(0f32, f32::max));
            } else {
                println!("{}: encoder length {} vs reference {}", c.id, ours.len(), r.len());
            }
        }
        audio += sec;
        busy += med;
        let row = serde_json::json!({"id": c.id, "sec": sec, "ms": med, "ms_all": ms, "tokens": tr.tokens, "text": tr.text,
            "positions": tr.encoder_positions, "cap": m.max_new_tokens(pcm.len()), "enc_max_diff": dmax, "enc_max_ref": rmax});
        writeln!(f, "{row}").unwrap();
        if (i + 1) % 10 == 0 || i + 1 == all.len() {
            println!("{label}: {}/{} clips, rtf {:.3}", i + 1, all.len(), busy / 1e3 / audio);
        }
    }
}
