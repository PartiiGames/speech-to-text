Speech to text (Moonshine and Whisper) on burn for native, WebGPU and the browser, by Partii (https://partii.games), the place to make games together.

# speech-to-text

On-device speech recognition with Useful Sensors' Moonshine (`tiny`, `base`) and OpenAI's Whisper (`tiny.en` and
`base.en`), written in Rust on [burn](https://burn.dev) 0.22.1. It is built for short spoken commands, such as a player
talking to a game, and keeps the audio on the device.

- **One crate, three places:** the native CPU (burn's `flex` backend), the GPU through wgpu (Metal, Vulkan, D3D12
  natively; WebGPU in the browser), and the browser as wasm in a Web Worker (WebGPU or the CPU).
- **Async throughout:** `transcribe(...).await` never blocks, so it runs on a single-threaded wasm Worker.
- **Fast on short clips:** the encoder sees the clip's own length plus a short silence pad, not Whisper's 30 s window;
  token ids stay on the device between readbacks; a repeat cutter stops greedy-decoding loops.
- **Streaming push-to-talk:** `Utterance` takes mic chunks of any size, gives partial transcripts while the user is
  speaking, and the final one at release. `Cancel` stops a decode early.
- **Small:** no `tokenizers` or regex crate, no Python. Weights are plain Hugging Face safetensors (f32, f16 or bf16).

## Quick start

```sh
cargo run --release --example transcribe -- clip.wav
```

The example downloads `openai/whisper-tiny.en` (~152 MB), checks the sha256 of every file, caches them in
`~/.cache/speech-to-text` (or `$SPEECH_TO_TEXT_CACHE`) and prints the transcript of a 16 kHz WAV, on the CPU. It
fetches from Partii's model mirror, `https://api.partii.games/v1/ai/speech-to-text/<sha256>/<file>`, and falls back to
Hugging Face at the pinned revision (`SPEECH_TO_TEXT_SOURCE=hf` goes straight there). Either way the bytes must match
the sha256 in NOTICE. To make a 16 kHz mono WAV from other audio: `ffmpeg -i in.mp3 -ar 16000 -ac 1 clip.wav`.

## Use

```toml
[dependencies]
speech-to-text = "0.2"
burn = { version = "=0.22.1", default-features = false, features = ["std", "flex"] }
```

```rust
use burn::tensor::Device;
use speech_to_text::{Opts, Whisper};

// The bytes of config.json, model.safetensors and tokenizer.json from openai/whisper-tiny.en (or base.en).
let whisper = Whisper::load(&Device::flex(), &config_json, &safetensors, &tokenizer_json, Opts::default())?;
let t = whisper.transcribe(&pcm_16k_mono_f32).await?;
println!("{}", t.text);
```

The crate takes bytes and never downloads anything itself, so you choose where models come from and how they are
verified. For the GPU, pass a wgpu device instead of `Device::flex()` (feature `gpu`, on by default).

Push-to-talk with partials:

```rust
use speech_to_text::{Cancel, PartialEvery, Utterance};

let cancel = Cancel::new();                       // clone it into the UI or a deadline timer
let mut u = Utterance::new(PartialEvery::default()); // a partial per 1 s of new audio
u.push(&chunk);                                   // from the mic, any chunk size
if u.partial_due() { let p = u.partial(&whisper, Some(&cancel)).await?; /* show p.text */ }
let t = u.finish(&whisper, Some(&cancel)).await?; // at release
```

### In the browser

Build with the `web` feature for `wasm32-unknown-unknown`, then run `wasm-bindgen --target web`:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --lib --target wasm32-unknown-unknown \
  --no-default-features --features web
wasm-bindgen --target web --out-dir pkg target/wasm32-unknown-unknown/release/speech_to_text.wasm
```

In a Web Worker:

```js
import init, { Listen } from "./pkg/speech_to_text.js";
await init();
const l = await Listen.load(configJson, weightsBytes, tokenizerJson, "webgpu", false); // or "cpu"
const r = JSON.parse(await l.transcribe(pcm16k)); // {text, tokens, ops, readbacks, positions}
```

`web/worker.mjs` in the repository is a complete Worker.

## Moonshine

[Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) (Useful Sensors, English `moonshine-tiny` and
`moonshine-base`, MIT weights) runs through the same API. It reads raw 16 kHz audio of any length (no mel
spectrogram, no 30 s window), so a short command costs only its own length. In Partii's GPU screen (Hugging Face
Transformers, fp16, full sets, utterances over 30 s dropped) it beat Whisper of the same size, and invented far fewer
words in noise:

| model | WER LibriSpeech test-clean | WER test-other, 0 dB noise |
|---|---|---|
| moonshine base | 2.9 % | 51 % |
| whisper base.en | 3.7 % | 83 % |
| moonshine tiny | 3.6 % | 70 % |
| whisper tiny.en | 4.6 % | 139 % |

```sh
cargo run --release --example moonshine -- clip.wav          # tiny
cargo run --release --example moonshine -- --base clip.wav   # base
```

```rust
use burn::tensor::Device;
use speech_to_text::{Moonshine, MoonshineOpts, Stt};

// config.json, model.safetensors, tokenizer.json from UsefulSensors/moonshine-tiny (or moonshine-base)
let m = Moonshine::load(&Device::flex(), &config_json, &safetensors, &tokenizer_json, MoonshineOpts::default())?;
let t = m.transcribe(&pcm_16k_mono_f32).await?;

// or either model, picked from config.json's model_type
let stt = Stt::load(&Device::flex(), &config_json, &safetensors, &tokenizer_json)?;
let t = stt.transcribe_with(&pcm_16k_mono_f32, Some(&cancel)).await?;
```

Greedy decoding with a KV cache and the same repeat cutter as Whisper; the token cap follows the clip length
(6.5 tokens per second, at least 12). `Utterance` (partials) takes a `Whisper` for now; call `Moonshine::transcribe`
on its buffer the same way.

On the CPU (burn `flex`, f32, an x86-64 Linux desktop), 100 LibriSpeech dev-clean utterances (11.6 min of audio):

| model | WER (ours = HF Transformers greedy) | real-time factor, 1 / 4 threads | 2 s clip | 5 s clip |
|---|---|---|---|---|
| moonshine tiny | 4.60 % | 0.028 / 0.021 | 45 / 38 ms | 123 / 97 ms |
| moonshine base | 2.65 % | 0.057 / 0.041 | 121 / 93 ms | 268 / 192 ms |

Greedy tokens match Hugging Face Transformers 4.57.1 (fp32, `num_beams=1`, same token cap) on all 100 utterances
for both models; the encoder output differs by at most 7e-5 (values up to 7.5).


**Training data.** Useful Sensors trained Moonshine on about 200K hours: about 90K hours of open sets (Common Voice,
AMI, LibriSpeech, Multilingual LibriSpeech English, People's Speech, and GigaSpeech, which is licensed for
non-commercial use only and makes up at most about 5 %) plus audio collected from the web; see
[arXiv 2410.15608](https://arxiv.org/abs/2410.15608). Only the English `moonshine-tiny` and `moonshine-base` are
MIT; Useful Sensors' non-English models are under a non-commercial Community License and are not supported here.

## Features

| Feature | Default | What |
|---|---|---|
| `gpu` | yes | the wgpu backend (WebGPU, Metal, Vulkan, D3D12) |
| `flex-simd` | yes | SIMD for the CPU backend (gemm's simd128 kernel on wasm) |
| `flex-rayon` | yes | threads for the CPU backend (native) |
| `web` | no | the wasm-bindgen browser entry (`Listen`) |
| `bench` | no | the `listen_bench` (clip-set WER and latency) and `mm_probe` binaries |
| `partii-kernels` | no | switches for Partii's hand-written WGSL kernels; needs Partii's fork of `burn-cubecl` through `[patch.crates-io]`, so it does not build against stock burn |

Everything except `partii-kernels` builds against stock burn 0.22.1 from crates.io.

## Numbers

Greedy decoding, f32 weights, `Opts::default()`.

| Where | Model | Result |
|---|---|---|
| Chrome, CPU (wasm in a Worker) | tiny.en | 40 clips: WER 3.36 %, p50 694 ms per clip; clips of 5 s or less: p50 436 ms |
| Apple M5 (Metal, wgpu native) | tiny.en, base.en | median per clip tiny.en 0.14-0.23 s, base.en 0.24 s (40 short command clips) |
| LibriSpeech, 1000 ms silence pad | tiny.en | WER 5.05 % |
| LibriSpeech, 1000 ms silence pad | base.en | WER 5.25 % |

The clip set mixes LibriSpeech utterances and short synthesized game commands. The default pad is 1000 ms for tiny
and 1500 ms for base and up (base did better at 1500 ms in our tests).

## Limits

- English-only models (`*.en`); greedy decoding, no timestamps, no beam search.
- Up to 30 s per utterance (Whisper's window). Partials re-run the whole buffer.
- 16 kHz mono input: resample before calling.
- f32 weights on the device (~150 MB for tiny, ~290 MB for base).

## License

The code is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The model weights are not part of this crate. OpenAI's Whisper weights and Useful Sensors' English Moonshine weights are
MIT-licensed (Moonshine's training data: see [Moonshine](#moonshine)); [NOTICE](NOTICE) lists every file the crate reads
at runtime, with its license, source URL, pinned revision and sha256, and credits the code this crate ports (OpenAI
Whisper's log-mel front end, librosa's mel filter bank, GPT-2's byte-to-unicode table, the Moonshine architecture).

Unless you explicitly state otherwise, any contribution you submit for inclusion in this crate shall be dual licensed
as above, without any additional terms or conditions.
