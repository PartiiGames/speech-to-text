# Changelog

## 0.2.0

- **Moonshine** (Useful Sensors `moonshine-tiny` and `moonshine-base`, English, MIT weights): `Moonshine::load` /
  `transcribe` / `transcribe_with`, `MoonshineOpts`, `MoonshineConfig`. Raw 16 kHz audio of any length (no mel, no
  30 s window), a KV cache, greedy decoding with Whisper's repeat cutter, and a token cap from the clip length.
  Greedy tokens match Hugging Face Transformers 4.57.1 on 100 of 100 LibriSpeech dev-clean utterances (both models).
  On 4 desktop CPU threads: a 2 s command in 38 ms (tiny) / 93 ms (base); WER 4.60 % / 2.65 % on those 100.
- `Stt`: Whisper or Moonshine behind one API, chosen from `config.json`'s `model_type`.
- The `moonshine` example (tiny, or `--base`): Partii's mirror first, then Hugging Face at the pinned revision, each
  file checked by length and sha256.
- NOTICE: the Moonshine files (pinned revisions, sha256, sizes), the MIT text, and a note on Moonshine's training data.

## 0.1.0

- First release: OpenAI Whisper `tiny.en` / `base.en` on burn 0.22.1 (CPU `flex`, wgpu, wasm), streaming partials
  (`Utterance`), cancellation (`Cancel`), the `transcribe` example.
