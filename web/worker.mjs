// The listen Worker: hosts only this model (so fmm's process-wide switch applies to it alone). Messages:
//   {cmd: "load", model, backend, fmm, padMs, warm}   -> {ok, loadMs, fmm: [on, smallLaunches, launches]}
//   {cmd: "wav", url}                                 -> {ok, ms, result: {text, tokens, ops, readbacks, positions}}
// Errors come back as {ok: false, error}.
import init, { Listen } from "/pkg/speech_to_text.js";
let listen = null;
const bytes = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
const text = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.text(); };
self.onmessage = async ({ data: m }) => {
  try {
    if (m.cmd === "load") {
      if (m.backend === "webgpu" && !self.navigator.gpu) throw new Error("no navigator.gpu in this Worker");
      await init();
      const t = performance.now();
      const d = `/hf/${m.model}`;
      const [cfg, w, tok] = await Promise.all([text(`${d}/config.json`), bytes(`${d}/model.safetensors`), text(`${d}/tokenizer.json`)]);
      const warm = m.warm ? await bytes(m.warm) : undefined;
      listen = await Listen.load(cfg, w, tok, m.backend, !!m.fmm, m.padMs ?? undefined, warm);
      self.postMessage({ ok: true, loadMs: performance.now() - t, fmm: Array.from(Listen.fmm(), Number) });
    } else if (m.cmd === "wav") {
      const wav = await bytes(m.url);
      const t = performance.now();
      const r = JSON.parse(await listen.transcribeWav(wav));
      self.postMessage({ ok: true, ms: performance.now() - t, result: r, fmm: Array.from(Listen.fmm(), Number) });
    }
  } catch (e) { self.postMessage({ ok: false, error: String(e && e.stack || e) }); }
};
