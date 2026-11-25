// Headless driver for the listen Worker bench: one browser, one setting per run (a fresh browser and page).
//   BROWSER=chromium|webkit MODEL=whisper-tiny.en BACKEND=webgpu|cpu FMM=0|1 N=40 PAD_MS=1000 \
//   PKG=<pkg dir> HF=<dir of model dirs> CLIPS=<clips.json> OUT=<out.json> node run.mjs
// Prints a progress line per clip ("[bench] i/N ...") and a summary: WER, per-clip p50/p90, and the fmm counters
// (on, small-M launches, all fmm launches) that prove which kernel ran. Headless only.
import { chromium, webkit } from "playwright-core";
import fs from "node:fs"; import path from "node:path";
import { serve } from "./serve.mjs";
const E = process.env;
const WK = E.BROWSER === "webkit";
// the per-model trailing pad (Opts::pad_ms PAD_AUTO: tiny 1000, base 1500), passed explicitly so older pkgs get it too
const PAD = +(E.PAD_MS || ((E.MODEL || "whisper-tiny.en").includes("tiny") ? 1000 : 1500));
const N = +(E.N || 40), MODEL = E.MODEL || "whisper-tiny.en", BACKEND = E.BACKEND || "webgpu", FMM = E.FMM === "1";
const all = JSON.parse(fs.readFileSync(E.CLIPS, "utf8"));
const clips = all.filter((c) => c.text && (c.set === "libri" || c.set === "commands")).slice(0, N)
  .map((c) => ({ id: c.id, sec: c.sec, text: c.text, file: path.basename(c.path) }));
// PORT, or the job's port block (K2_JOB_PORT_BASE) on a shared CI host; 0 = any free port
const srv = await serve({ pkg: E.PKG, hf: E.HF, clips: path.join(path.dirname(E.CLIPS), "clips") }, +(E.PORT || E.K2_JOB_PORT_BASE || 0));
const browser = WK ? await webkit.launch({ headless: true })
  : await chromium.launch({ headless: true, channel: "chromium", args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", ...(E.CHROME_ARGS || "").split(" ").filter(Boolean)] });
for (const sig of ["SIGTERM", "SIGINT"]) process.on(sig, () => { browser.close().finally(() => process.exit(143)); setTimeout(() => process.exit(143), 3000); });
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (t.startsWith("[bench]") || /panicked at|^Error|assert/i.test(t)) console.log(t); });
page.on("pageerror", (e) => console.error("pageerror", e));
await page.goto(`http://127.0.0.1:${srv.address().port}/bench.html?model=${MODEL}&backend=${BACKEND}&fmm=${FMM ? 1 : 0}&pad=${PAD}`);
await page.waitForFunction(() => window.benchReady, null, { timeout: 60000 });
const res = await page.evaluate((c) => window.runBench(c), clips);
await browser.close(); srv.close();

const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").split(/\s+/).filter(Boolean);
const edits = (a, b) => { let d = [...Array(b.length + 1).keys()];
  for (let i = 1; i <= a.length; i++) { let prev = d[0]; d[0] = i;
    for (let j = 1; j <= b.length; j++) { const t = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = t; } }
  return d[b.length]; };
let e = 0, n = 0;
for (const [i, r] of res.rows.entries()) { const rf = norm(clips[i].text); r.ref = clips[i].text; r.edits = edits(rf, norm(r.text)); r.words = rf.length; e += r.edits; n += rf.length; }
const ms = res.rows.map((r) => r.ms).sort((a, b) => a - b), pct = (p) => ms[Math.min(ms.length - 1, Math.floor(p * ms.length))];
const short = res.rows.filter((r) => r.sec <= 5).map((r) => r.ms).sort((a, b) => a - b);
res.wer = e / Math.max(1, n);
const tag = `${WK ? "webkit" : "chromium"}-${BACKEND}-${FMM ? "fmm" : "default"}-${MODEL}`;
fs.writeFileSync(E.OUT || `out/${tag}.json`, JSON.stringify(res, null, 1));
console.log(`${tag}: WER ${res.wer.toFixed(4)} n=${ms.length} p50 ${pct(0.5)?.toFixed(0)} ms p90 ${pct(0.9)?.toFixed(0)} ms | clips <= 5 s: n=${short.length} p50 ${short[Math.floor((short.length - 1) / 2)]?.toFixed(0)} ms | load ${res.loadMs.toFixed(0)} ms | isolated ${res.isolated} | fmm [on, small, all] ${res.fmmEnd}`);
