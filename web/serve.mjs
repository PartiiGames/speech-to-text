// Static server for the listen bench, 127.0.0.1 only, cross-origin isolated (COOP + COEP require-corp, as the edge
// serves a "threads": true game). Mounts: /pkg/ the wasm-bindgen pkg, /hf/ the weights dirs, /c/ the clip wavs, / web/.
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
const TYPES = { ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
  ".json": "application/json", ".wav": "audio/wav" };
export function serve({ pkg, hf, clips }, port = 0) {
  const roots = { "/pkg/": path.resolve(pkg), "/hf/": path.resolve(hf), "/c/": path.resolve(clips), "/": path.dirname(new URL(import.meta.url).pathname) };
  return new Promise((res) => {
    const s = http.createServer((req, rsp) => {
      const url = decodeURIComponent(req.url.split("?")[0]);
      const pre = Object.keys(roots).find((p) => url.startsWith(p));
      const root = roots[pre], file = path.join(root, url.slice(pre.length));
      // weights may be symlinks into a model store: resolve, but only serve names under the mount
      if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { rsp.writeHead(404); return rsp.end(); }
      rsp.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream", "Content-Length": fs.statSync(file).size,
        "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin", "Cache-Control": "no-store" });
      fs.createReadStream(file).pipe(rsp);
    });
    s.listen(port, "127.0.0.1", () => res(s));
  });
}
