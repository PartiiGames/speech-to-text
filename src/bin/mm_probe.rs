//! flex matmul probe: the M>1 shapes of the three callers (Whisper encoder, chat prefill, pose pointwise) plus two
//! M=1 controls. p50 of 9 runs after a warm-up. Built with and without the `flex-simd` feature to see whether gemm's
//! wasm simd128 kernel is in use. Usage: mm_probe <label>
use burn::tensor::{Device, Distribution, Tensor};
use std::time::Instant;

fn main() {
    let label = std::env::args().nth(1).unwrap_or_default();
    let dev = Device::flex();
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("listen enc tiny qkvo P200", 200, 384, 384),
        ("listen enc tiny mlp P200", 200, 384, 1536),
        ("listen enc base mlp P200", 200, 512, 2048),
        ("chat prefill 64", 64, 2048, 2048),
        ("pose pw 144->40 @64x64", 4096, 144, 40),
        ("pose pw 96->576 @16x16", 256, 96, 576),
        ("M=1 tiny dec mlp", 1, 384, 1536),
        ("M=1 vocab tiny", 1, 384, 51864),
    ];
    // gemm picks its kernel once, at the first matmul, and caches it (gemm-common GEMM_PTR), so the switch must be
    // set before any matmul and only one value can be measured per process. MM_SWITCH=on|off; unset = gemm default.
    match std::env::var("MM_SWITCH").as_deref() {
        Ok("on") => gemm_common::set_wasm_simd128(true),
        Ok("off") => gemm_common::set_wasm_simd128(false),
        _ => {}
    }
    let default_switch = gemm_common::get_wasm_simd128();
    // correctness against a naive f64 reference on sampled entries (for this process's kernel)
    for &(name, m, k, n) in shapes {
        let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);
        let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 1.0), &dev);
        let x = a.clone().matmul(b.clone()).into_data().to_vec::<f32>().unwrap();
        let av = a.into_data().to_vec::<f32>().unwrap();
        let bv = b.into_data().to_vec::<f32>().unwrap();
        let mut worst = 0f64;
        for i in (0..m).step_by((m / 4).max(1)) {
            for j in (0..n).step_by((n / 64).max(1)) {
                let r: f64 = (0..k).map(|t| av[i * k + t] as f64 * bv[t * n + j] as f64).sum();
                worst = worst.max((x[i * n + j] as f64 - r).abs() / (k as f64).sqrt());
            }
        }
        println!("{label} | correctness | {name} | worst |err| / sqrt(K) vs f64 {worst:.2e} | {}", if worst < 1e-4 { "OK" } else { "FAIL" });
    }
    println!("probe {label} flex_simd_feature={} target_feature_simd128={} gemm_wasm_simd128_switch={default_switch}",
        cfg!(feature = "flex-simd"), cfg!(target_feature = "simd128"));
    // on wasm: the same binary with gemm's switch forced each way, so the kernel that ran is known
    // timing: p50 of 9 per shape; the bench script alternates processes (on, off, on, off, ...) for rounds
    let kern = if !cfg!(target_arch = "wasm32") { "native (runtime-detected)" } else if default_switch { "gemm simd128 microkernel" } else { "gemm portable kernel" };
    for &(name, m, k, n) in shapes {
        let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);
        let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 1.0), &dev);
        let _ = a.clone().matmul(b.clone()).into_data();
        let mut t = Vec::new();
        for _ in 0..9 {
            let s = Instant::now();
            let _ = a.clone().matmul(b.clone()).into_data();
            t.push(s.elapsed().as_secs_f64() * 1e3);
        }
        t.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let gf = 2.0 * (m * k * n) as f64 / (t[4] / 1e3) / 1e9;
        println!("{label} | {kern} | {name} | {m}x{k}x{n} | p50 {:.3} ms | {gf:.2} GFLOP/s", t[4]);
    }
}
