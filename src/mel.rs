//! Whisper's log-mel front end on the CPU (input-only glue stays off the GPU).
//! Ported from OpenAI Whisper's `audio.py` (MIT) and librosa's `filters.mel` (ISC); see NOTICE.
//! 16 kHz mono, n_fft 400, hop 160, periodic Hann, centred with reflect padding, power spectrum, Slaney mel filters,
//! log10 clamped to (max - 8), then (x + 4) / 4. Output: [n_mels][frames] row-major.
use rustfft::{num_complex::Complex, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_SAMPLES_30S: usize = 30 * SAMPLE_RATE;

/// librosa.filters.mel(sr=16000, n_fft=400, n_mels), Slaney scale and norm (what Whisper's mel_filters.npz holds).
pub fn mel_filters(n_mels: usize) -> Vec<f32> {
    let sr = SAMPLE_RATE as f64;
    let n_freq = N_FFT / 2 + 1;
    let f_sp = 200.0 / 3.0;
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| if f < 1000.0 { f / f_sp } else { 15.0 + (f / 1000.0).ln() / logstep };
    let mel_to_hz = |m: f64| if m < 15.0 { f_sp * m } else { 1000.0 * (logstep * (m - 15.0)).exp() };
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut w = vec![0f32; n_mels * n_freq];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for j in 0..n_freq {
            let f = j as f64 * sr / N_FFT as f64;
            let lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            w[i * n_freq + j] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

/// Log-mel of `pcm`, zero-padded to `pad_to` samples first (N_SAMPLES_30S for the standard input; 0 = no pad).
/// Returns (mel row-major [n_mels * frames], frames). frames = padded_len / HOP.
pub fn log_mel(pcm: &[f32], n_mels: usize, filters: &[f32], pad_to: usize) -> (Vec<f32>, usize) {
    let n = pcm.len().max(pad_to).max(N_FFT);
    let mut x = pcm.to_vec();
    x.resize(n, 0.0);
    let half = N_FFT / 2;
    // reflect padding (torch.stft center=True)
    let mut p = Vec::with_capacity(n + N_FFT);
    for i in (1..=half).rev() {
        p.push(x[i.min(n - 1)]);
    }
    p.extend_from_slice(&x);
    for i in 0..half {
        p.push(x[n.saturating_sub(2 + i)]);
    }
    let frames = n / HOP; // torch.stft gives n/HOP + 1 frames; Whisper drops the last
    let window: Vec<f32> = (0..N_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()).collect();
    let n_freq = N_FFT / 2 + 1;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0f32, 0f32); N_FFT];
    let mut power = vec![0f32; n_freq];
    let mut mel = vec![0f32; n_mels * frames];
    for t in 0..frames {
        let s = t * HOP;
        for i in 0..N_FFT {
            buf[i] = Complex::new(p[s + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..n_freq {
            power[k] = buf[k].norm_sqr();
        }
        for m in 0..n_mels {
            let row = &filters[m * n_freq..(m + 1) * n_freq];
            let v: f32 = row.iter().zip(&power).map(|(a, b)| a * b).sum();
            mel[m * frames + t] = v.max(1e-10).log10();
        }
    }
    let max = mel.iter().cloned().fold(f32::MIN, f32::max);
    for v in mel.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    (mel, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_shape_and_norm() {
        let f = mel_filters(80);
        assert_eq!(f.len(), 80 * 201);
        assert!(f.iter().all(|v| *v >= 0.0));
        // every filter has some weight, and the Slaney norm keeps each filter's area ~ 2/(bandwidth)
        for m in 0..80 {
            assert!(f[m * 201..(m + 1) * 201].iter().any(|v| *v > 0.0), "filter {m} empty");
        }
    }

    #[test]
    fn mel_frames_and_range() {
        let pcm: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let f = mel_filters(80);
        let (m, frames) = log_mel(&pcm, 80, &f, 0);
        assert_eq!(frames, 100);
        assert_eq!(m.len(), 80 * 100);
        let (lo, hi) = m.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(hi - lo <= 2.0 + 1e-4, "dynamic range is clamped to 8 log10 units = 2.0 after /4: {lo}..{hi}");
        let (_, f30) = log_mel(&pcm, 80, &f, N_SAMPLES_30S);
        assert_eq!(f30, 3000);
    }
}
