//! M=1 decode attention over a fixed-capacity, head-major KV cache, in plain burn ops (the layout and flags of a
//! one-dispatch decode-attention kernel, so a kernel can replace the body).
//! - Caches are `[n_kv_heads, capacity, head_dim]` (batch 1 dropped), allocated once: no reallocation, no `cat`.
//! - `append`: k_new/v_new (`[q_len, n_kv_heads, head_dim]`) go in at `kv_len`; off = a read-only KV (cross-attention:
//!   any buffer in this layout, e.g. the encoder's K/V projection, with capacity = seq).
//! - `causal`: bottom-right aligned (query i sees keys < kv_len - q_len + i + 1), so a q_len 2 first step works.
//! - Plain MHA here (n_kv_heads == n_q_heads, no RoPE: Whisper adds learned positions before).
//! kv_len lives on the CPU in the fallback; the kernel keeps it in a GPU buffer it bumps itself.
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{module, Bool, Device, Tensor, TensorData};

pub struct KvCache {
    pub k: Tensor<3>,
    pub v: Tensor<3>,
    pub capacity: usize,
    pub kv_len: usize,
}

impl KvCache {
    pub fn new(heads: usize, capacity: usize, head_dim: usize, dev: &Device) -> KvCache {
        KvCache { k: Tensor::zeros([heads, capacity, head_dim], dev), v: Tensor::zeros([heads, capacity, head_dim], dev), capacity, kv_len: 0 }
    }
    /// A read-only KV from existing buffers (`[heads, seq, head_dim]`), kv_len = seq.
    pub fn fixed(k: Tensor<3>, v: Tensor<3>) -> KvCache {
        let n = k.dims()[1];
        KvCache { k, v, capacity: n, kv_len: n }
    }
}

pub struct Flags {
    pub causal: bool,
    pub append: bool,
    pub scale: f64,
}

/// q: `[q_len, n_q_heads, head_dim]`; returns `[q_len, n_q_heads, head_dim]`.
pub fn decode_attn(q: Tensor<3>, cache: &mut KvCache, kv_new: Option<(Tensor<3>, Tensor<3>)>, f: &Flags) -> Result<Tensor<3>, String> {
    let [q_len, heads, d] = q.dims();
    if f.append {
        let (kn, vn) = kv_new.ok_or("append without k_new/v_new")?;
        if cache.kv_len + q_len > cache.capacity {
            return Err(format!("kv cache full: {} + {q_len} > {}", cache.kv_len, cache.capacity));
        }
        let r = cache.kv_len..cache.kv_len + q_len;
        crate::op(2);
        cache.k = cache.k.clone().slice_assign([0..heads, r.clone(), 0..d], kn.swap_dims(0, 1));
        cache.v = cache.v.clone().slice_assign([0..heads, r, 0..d], vn.swap_dims(0, 1));
        cache.kv_len += q_len;
    }
    let n = cache.kv_len;
    let k = cache.k.clone().narrow(1, 0, n).unsqueeze_dim::<4>(0);
    let v = cache.v.clone().narrow(1, 0, n).unsqueeze_dim::<4>(0);
    let q4 = q.swap_dims(0, 1).unsqueeze_dim::<4>(0); // [1, heads, q_len, d]
    let mask = (f.causal && q_len > 1).then(|| {
        let m: Vec<bool> = (0..q_len).flat_map(|i| (0..n).map(move |j| j > n - q_len + i)).collect();
        Tensor::<4, Bool>::from_data(TensorData::new(m, [1, 1, q_len, n]), &q4.device())
    });
    crate::op(1);
    let o = module::attention(q4, k, v, mask, None, AttentionModuleOptions { scale: Some(f.scale), softcap: None, is_causal: false });
    Ok(o.squeeze_dim::<3>(0).swap_dims(0, 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::flex()
    }

    /// Two steps through the cache (q_len 2 causal, then q_len 1) match one full causal attention over all 3.
    #[test]
    fn cache_matches_full_causal() {
        let d = dev();
        let (h, hd) = (2, 8);
        let q = Tensor::<3>::random([3, h, hd], Distribution::Normal(0.0, 1.0), &d);
        let k = Tensor::<3>::random([3, h, hd], Distribution::Normal(0.0, 1.0), &d);
        let v = Tensor::<3>::random([3, h, hd], Distribution::Normal(0.0, 1.0), &d);
        let fl = Flags { causal: true, append: true, scale: 0.35 };
        let mut c = KvCache::new(h, 8, hd, &d);
        let a = decode_attn(q.clone().narrow(0, 0, 2), &mut c, Some((k.clone().narrow(0, 0, 2), v.clone().narrow(0, 0, 2))), &fl).unwrap();
        let b = decode_attn(q.clone().narrow(0, 2, 1), &mut c, Some((k.clone().narrow(0, 2, 1), v.clone().narrow(0, 2, 1))), &fl).unwrap();
        assert_eq!(c.kv_len, 3);
        let got = Tensor::cat(vec![a, b], 0);
        let full = module::attention(q.swap_dims(0, 1).unsqueeze_dim::<4>(0), k.swap_dims(0, 1).unsqueeze_dim::<4>(0), v.swap_dims(0, 1).unsqueeze_dim::<4>(0), None, None,
            AttentionModuleOptions { scale: Some(0.35), softcap: None, is_causal: true }).squeeze_dim::<3>(0).swap_dims(0, 1);
        let diff = (got - full).abs().max().into_data().to_vec::<f32>().unwrap()[0];
        assert!(diff < 1e-5, "diff {diff}");
    }

    #[test]
    fn full_cache_errors() {
        let d = dev();
        let mut c = KvCache::new(1, 1, 4, &d);
        let x = || Tensor::<3>::zeros([1, 1, 4], &d);
        let fl = Flags { causal: true, append: true, scale: 0.5 };
        decode_attn(x(), &mut c, Some((x(), x())), &fl).unwrap();
        assert!(decode_attn(x(), &mut c, Some((x(), x())), &fl).is_err());
    }
}
