//! Whisper's GPT-2 byte-level BPE, decode only (ids to text) from tokenizer.json: no regex, no `tokenizers` crate, so
//! it is small in wasm. Special tokens (id >= the first `<|...|>`) are skipped.
//! `bytes_to_unicode` is ported from OpenAI's GPT-2 `encoder.py` (MIT); see NOTICE.
use std::collections::HashMap;

pub struct Tokenizer {
    vocab: Vec<String>,
    unicode_to_byte: HashMap<char, u8>,
    special: HashMap<String, u32>,
    pub first_special: u32,
}

/// GPT-2's bytes_to_unicode table.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (b as u8, char::from_u32(c).unwrap())).collect()
}

impl Tokenizer {
    pub fn from_json(json: &str) -> Result<Tokenizer, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let map = v["model"]["vocab"].as_object().ok_or("tokenizer.json: no model.vocab")?;
        let mut special = HashMap::new();
        for t in v["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(c), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) {
                special.insert(c.to_string(), id as u32);
            }
        }
        let n = map.values().filter_map(|x| x.as_u64()).max().unwrap_or(0).max(special.values().copied().max().unwrap_or(0) as u64) as usize + 1;
        let mut vocab = vec![String::new(); n];
        for (tok, id) in map {
            if let Some(id) = id.as_u64() {
                vocab[id as usize] = tok.clone();
            }
        }
        for (tok, id) in &special {
            vocab[*id as usize] = tok.clone();
        }
        let first_special = special.values().copied().min().unwrap_or(n as u32);
        Ok(Tokenizer { vocab, unicode_to_byte: bytes_to_unicode().into_iter().map(|(b, c)| (c, b)).collect(), special, first_special })
    }

    pub fn special(&self, s: &str) -> Option<u32> {
        self.special.get(s).copied()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id >= self.first_special {
                continue;
            }
            if let Some(t) = self.vocab.get(id as usize) {
                bytes.extend(t.chars().filter_map(|c| self.unicode_to_byte.get(&c)));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_roundtrip() {
        let json = r#"{"model":{"vocab":{"Hello":0,"Ġworld":1,"!":2}},"added_tokens":[{"id":3,"content":"<|endoftext|>"}]}"#;
        let t = Tokenizer::from_json(json).unwrap();
        assert_eq!(t.decode(&[0, 1, 2, 3]), "Hello world!");
        assert_eq!(t.special("<|endoftext|>"), Some(3));
        assert_eq!(t.first_special, 3);
    }
}
