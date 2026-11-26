//! Moonshine's tokenizer (a SentencePiece-style BPE with `▁` for spaces and `<0xXX>` byte fallback), decode only
//! (ids to text) from tokenizer.json. Added tokens marked `"special": true` are skipped, as Hugging Face's
//! `decode(..., skip_special_tokens=True)` does.
use std::collections::HashMap;

pub struct SpTokenizer {
    vocab: Vec<String>,
    skip: Vec<bool>,
    special: HashMap<String, u32>,
}

/// `<0xAB>` to the byte 0xAB.
fn byte_piece(p: &str) -> Option<u8> {
    let h = p.strip_prefix("<0x")?.strip_suffix('>')?;
    if h.len() == 2 { u8::from_str_radix(h, 16).ok() } else { None }
}

impl SpTokenizer {
    pub fn from_json(json: &str) -> Result<SpTokenizer, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let map = v["model"]["vocab"].as_object().ok_or("tokenizer.json: no model.vocab")?;
        let added: Vec<(String, u32, bool)> = v["added_tokens"].as_array().into_iter().flatten()
            .filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32, t["special"].as_bool().unwrap_or(false))))
            .collect();
        let n = map.values().filter_map(|x| x.as_u64()).max().unwrap_or(0).max(added.iter().map(|a| a.1 as u64).max().unwrap_or(0)) as usize + 1;
        let mut vocab = vec![String::new(); n];
        for (tok, id) in map {
            if let Some(id) = id.as_u64() {
                vocab[id as usize] = tok.clone();
            }
        }
        let mut skip = vec![false; n];
        let mut special = HashMap::new();
        for (c, id, sp) in added {
            vocab[id as usize] = c.clone();
            skip[id as usize] = sp;
            special.insert(c, id);
        }
        Ok(SpTokenizer { vocab, skip, special })
    }

    pub fn special(&self, s: &str) -> Option<u32> {
        self.special.get(s).copied()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i >= self.vocab.len() || self.skip[i] {
                continue;
            }
            let p = &self.vocab[i];
            match byte_piece(p) {
                Some(b) => bytes.push(b),
                None => bytes.extend(p.replace('\u{2581}', " ").as_bytes()),
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metaspace_and_bytes() {
        let json = r#"{"model":{"vocab":{"<unk>":0,"<s>":1,"</s>":2,"▁Hello":3,"▁wor":4,"ld":5,"<0x21>":6,"<0xC3>":7,"<0xA9>":8}},
            "added_tokens":[{"id":0,"content":"<unk>","special":true},{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]}"#;
        let t = SpTokenizer::from_json(json).unwrap();
        assert_eq!(t.decode(&[1, 3, 4, 5, 6, 7, 8, 2]), "Hello world!\u{e9}");
        assert_eq!(t.special("</s>"), Some(2));
    }
}
