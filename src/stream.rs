//! Streaming listen: one `Utterance` per push-to-talk press (or per wake-word turn). The host pushes 16 kHz mono f32
//! chunks of any size as they arrive from the mic, asks for a partial whenever `partial_due` says enough new audio
//! has come in, and calls `finish` at release. Async only; nothing blocks. `Cancel` stops a running decode at its next
//! token readback (talking again, closing the panel, a deadline).
//! Partials re-run the whole buffer (Whisper has no streaming decoder). On an Apple M5 (Metal) the median per short
//! command clip is 0.14-0.23 s for tiny.en and 0.24 s for base.en, so a partial every 1 s of new audio costs well under the audio time.
use crate::{Transcript, Whisper};
use crate::mel::SAMPLE_RATE;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shared stop flag: clone it into whatever can cancel (the UI, a deadline timer). A decode checks it at every
/// token readback and returns `Err("cancelled")`.
#[derive(Clone, Default, Debug)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A new, unset flag.
    pub fn new() -> Self {
        Self::default()
    }
    /// Ask every decode holding a clone of this flag to stop.
    pub fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    /// Whether [`Cancel::set`] was called.
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// When partials are due, and the longest utterance kept.
#[derive(Clone, Copy, Debug)]
pub struct PartialEvery {
    /// New audio (ms) between partials.
    pub ms: usize,
    /// No partial before this much audio (ms): a word or two.
    pub min_ms: usize,
    /// The buffer is capped at this (ms); later audio is dropped (whisper's window is 30 s).
    pub max_ms: usize,
}

impl Default for PartialEvery {
    fn default() -> Self {
        PartialEvery { ms: 1000, min_ms: 700, max_ms: 29_000 }
    }
}

/// One utterance's audio so far: one per push-to-talk press (or per voice-activity turn). The host pushes 16 kHz
/// mono f32 chunks of any size as they arrive from the mic, asks for a partial whenever [`Utterance::partial_due`]
/// says enough new audio has come in, and calls [`Utterance::finish`] at release.
///
/// ```no_run
/// # async fn run(w: &speech_to_text::Whisper, mic: Vec<Vec<f32>>) -> Result<(), String> {
/// use speech_to_text::{Cancel, PartialEvery, Utterance};
/// let cancel = Cancel::new();
/// let mut u = Utterance::new(PartialEvery::default());
/// for chunk in mic {
///     u.push(&chunk);
///     if u.partial_due() {
///         let t = u.partial(w, Some(&cancel)).await?;
///         println!("so far: {}", t.text);
///     }
/// }
/// let t = u.finish(w, Some(&cancel)).await?;
/// println!("final: {}", t.text);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Utterance {
    pcm: Vec<f32>,
    every: PartialEvery,
    at_last_partial: usize,
    dropped: usize,
}

fn samples(ms: usize) -> usize {
    ms * SAMPLE_RATE / 1000
}

impl Utterance {
    /// An empty utterance with the given partial schedule.
    pub fn new(every: PartialEvery) -> Self {
        Utterance { pcm: Vec::with_capacity(samples(8000)), every, at_last_partial: 0, dropped: 0 }
    }

    /// Append mic audio (16 kHz mono f32). Audio past `max_ms` is dropped and counted.
    pub fn push(&mut self, chunk: &[f32]) {
        let room = samples(self.every.max_ms).saturating_sub(self.pcm.len());
        let take = room.min(chunk.len());
        self.pcm.extend_from_slice(&chunk[..take]);
        self.dropped += chunk.len() - take;
    }

    /// Audio buffered so far, in ms.
    pub fn ms(&self) -> usize {
        self.pcm.len() * 1000 / SAMPLE_RATE
    }

    /// Samples dropped past the cap.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// Whether enough new audio has come in for another partial.
    pub fn partial_due(&self) -> bool {
        self.pcm.len() >= samples(self.every.min_ms) && self.pcm.len() - self.at_last_partial >= samples(self.every.ms)
    }

    /// Text so far (the whole buffer). Marks the partial as taken even if cancelled.
    pub async fn partial(&mut self, w: &Whisper, cancel: Option<&Cancel>) -> Result<Transcript, String> {
        self.at_last_partial = self.pcm.len();
        w.transcribe_with(&self.pcm, cancel).await
    }

    /// The final text, at release. An empty or too-short press gives empty text without running the model.
    pub async fn finish(self, w: &Whisper, cancel: Option<&Cancel>) -> Result<Transcript, String> {
        if self.pcm.len() < samples(250) {
            return Ok(Transcript { text: String::new(), tokens: vec![], ops: 0, readbacks: 0, encoder_positions: 0, mel_ms: 0.0 });
        }
        w.transcribe_with(&self.pcm, cancel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partials_follow_new_audio() {
        let mut u = Utterance::new(PartialEvery { ms: 1000, min_ms: 700, max_ms: 3000 });
        u.push(&vec![0.0; samples(500)]);
        assert!(!u.partial_due(), "below min_ms");
        u.push(&vec![0.0; samples(600)]);
        assert!(u.partial_due(), "1.1 s in, none taken");
        u.at_last_partial = u.pcm.len();
        u.push(&vec![0.0; samples(900)]);
        assert!(!u.partial_due(), "0.9 s new");
        u.push(&vec![0.0; samples(200)]);
        assert!(u.partial_due(), "1.1 s new");
    }

    #[test]
    fn cap_drops_and_counts() {
        let mut u = Utterance::new(PartialEvery { ms: 1000, min_ms: 700, max_ms: 1000 });
        u.push(&vec![0.0; samples(800)]);
        u.push(&vec![0.0; samples(800)]);
        assert_eq!(u.ms(), 1000);
        assert_eq!(u.dropped(), samples(600));
    }

    #[test]
    fn cancel_is_shared() {
        let c = Cancel::new();
        let d = c.clone();
        assert!(!c.is_set());
        d.set();
        assert!(c.is_set());
    }
}
