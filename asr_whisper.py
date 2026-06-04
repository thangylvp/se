"""
Frozen Whisper ASR wrapper for transcription + WER (best-practice HF stack).

Used for evaluation (headroom check, per-checkpoint WER). The differentiable
log-mel path needed for the *training* loss is added separately — this module is
inference-only (no_grad) and uses the stock processor.
"""
import jiwer
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

SR = 16000


class WhisperASR:
    def __init__(self, model_name="openai/whisper-small", device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.normalizer = EnglishTextNormalizer(
            self.processor.tokenizer.english_spelling_normalizer
        )

    @torch.inference_mode()
    def transcribe(self, wavs, batch_size=16, language="en", num_beams=1):
        """wavs: list of 1-D float32 np arrays @ 16 kHz. Returns list of strings.
        num_beams>1 enables beam search (lower WER, slower)."""
        out = []
        for i in range(0, len(wavs), batch_size):
            batch = [np.asarray(w, dtype=np.float32) for w in wavs[i:i + batch_size]]
            feats = self.processor(
                batch, sampling_rate=SR, return_tensors="pt"
            ).input_features.to(self.device)
            ids = self.model.generate(feats, language=language, task="transcribe",
                                      num_beams=num_beams)
            out.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return out

    def wer(self, refs, hyps):
        """Corpus-level WER after Whisper English normalization (the standard
        LibriSpeech/Whisper protocol). Drops pairs that normalize to empty refs."""
        r = [self.normalizer(x) for x in refs]
        h = [self.normalizer(x) for x in hyps]
        pairs = [(a, b) for a, b in zip(r, h) if a.strip()]
        r, h = [a for a, _ in pairs], [b for _, b in pairs]
        return jiwer.wer(r, h)
