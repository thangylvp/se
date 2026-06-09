"""Frozen Whisper wrapper for transcription + WER (evaluation).

Inference-only (no_grad) with the stock processor. The differentiable log-mel
path needed for the *training* loss lives in `afe.asr.diff_logmel`.
"""
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from afe.utils.constants import DEFAULT_WHISPER, SR
from afe.utils.text import cer as _cer
from afe.utils.text import make_english_normalizer
from afe.utils.text import wer as _wer


class WhisperASR:
    def __init__(self, model_name=DEFAULT_WHISPER, device="cuda:0", dtype=None):
        """dtype: e.g. torch.float16 to fit large models on small GPUs (None=fp32)."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.normalizer = make_english_normalizer(self.processor)

    @torch.inference_mode()
    def transcribe(self, wavs, batch_size=16, language="en", num_beams=1, **gen_kwargs):
        """wavs: list of 1-D float32 np arrays @ 16 kHz. Returns list of strings.

        `language=None` lets Whisper auto-detect. Extra **gen_kwargs pass to
        model.generate (e.g. no_repeat_ngram_size=3 to suppress repetition-loop
        hallucinations on noisy/short audio).
        """
        out = []
        for i in range(0, len(wavs), batch_size):
            batch = [np.asarray(w, dtype=np.float32) for w in wavs[i:i + batch_size]]
            feats = self.processor(
                batch, sampling_rate=SR, return_tensors="pt"
            ).input_features.to(self.device, self.model.dtype)
            ids = self.model.generate(feats, language=language, task="transcribe",
                                      num_beams=num_beams, **gen_kwargs)
            out.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return out

    def wer(self, refs, hyps):
        """Corpus WER under Whisper English normalization (LibriSpeech protocol)."""
        return _wer(refs, hyps, self.normalizer)

    def cer(self, refs, hyps):
        return _cer(refs, hyps, self.normalizer)
