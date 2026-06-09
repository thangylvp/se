"""Text normalization + WER/CER metrics, shared by all evaluators.

Two normalization paths:
  - English: Whisper's `EnglishTextNormalizer` (the standard LibriSpeech/Whisper
    protocol). Build one with `make_english_normalizer(processor)`.
  - Vietnamese (and other diacritic languages): `vi_norm` — NFC, lowercase,
    strip punctuation, collapse whitespace, keeping Vietnamese diacritics.

All metric helpers take a `normalize` callable so the caller picks the language;
empty-reference pairs are dropped (standard practice).
"""
import re
import unicodedata

import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# Vietnamese lowercase diacritic letters to KEEP when stripping punctuation.
_VI_CHARS = ("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
             "ùúủũụưừứửữựỳýỷỹỵđ")
_VI_PUNCT = re.compile(rf"[^\w\s{_VI_CHARS}]", re.UNICODE)


def vi_norm(text: str) -> str:
    """NFC, lowercase, drop punctuation, collapse whitespace. Keeps VI diacritics."""
    text = unicodedata.normalize("NFC", text).lower().replace("\xa0", " ")
    text = _VI_PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_english_normalizer(processor) -> EnglishTextNormalizer:
    """English normalizer from a loaded WhisperProcessor."""
    return EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)


def _clean(refs, hyps, normalize):
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a.strip()]
    return [a for a, _ in pairs], [b for _, b in pairs]


def wer(refs, hyps, normalize) -> float:
    """Corpus WER after `normalize`, dropping pairs whose ref normalizes to empty."""
    r, h = _clean(refs, hyps, normalize)
    return jiwer.wer(r, h)


def cer(refs, hyps, normalize) -> float:
    """Corpus CER after `normalize`."""
    r, h = _clean(refs, hyps, normalize)
    return jiwer.cer(r, h)


def wer_cer_breakdown(refs, hyps, normalize) -> dict:
    """Full corpus stats: WER, CER, and word S/D/I/H + ref/hyp word counts."""
    r, h = _clean(refs, hyps, normalize)
    w = jiwer.process_words(r, h)
    c = jiwer.process_characters(r, h)
    return {"n": len(r), "wer": w.wer, "cer": c.cer,
            "S": w.substitutions, "D": w.deletions, "I": w.insertions, "H": w.hits,
            "ref_w": sum(len(x.split()) for x in r),
            "hyp_w": sum(len(x.split()) for x in h)}
