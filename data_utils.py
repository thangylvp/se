"""
Data utilities: SNR mixing + deterministic noisy eval-set construction.

All audio is mono float32 @ 16 kHz (Whisper and DeepVQE both expect 16 kHz).
"""
import glob
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def load_wav(path, sr=SR):
    audio, fs = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if fs != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=sr)
    return audio


def _rms(x, eps=1e-12):
    return np.sqrt(np.mean(x ** 2) + eps)


def fit_noise_length(noise, length, rng):
    """Tile/crop noise to exactly `length` samples, with a random start offset."""
    if len(noise) < length:
        reps = int(np.ceil(length / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    start = 0 if len(noise) == length else rng.randint(0, len(noise) - length)
    return noise[start:start + length]


def mix_at_snr(speech, noise, snr_db, rng=None):
    """Additively mix `noise` into `speech` at a target SNR (dB).

    SNR = 10*log10(P_speech / P_noise_scaled) -> scale noise power accordingly.
    Returns mixed signal at the SAME scale as `speech` (speech is the anchor).
    """
    rng = rng or random
    noise = fit_noise_length(noise, len(speech), rng)
    sp_rms, no_rms = _rms(speech), _rms(noise)
    target_noise_rms = sp_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / no_rms)
    mixed = speech + noise
    # Guard against clipping while preserving relative speech/noise scale.
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


# ---------- LibriSpeech ----------

def list_librispeech(root):
    """Return [(utt_id, wav_path, transcript)] for a LibriSpeech split dir.

    `root` is e.g. dataset/LibriSpeech/test-clean. Transcripts live in
    per-chapter `*.trans.txt` files: each line is "<utt_id> <TEXT>".
    """
    root = Path(root)
    items = []
    for trans in sorted(root.rglob("*.trans.txt")):
        with open(trans) as f:
            for line in f:
                uid, text = line.strip().split(" ", 1)
                wav = trans.parent / f"{uid}.flac"
                if not wav.exists():
                    wav = trans.parent / f"{uid}.wav"
                items.append((uid, str(wav), text))
    return items


def build_eval_set(librispeech_split, noise_dir, n_utts=200, seed=1234):
    """Deterministically pick utterances + one noise clip each.

    The SAME clean utterance and SAME noise clip are reused across all SNRs, so
    SNR is the only variable. Returns a list of dicts with everything needed to
    reconstruct each noisy mixture reproducibly.
    """
    rng = random.Random(seed)
    utts = list_librispeech(librispeech_split)
    rng.shuffle(utts)
    utts = utts[:n_utts]

    noises = sorted(glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True))
    if not noises:
        raise ValueError(f"No .wav noise files under {noise_dir}")

    eval_items = []
    for uid, wav, text in utts:
        noise_path = noises[rng.randrange(len(noises))]
        # a per-item seed so noise offset is reproducible yet decorrelated
        eval_items.append({
            "uid": uid, "clean_wav": wav, "text": text,
            "noise_wav": noise_path, "mix_seed": rng.randrange(1 << 30),
        })
    return eval_items
