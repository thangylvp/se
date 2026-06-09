"""Audio IO + SNR mixing. All audio is mono float32 @ 16 kHz."""
import numpy as np
import soundfile as sf

from afe.utils.constants import SR


def load_wav(path, sr=SR):
    audio, fs = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if fs != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=sr)
    return audio


def rms(x, eps=1e-12):
    return np.sqrt(np.mean(x ** 2) + eps)


def fit_noise_length(noise, length, rng):
    """Tile/crop noise to exactly `length` samples, with a random start offset."""
    if len(noise) < length:
        reps = int(np.ceil(length / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    start = 0 if len(noise) == length else rng.randint(0, len(noise) - length)
    return noise[start:start + length]


def mix_at_snr(speech, noise, snr_db, rng=None):
    """Additively mix `noise` into `speech` at a target SNR (dB). Speech is the
    scale anchor; a peak-clip guard preserves the speech/noise ratio."""
    import random as _random
    rng = rng or _random
    noise = fit_noise_length(noise, len(speech), rng)
    sp_rms, no_rms = rms(speech), rms(noise)
    target_noise_rms = sp_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / no_rms)
    mixed = speech + noise
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)
