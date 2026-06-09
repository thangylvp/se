"""Single-checkpoint headroom WER table: clean / noisy / enhanced across SNRs.

Returns structured rows; the CLI (tools/eval_headroom.py) handles printing.
"""
import random

import numpy as np
import torch

from afe.asr.whisper import WhisperASR
from afe.checkpoint.io import load_se_model
from afe.data.audio import load_wav, mix_at_snr
from afe.data.librispeech import build_eval_set


def _enhance(se, wav, device):
    t = torch.from_numpy(np.asarray(wav, np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        return se(t).squeeze(0).cpu().numpy()


def evaluate(checkpoint, librispeech, noise_dir, snrs, n_utts, seed,
             whisper="openai/whisper-small", device="0"):
    """Return {'clean_wer': float, 'rows': [(snr, noisy_wer, enh_wer, delta), ...]}."""
    dev = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    asr = WhisperASR(whisper, device=dev)
    se = load_se_model(checkpoint, device=dev)

    items = build_eval_set(librispeech, noise_dir, n_utts=n_utts, seed=seed)
    refs = [it["text"] for it in items]
    cleans = [load_wav(it["clean_wav"]) for it in items]
    noises = [load_wav(it["noise_wav"]) for it in items]

    clean_wer = asr.wer(refs, asr.transcribe(cleans))
    rows = []
    for snr in snrs:
        noisy, enhanced = [], []
        for clean, noise, it in zip(cleans, noises, items):
            mix = mix_at_snr(clean, noise, snr, rng=random.Random(it["mix_seed"]))
            noisy.append(mix)
            enhanced.append(_enhance(se, mix, dev))
        nw = asr.wer(refs, asr.transcribe(noisy))
        ew = asr.wer(refs, asr.transcribe(enhanced))
        rows.append((snr, nw, ew, ew - nw))
    return {"clean_wer": clean_wer, "rows": rows, "n": len(items), "whisper": whisper}
