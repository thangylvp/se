"""Multi-checkpoint WER eval with ONE Whisper load (fast comparisons).

Reuses clean/noisy references across all checkpoints. Returns structured results;
the CLI (tools/eval_trajectory.py) prints the table.
"""
import random

import numpy as np
import torch

from afe.asr.whisper import WhisperASR
from afe.checkpoint.io import load_se_model
from afe.data.audio import load_wav, mix_at_snr
from afe.data.librispeech import build_eval_set


def _enhance_all(se, mixes, device):
    out = []
    with torch.inference_mode():
        for w in mixes:
            t = torch.from_numpy(np.asarray(w, np.float32)).unsqueeze(0).to(device)
            out.append(se(t).squeeze(0).cpu().numpy())
    return out


def evaluate(checkpoints, librispeech, noise_dir, snrs, n_utts, seed,
             whisper="openai/whisper-small", num_beams=1, device="0"):
    """Return {'clean': wer, 'noisy': {snr: wer}, 'models': {name: {snr: wer}}}."""
    dev = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    asr = WhisperASR(whisper, device=dev)

    items = build_eval_set(librispeech, noise_dir, n_utts=n_utts, seed=seed)
    refs = [it["text"] for it in items]
    cleans = [load_wav(it["clean_wav"]) for it in items]
    noises = [load_wav(it["noise_wav"]) for it in items]

    noisy_by_snr = {snr: [mix_at_snr(c, n, snr, rng=random.Random(it["mix_seed"]))
                          for c, n, it in zip(cleans, noises, items)]
                    for snr in snrs}

    clean = asr.wer(refs, asr.transcribe(cleans, num_beams=num_beams))
    noisy = {s: asr.wer(refs, asr.transcribe(noisy_by_snr[s], num_beams=num_beams))
             for s in snrs}

    models = {}
    for ckpt in checkpoints:
        se = load_se_model(ckpt, device=dev)
        name = ckpt.split("/")[-1].replace("deepvqe_", "").replace(".tar", "")
        models[name] = {s: asr.wer(refs, asr.transcribe(
            _enhance_all(se, noisy_by_snr[s], dev), num_beams=num_beams)) for s in snrs}
        del se
        torch.cuda.empty_cache()
    return {"clean": clean, "noisy": noisy, "models": models, "snrs": list(snrs),
            "n": len(items), "whisper": whisper}
