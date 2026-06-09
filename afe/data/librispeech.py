"""LibriSpeech listing + the deterministic, fixed noisy eval-set builder."""
import glob
import os
import random
from pathlib import Path

from afe.utils.constants import EVAL_N_UTTS, EVAL_SEED


def list_librispeech(root):
    """Return [(utt_id, wav_path, transcript)] for a LibriSpeech split dir.

    Transcripts live in per-chapter `*.trans.txt` files: "<utt_id> <TEXT>".
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


def build_eval_set(librispeech_split, noise_dir, n_utts=EVAL_N_UTTS, seed=EVAL_SEED):
    """Deterministically pick utterances + one noise clip each.

    The SAME clean utterance and SAME noise clip are reused across all SNRs, so
    SNR is the only variable. Returns dicts with everything to reconstruct each
    noisy mixture reproducibly. Do NOT change — breaks comparability with results/.
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
        eval_items.append({
            "uid": uid, "clean_wav": wav, "text": text,
            "noise_wav": noise_path, "mix_seed": rng.randrange(1 << 30),
        })
    return eval_items
