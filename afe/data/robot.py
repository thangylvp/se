"""Pairing for the real Vietnamese robot test set (../data_test_robot).

Layout: audio at <subset>/<cat>/<stem>.wav (or <subset>/audio_16k/<cat>/<stem>.wav
for recording_2404); transcript at <subset>/transcripts/<cat>/<stem>.txt.
"""
import glob
import os
from pathlib import Path

DEFAULT_ROOT = "../data_test_robot"
DEFAULT_SUBSETS = ("vmo_1703_filter_16k", "vmo_2305_16k", "recording_2404_filter_16k")


def find_wav(data_root, subset, cat, stem):
    for cand in (
        os.path.join(data_root, subset, cat, f"{stem}.wav"),
        os.path.join(data_root, subset, "audio_16k", cat, f"{stem}.wav"),
    ):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(data_root, subset, "**", cat, f"{stem}.wav"),
                     recursive=True)
    return hits[0] if hits else None


def build_pairs(data_root=DEFAULT_ROOT, subsets=DEFAULT_SUBSETS, verbose=True):
    """Return [{subset, cat, wav, text}] for every transcript with a matching wav."""
    pairs = []
    for subset in subsets:
        tdir = os.path.join(data_root, subset, "transcripts")
        for t in sorted(glob.glob(os.path.join(tdir, "**", "*.txt"), recursive=True)):
            rel = os.path.relpath(t, tdir)
            cat = os.path.dirname(rel)
            stem = Path(rel).stem
            wav = find_wav(data_root, subset, cat, stem)
            if wav is None:
                if verbose:
                    print(f"  [skip] no wav for {t}")
                continue
            pairs.append({"subset": subset, "cat": f"{subset}/{cat}", "wav": wav,
                          "text": open(t, encoding="utf-8").read().strip()})
    return pairs


def autofind_transcript(wav_path):
    """Walk up from a wav looking for a sibling 'transcripts/' tree holding <stem>.txt."""
    stem = Path(wav_path).stem
    d = Path(wav_path).resolve().parent
    for _ in range(4):
        tdir = d / "transcripts"
        if tdir.is_dir():
            hits = glob.glob(str(tdir / "**" / f"{stem}.txt"), recursive=True)
            if hits:
                return hits[0]
        d = d.parent
    return None
