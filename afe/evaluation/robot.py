"""Real-data evaluation on the Vietnamese robot test set.

Transcribes each condition (noisy + one per checkpoint) and computes WER/CER at
the overall / subset / category / per-file levels, plus per-utterance win-loss.
Consolidates the old eval_robot, eval_robot_detail, make_per_file_csv,
make_folder_report scripts. CLI: tools/eval_robot.py.
"""
from collections import defaultdict

import jiwer
import numpy as np
import torch

from afe.asr.whisper import WhisperASR
from afe.checkpoint.io import load_se_model
from afe.data.audio import load_wav
from afe.data.robot import build_pairs
from afe.utils.text import vi_norm, wer_cer_breakdown


def enhance_all(se, wavs, device):
    out = []
    with torch.inference_mode():
        for w in wavs:
            x = torch.from_numpy(w).unsqueeze(0).to(device)
            out.append(se(x).squeeze(0).cpu().numpy().astype(np.float32))
    return out


def run(checkpoints, data_root="../data_test_robot", whisper="openai/whisper-small",
        dtype=None, language="vi", no_repeat_ngram=3, batch_size=16, device="0"):
    """checkpoints: list of (name, path); path=None -> raw 'noisy' condition.

    Returns (pairs, hyps) where hyps maps condition-name -> list of transcripts.
    """
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    gk = {"no_repeat_ngram_size": no_repeat_ngram} if no_repeat_ngram > 0 else {}

    pairs = build_pairs(data_root, verbose=False)
    wavs = [load_wav(p["wav"]) for p in pairs]
    asr = WhisperASR(whisper, device=str(dev), dtype=dtype)

    hyps = {}
    for name, path in checkpoints:
        sig = wavs if path is None else enhance_all(load_se_model(path, device=dev), wavs, dev)
        torch.cuda.empty_cache()
        hyps[name] = asr.transcribe(sig, batch_size=batch_size, language=language, **gk)
    return pairs, hyps


# ---------- aggregation ----------
def aggregate(pairs, hyps, level="cat"):
    """level: 'OVERALL' | 'subset' | 'cat'. Returns {group: {cond: breakdown}}."""
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        key = "OVERALL" if level == "OVERALL" else (p["subset"] if level == "subset" else p["cat"])
        groups[key].append(i)
    result = {}
    for key, idx in groups.items():
        refs = [pairs[i]["text"] for i in idx]
        result[key] = {c: wer_cer_breakdown(refs, [hyps[c][i] for i in idx], vi_norm)
                       for c in hyps}
    return result


def per_file_records(pairs, hyps):
    """One dict per file: file, folder, ref, per-cond hyp + per-cond wer/cer."""
    conds = list(hyps)
    recs = []
    for i, p in enumerate(pairs):
        rec = {"file": p["wav"].split("/")[-1], "folder": p["cat"], "gt": p["text"]}
        r = vi_norm(p["text"])
        for c in conds:
            h = hyps[c][i].strip()
            rec[f"out_{c}"] = h
            hn = vi_norm(h)
            rec[f"wer_{c}"] = jiwer.wer(r, hn) if r.strip() else None
            rec[f"cer_{c}"] = jiwer.cer(r, hn) if r.strip() else None
        recs.append(rec)
    return recs


def folder_means(records, conds):
    """Mean per-file WER/CER per folder + win/loss of each non-base cond vs base."""
    base = conds[0]
    agg = defaultdict(lambda: {c: {"wer": [], "cer": []} for c in conds})
    wl = defaultdict(lambda: {c: [0, 0, 0] for c in conds[1:]})  # [impr, worse, tie]
    for r in records:
        f = r["folder"]
        if any(r[f"cer_{c}"] is None for c in conds):
            continue
        for c in conds:
            agg[f][c]["wer"].append(r[f"wer_{c}"])
            agg[f][c]["cer"].append(r[f"cer_{c}"])
        for c in conds[1:]:
            d = r[f"cer_{c}"] - r[f"cer_{base}"]
            wl[f][c][0 if d < -1e-6 else 1 if d > 1e-6 else 2] += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    out = {}
    for f in agg:
        out[f] = {"n": len(agg[f][base]["wer"]),
                  **{c: {"wer": mean(agg[f][c]["wer"]), "cer": mean(agg[f][c]["cer"])}
                     for c in conds},
                  "winloss": {c: tuple(wl[f][c]) for c in conds[1:]}}
    return out


def win_loss(records, base, cond):
    """(improved, worse, tie) over all files by per-utterance CER."""
    impr = worse = tie = 0
    for r in records:
        if r[f"cer_{base}"] is None or r[f"cer_{cond}"] is None:
            continue
        d = r[f"cer_{cond}"] - r[f"cer_{base}"]
        if d < -1e-6:
            impr += 1
        elif d > 1e-6:
            worse += 1
        else:
            tie += 1
    return impr, worse, tie
