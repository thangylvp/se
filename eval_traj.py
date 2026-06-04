"""Evaluate multiple SE checkpoints on the fixed eval set with ONE Whisper load.
Reuses clean/noisy references across checkpoints. Prints a WER-vs-checkpoint table."""
import argparse
import random

import numpy as np
import torch

from asr_whisper import WhisperASR
from data_utils import build_eval_set, load_wav, mix_at_snr
from se_model import load_se_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", default="../dataset/LibriSpeech/test-clean")
    ap.add_argument("--noise-dir", default="../dataset/musan/noise")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--whisper", default="openai/whisper-small")
    ap.add_argument("--n-utts", type=int, default=200)
    ap.add_argument("--snrs", type=int, nargs="+", default=[-10, -5, 0, 5, 10, 20])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    asr = WhisperASR(args.whisper, device=device)

    items = build_eval_set(args.librispeech, args.noise_dir, n_utts=args.n_utts, seed=args.seed)
    refs = [it["text"] for it in items]
    cleans = [load_wav(it["clean_wav"]) for it in items]
    noises = [load_wav(it["noise_wav"]) for it in items]

    # noisy mixtures per SNR (computed once, reused for every checkpoint)
    noisy_by_snr = {}
    for snr in args.snrs:
        mixes = []
        for clean, noise, it in zip(cleans, noises, items):
            mixes.append(mix_at_snr(clean, noise, snr, rng=random.Random(it["mix_seed"])))
        noisy_by_snr[snr] = mixes

    clean_wer = asr.wer(refs, asr.transcribe(cleans, num_beams=args.num_beams))
    noisy_wer = {snr: asr.wer(refs, asr.transcribe(noisy_by_snr[snr], num_beams=args.num_beams)) for snr in args.snrs}

    # header
    cols = [f"{s}dB" for s in args.snrs]
    print(f"\n{'model':28s} " + " ".join(f"{c:>8s}" for c in cols))
    print("-" * (29 + 9 * len(cols)))
    print(f"{'clean (upper bound)':28s} " + " ".join(f"{clean_wer*100:>7.2f}%" for _ in cols))
    print(f"{'noisy (baseline)':28s} " + " ".join(f"{noisy_wer[s]*100:>7.2f}%" for s in args.snrs))
    print("-" * (29 + 9 * len(cols)))

    @torch.inference_mode()
    def enhance_all(se, mixes):
        out = []
        for w in mixes:
            t = torch.from_numpy(np.asarray(w, np.float32)).unsqueeze(0).to(device)
            out.append(se(t).squeeze(0).cpu().numpy())
        return out

    for ckpt in args.checkpoints:
        se = load_se_model(ckpt, device=device)
        name = ckpt.split("/")[-1].replace("deepvqe_exp2_", "").replace(".tar", "")
        wers = {s: asr.wer(refs, asr.transcribe(enhance_all(se, noisy_by_snr[s]), num_beams=args.num_beams)) for s in args.snrs}
        print(f"{name:28s} " + " ".join(f"{wers[s]*100:>7.2f}%" for s in args.snrs))
        del se
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
