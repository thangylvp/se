"""
Step-1 headroom check (go/no-go before any training).

For a fixed eval set drawn from LibriSpeech test-clean, mixed with noise at each
target SNR, compute Whisper WER under three conditions:
    clean        - upper bound
    noisy        - the baseline to beat
    enhanced     - noisy passed through the current DeepVQE checkpoint

If noisy WER is not clearly worse than clean, there is no headroom at that SNR.
If `enhanced` is worse than `noisy`, the current SE actively hurts ASR (the
motivation for ASR-aware fine-tuning).

Example:
    python eval_headroom.py \
        --librispeech ../dataset/LibriSpeech/test-clean \
        --noise-dir ../dataset/musan/noise \
        --checkpoint "../ckpt/deepvqe/model_240 2.tar" \
        --n-utts 200 --snrs -10 -5 0 5 10 20 --device 0
"""
import argparse
import random

import numpy as np
import torch

from asr_whisper import WhisperASR
from data_utils import build_eval_set, load_wav, mix_at_snr
from se_model import load_se_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True, help="LibriSpeech test-clean dir")
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="DeepVQE .tar")
    ap.add_argument("--whisper", default="openai/whisper-small")
    ap.add_argument("--n-utts", type=int, default=200)
    ap.add_argument("--snrs", type=int, nargs="+", default=[-10, -5, 0, 5, 10, 20])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    asr = WhisperASR(args.whisper, device=device)
    se = load_se_model(args.checkpoint, device=device)

    items = build_eval_set(args.librispeech, args.noise_dir,
                           n_utts=args.n_utts, seed=args.seed)
    refs = [it["text"] for it in items]
    cleans = [load_wav(it["clean_wav"]) for it in items]
    noises = [load_wav(it["noise_wav"]) for it in items]

    @torch.inference_mode()
    def enhance(wav):
        t = torch.from_numpy(np.asarray(wav, np.float32)).unsqueeze(0).to(device)
        return se(t).squeeze(0).cpu().numpy()

    # clean baseline (SNR-independent)
    clean_wer = asr.wer(refs, asr.transcribe(cleans))

    print(f"\n{'SNR(dB)':>8} | {'noisy WER':>10} | {'enhanced WER':>12} | "
          f"{'Δ(enh-noisy)':>12}")
    print("-" * 54)
    rows = [("clean", clean_wer, None, None)]
    for snr in args.snrs:
        noisy, enhanced = [], []
        for clean, noise, it in zip(cleans, noises, items):
            rng = random.Random(it["mix_seed"])
            mix = mix_at_snr(clean, noise, snr, rng=rng)
            noisy.append(mix)
            enhanced.append(enhance(mix))
        nw = asr.wer(refs, asr.transcribe(noisy))
        ew = asr.wer(refs, asr.transcribe(enhanced))
        rows.append((snr, nw, ew, ew - nw))
        print(f"{snr:>8} | {nw*100:>9.2f}% | {ew*100:>11.2f}% | {(ew-nw)*100:>+11.2f}%")

    print("-" * 54)
    print(f"clean WER (upper bound): {clean_wer*100:.2f}%   "
          f"[whisper={args.whisper}, n={len(items)}]")


if __name__ == "__main__":
    main()
