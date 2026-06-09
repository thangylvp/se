"""Single-checkpoint headroom WER table (clean / noisy / enhanced x SNRs).

    python -m tools.eval_headroom --checkpoint checkpoints/model_D.tar
    python -m tools.eval_headroom --config configs/eval/librispeech.yaml --checkpoint <ckpt> --device 0
"""
import argparse

from afe.config.defaults import EvalConfig, load_config
from afe.evaluation.headroom import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/eval/librispeech.yaml")
    ap.add_argument("--librispeech")
    ap.add_argument("--noise-dir", dest="noise_dir")
    ap.add_argument("--whisper")
    ap.add_argument("--n-utts", dest="n_utts", type=int)
    ap.add_argument("--snrs", type=int, nargs="+")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    over = {k: v for k, v in vars(args).items() if k not in ("checkpoint", "config")}
    cfg = load_config(EvalConfig, args.config, **over)
    res = evaluate(args.checkpoint, cfg.librispeech, cfg.noise_dir, cfg.snrs,
                   cfg.n_utts, cfg.seed, whisper=cfg.whisper, device=cfg.device)

    print(f"\n{'SNR(dB)':>8} | {'noisy WER':>10} | {'enhanced WER':>12} | {'Δ(enh-noisy)':>12}")
    print("-" * 54)
    for snr, nw, ew, d in res["rows"]:
        print(f"{snr:>8} | {nw*100:>9.2f}% | {ew*100:>11.2f}% | {d*100:>+11.2f}%")
    print("-" * 54)
    print(f"clean WER (upper bound): {res['clean_wer']*100:.2f}%   "
          f"[whisper={res['whisper']}, n={res['n']}]")


if __name__ == "__main__":
    main()
