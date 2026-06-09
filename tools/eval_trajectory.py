"""Multi-checkpoint WER table with ONE Whisper load (fast comparisons).

    python -m tools.eval_trajectory --checkpoints runs/exp4/deepvqe_step3000.tar runs/exp4/deepvqe_step9000.tar
    python -m tools.eval_trajectory --config configs/eval/librispeech.yaml --checkpoints <a> <b> --device 0
"""
import argparse

from afe.config.defaults import EvalConfig, load_config
from afe.evaluation.trajectory import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--config", default="configs/eval/librispeech.yaml")
    ap.add_argument("--librispeech")
    ap.add_argument("--noise-dir", dest="noise_dir")
    ap.add_argument("--whisper")
    ap.add_argument("--n-utts", dest="n_utts", type=int)
    ap.add_argument("--snrs", type=int, nargs="+")
    ap.add_argument("--num-beams", dest="num_beams", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    over = {k: v for k, v in vars(args).items() if k not in ("checkpoints", "config")}
    cfg = load_config(EvalConfig, args.config, **over)
    res = evaluate(args.checkpoints, cfg.librispeech, cfg.noise_dir, cfg.snrs,
                   cfg.n_utts, cfg.seed, whisper=cfg.whisper,
                   num_beams=cfg.num_beams, device=cfg.device)

    snrs = res["snrs"]
    cols = [f"{s}dB" for s in snrs]
    width = 29 + 9 * len(cols)
    print(f"\n{'model':28s} " + " ".join(f"{c:>8s}" for c in cols))
    print("-" * width)
    print(f"{'clean (upper bound)':28s} " + " ".join(f"{res['clean']*100:>7.2f}%" for _ in cols))
    print(f"{'noisy (baseline)':28s} " + " ".join(f"{res['noisy'][s]*100:>7.2f}%" for s in snrs))
    print("-" * width)
    for name, wers in res["models"].items():
        print(f"{name:28s} " + " ".join(f"{wers[s]*100:>7.2f}%" for s in snrs))


if __name__ == "__main__":
    main()
