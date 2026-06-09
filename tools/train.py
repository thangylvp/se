"""Train (or smoke-test) the ASR-aware SE fine-tune.

    python -m tools.train --config configs/train/exp2.yaml --device 0
    python -m tools.train --config configs/train/exp2.yaml --steps 3000 --out-dir runs/myexp
    python -m tools.train --smoke --librispeech ../dataset/LibriSpeech/test-clean   # grad-flow check
CLI flags override the YAML; the YAML overrides dataclass defaults.
"""
import argparse

import torch

from afe.config.defaults import TrainConfig, load_config
from afe.engine.trainer import smoke_test, train


def main():
    ap = argparse.ArgumentParser(description="ASR-aware SE fine-tune.")
    ap.add_argument("--config", default=None, help="YAML in configs/train/")
    ap.add_argument("--smoke", action="store_true", help="gradient-flow check, no training")
    # overrides (None => fall back to YAML/defaults)
    ap.add_argument("--checkpoint")
    ap.add_argument("--librispeech")
    ap.add_argument("--noise-dir", dest="noise_dir")
    ap.add_argument("--whisper")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lambda-ce", dest="lambda_ce", type=float)
    ap.add_argument("--lambda-se", dest="lambda_se", type=float)
    ap.add_argument("--lambda-feat", dest="lambda_feat", type=float)
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "smoke")}
    cfg = load_config(TrainConfig, args.config, **overrides)

    if args.smoke:
        device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        smoke_test(cfg, device)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
