"""Config dataclasses + a tiny YAML loader (detectron2-lite, no yacs).

Usage:
    cfg = load_config(TrainConfig, "configs/train/exp2.yaml", device="0")
CLI overrides (non-None kwargs) win over YAML, which wins over the dataclass
defaults below.
"""
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import yaml

from afe.utils.constants import DEFAULT_WHISPER, EVAL_N_UTTS, EVAL_SEED, EVAL_SNRS


@dataclass
class TrainConfig:
    # data
    librispeech: str = "../dataset/LibriSpeech/train-clean-100"
    noise_dir: str = "../dataset/musan/noise"
    checkpoint: str = "../ckpt/deepvqe/model_240.tar"   # warm-start
    whisper: str = DEFAULT_WHISPER
    snr_pool: Optional[List[int]] = None                # None -> DEFAULT_SNR_POOL
    # optimization
    steps: int = 3000
    batch_size: int = 8
    lr: float = 1e-4
    warmup_steps: int = 300
    min_lr_ratio: float = 0.1
    grad_clip: float = 5.0
    # loss weights (settled recipe: CE primary, SE small stabilizer, feat off)
    lambda_ce: float = 1.0
    lambda_se: float = 0.05
    lambda_feat: float = 0.0
    # io / runtime
    out_dir: str = "runs/exp"
    save_every: int = 1000
    log_every: int = 25
    num_workers: int = 4
    device: str = "0"
    seed: int = 0


@dataclass
class EvalConfig:
    librispeech: str = "../dataset/LibriSpeech/test-clean"
    noise_dir: str = "../dataset/musan/noise"
    whisper: str = DEFAULT_WHISPER
    n_utts: int = EVAL_N_UTTS
    snrs: List[int] = field(default_factory=lambda: list(EVAL_SNRS))
    seed: int = EVAL_SEED
    num_beams: int = 1
    device: str = "0"


def load_config(cls, path=None, **overrides):
    """Build a config of type `cls` from optional YAML + non-None CLI overrides."""
    cfg = cls()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - set(asdict(cfg))
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        for k, v in data.items():
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
