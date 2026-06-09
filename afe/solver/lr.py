"""Optimizer + LR schedule for SE fine-tuning."""
import math

import torch


def cosine_lr(step, total, base, warmup, min_ratio):
    """Linear warmup then cosine decay to min_ratio*base."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def build_optimizer(se_model, lr):
    """AdamW over the trainable SE params only."""
    return torch.optim.AdamW(
        [p for p in se_model.parameters() if p.requires_grad], lr=lr)
