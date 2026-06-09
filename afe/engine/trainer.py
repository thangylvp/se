"""Training engine for ASR-aware SE fine-tuning.

DeepVQE is trainable (warm-started); Whisper is frozen. Gradients flow through the
frozen Whisper into DeepVQE via the differentiable log-mel. See exps.md / losses.
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from afe.asr.diff_logmel import DiffLogMel
from afe.checkpoint.io import load_se_model, save_se_checkpoint
from afe.data.mix_dataset import MixDataset, make_collate
from afe.losses.asr_aware import loss_step
from afe.losses.hybrid import HybridLoss
from afe.solver.lr import build_optimizer, cosine_lr


def build(cfg, device):
    """Return (processor, frozen whisper, trainable se, logmel, hybrid)."""
    processor = WhisperProcessor.from_pretrained(cfg.whisper)
    whisper = WhisperForConditionalGeneration.from_pretrained(cfg.whisper)
    whisper.to(device).eval()
    for p in whisper.parameters():
        p.requires_grad_(False)
    se = load_se_model(cfg.checkpoint, device=device)
    se.train()
    logmel = DiffLogMel(processor, device)
    hybrid = HybridLoss().to(device)
    return processor, whisper, se, logmel, hybrid


def _make_loader(cfg, processor, num_workers):
    ds = MixDataset(cfg.librispeech, cfg.noise_dir, snr_choices=cfg.snr_pool, seed=cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                      collate_fn=make_collate(processor), num_workers=num_workers,
                      drop_last=True, persistent_workers=num_workers > 0)


def smoke_test(cfg, device):
    """Gradient-flow sanity: grads reach DeepVQE, none reach frozen Whisper."""
    print("=== SMOKE TEST: gradient flow through frozen Whisper -> DeepVQE ===")
    processor, whisper, se, logmel, hybrid = build(cfg, device)
    ds = MixDataset(cfg.librispeech, cfg.noise_dir, snr_choices=cfg.snr_pool, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        collate_fn=make_collate(processor), num_workers=0)
    batch = next(iter(loader))
    total, ce, se_loss, feat = loss_step(batch, whisper, se, logmel, hybrid, device,
                                         cfg.lambda_ce, cfg.lambda_se, cfg.lambda_feat)
    total.backward()
    g = [p.grad for p in se.parameters() if p.grad is not None]
    gnorm = sum(float(x.norm()) for x in g)
    w_grads = sum(1 for p in whisper.parameters() if p.grad is not None)
    print(f"CE={ce.item():.4f}  HybridLoss={se_loss.item():.4f}  "
          f"FeatLoss={feat.item():.4f}  total={total.item():.4f}")
    print(f"DeepVQE params with grad: {len(g)}/{sum(1 for _ in se.parameters())}  "
          f"grad-norm={gnorm:.3e}")
    print(f"Whisper params with grad (should be 0): {w_grads}")
    ok = len(g) > 0 and gnorm > 0 and w_grads == 0
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return ok


def train(cfg):
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    processor, whisper, se, logmel, hybrid = build(cfg, device)
    loader = _make_loader(cfg, processor, num_workers=cfg.num_workers)
    opt = build_optimizer(se, cfg.lr)

    print(f"Training: steps={cfg.steps} bs={cfg.batch_size} lr={cfg.lr} "
          f"lambda_ce={cfg.lambda_ce} lambda_se={cfg.lambda_se} "
          f"lambda_feat={cfg.lambda_feat} -> {out}")
    step, t0 = 0, time.time()
    ce_ema = se_ema = feat_ema = None
    log_path = out / "train_log.tsv"
    with open(log_path, "w") as logf:
        logf.write("step\tce\tse\tfeat\ttotal\tit_per_s\n")

    while step < cfg.steps:
        for batch in loader:
            lr_now = cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup_steps, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr_now
            total, ce, se_loss, feat = loss_step(
                batch, whisper, se, logmel, hybrid, device,
                cfg.lambda_ce, cfg.lambda_se, cfg.lambda_feat)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(se.parameters(), cfg.grad_clip)
            opt.step()
            ce_ema = ce.item() if ce_ema is None else 0.98 * ce_ema + 0.02 * ce.item()
            se_ema = se_loss.item() if se_ema is None else 0.98 * se_ema + 0.02 * se_loss.item()
            feat_ema = feat.item() if feat_ema is None else 0.98 * feat_ema + 0.02 * feat.item()
            step += 1
            if step % cfg.log_every == 0:
                ips = step / (time.time() - t0)
                print(f"step {step:5d} | CE {ce_ema:.4f} | HybridLoss {se_ema:.4f} "
                      f"| Feat {feat_ema:.4f} | total {total.item():.4f} | {ips:.2f} it/s")
                with open(log_path, "a") as logf:
                    logf.write(f"{step}\t{ce_ema:.5f}\t{se_ema:.5f}\t{feat_ema:.5f}\t"
                               f"{total.item():.5f}\t{ips:.3f}\n")
            if step % cfg.save_every == 0 or step >= cfg.steps:
                ckpt = out / f"deepvqe_step{step}.tar"
                save_se_checkpoint(se, ckpt, step=step)
                print(f"  saved {ckpt}")
            if step >= cfg.steps:
                break
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")
