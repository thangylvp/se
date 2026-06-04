"""
Exp 2 — ASR-aware fine-tuning of DeepVQE against a frozen Whisper-small.

Loss = lambda_ce * CE(WhisperDecoder(enhanced), transcript)
     + lambda_se * HybridLoss(enhanced, clean)

DeepVQE is trainable (warm-started from the teammate's checkpoint); Whisper is
frozen (eval, no grad on params — gradients still flow through it into DeepVQE
via the differentiable log-mel). CE is the primary driver; HybridLoss is a small
stabilizer (Dissen et al., Interspeech 2024).

Smoke test (gradient-flow sanity, no real training):
    python train_exp2.py --smoke

Real run:
    python train_exp2.py --steps 3000 --batch-size 8 --lr 1e-4 \
        --lambda-ce 1.0 --lambda-se 0.02 --out-dir runs/exp2 --device 0
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data_utils import fit_noise_length, list_librispeech, load_wav, mix_at_snr
from diff_logmel import DiffLogMel
from losses import HybridLoss
from se_model import load_se_model

SR = 16000
MAX_TRAIN_SAMPLES = 20 * SR  # cap training utts at 20 s for batching sanity


# ---------------- dataset ----------------
class MixDataset(Dataset):
    """On-the-fly clean+noise mixing. SNR sampled across the full range but
    weighted toward the −5..+5 dB band where the current SE hurts WER."""

    def __init__(self, librispeech_dir, noise_dir, snr_choices=None, seed=0):
        self.utts = [u for u in list_librispeech(librispeech_dir)]
        self.noises = sorted(Path(noise_dir).rglob("*.wav"))
        # weighted SNR pool: emphasize the failure band
        self.snr_pool = snr_choices or (
            [-10, -5, -5, 0, 0, 0, 5, 5, 10, 20]
        )
        self.seed = seed

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx)
        uid, wav_path, text = self.utts[idx]
        clean = load_wav(wav_path)
        if len(clean) > MAX_TRAIN_SAMPLES:
            start = rng.randint(0, len(clean) - MAX_TRAIN_SAMPLES)
            clean = clean[start:start + MAX_TRAIN_SAMPLES]
            text = None  # transcript no longer matches a crop -> skip CE for this item
        noise = load_wav(str(self.noises[rng.randrange(len(self.noises))]))
        noise = fit_noise_length(noise, len(clean), rng)
        snr = rng.choice(self.snr_pool)
        noisy = mix_at_snr(clean, noise, snr, rng=rng)
        return {"noisy": noisy.astype(np.float32),
                "clean": clean.astype(np.float32),
                "text": text, "snr": snr}


def make_collate(processor):
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="en", task="transcribe")

    def collate(batch):
        L = max(len(b["noisy"]) for b in batch)
        noisy = torch.zeros(len(batch), L)
        clean = torch.zeros(len(batch), L)
        for i, b in enumerate(batch):
            noisy[i, :len(b["noisy"])] = torch.from_numpy(b["noisy"])
            clean[i, :len(b["clean"])] = torch.from_numpy(b["clean"])
        # labels: pad with -100 (ignored by CE). Items with text=None get all -100.
        label_ids = [tok(b["text"]).input_ids if b["text"] else [] for b in batch]
        Ll = max((len(x) for x in label_ids), default=1) or 1
        labels = torch.full((len(batch), Ll), -100, dtype=torch.long)
        for i, ids in enumerate(label_ids):
            if ids:
                labels[i, :len(ids)] = torch.tensor(ids)
        return {"noisy": noisy, "clean": clean, "labels": labels}

    return collate


# ---------------- training ----------------
def cosine_lr(step, total, base, warmup, min_ratio):
    """Linear warmup then cosine decay to min_ratio*base."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def build(args, device):
    processor = WhisperProcessor.from_pretrained(args.whisper)
    whisper = WhisperForConditionalGeneration.from_pretrained(args.whisper)
    whisper.to(device).eval()
    for p in whisper.parameters():
        p.requires_grad_(False)
    se = load_se_model(args.checkpoint, device=device)  # STFTWrapper(DeepVQE)
    se.train()
    logmel = DiffLogMel(processor, device)
    hybrid = HybridLoss().to(device)
    return processor, whisper, se, logmel, hybrid


# encoder hidden-state layers to match for the feature loss (early/mid; lit:
# Plantinga/Kataria found shallow layers transfer best). Indices into the
# encoder hidden_states tuple (0 = conv embedding, 1..12 = after each block).
FEAT_LAYERS = (2, 4, 6)


def loss_step(batch, whisper, se, logmel, hybrid, device, lambda_ce, lambda_se,
              lambda_feat=0.0):
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)
    labels = batch["labels"].to(device)

    enhanced = se(noisy)                          # (B, L) differentiable
    mel = logmel(enhanced)                         # (B, 80, 3000)
    # CE through frozen Whisper; also grab encoder hidden states if needed
    out = whisper(input_features=mel, labels=labels,
                  output_hidden_states=(lambda_feat > 0))
    ce = out.loss
    se_loss = hybrid(enhanced, clean)

    feat_loss = torch.zeros((), device=device)
    if lambda_feat > 0:
        with torch.no_grad():
            clean_mel = logmel(clean)
            clean_h = whisper.model.encoder(
                clean_mel, output_hidden_states=True).hidden_states
        enh_h = out.encoder_hidden_states
        feat_loss = sum(F.l1_loss(enh_h[l], clean_h[l]) for l in FEAT_LAYERS) / len(FEAT_LAYERS)

    total = lambda_ce * ce + lambda_se * se_loss + lambda_feat * feat_loss
    return total, ce.detach(), se_loss.detach(), feat_loss.detach()


def smoke_test(args, device):
    print("=== SMOKE TEST: gradient flow through frozen Whisper -> DeepVQE ===")
    processor, whisper, se, logmel, hybrid = build(args, device)
    ds = MixDataset(args.librispeech, args.noise_dir, seed=0)
    collate = make_collate(processor)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate,
                        num_workers=0)
    batch = next(iter(loader))
    total, ce, se_loss, feat = loss_step(batch, whisper, se, logmel, hybrid, device,
                                         args.lambda_ce, args.lambda_se, args.lambda_feat)
    total.backward()
    # check grads reached DeepVQE
    g = [p.grad for n, p in se.named_parameters() if p.grad is not None]
    gnorm = sum(float(x.norm()) for x in g)
    n_with_grad = len(g)
    n_total = sum(1 for _ in se.parameters())
    # confirm Whisper got no param grads
    w_grads = sum(1 for p in whisper.parameters() if p.grad is not None)
    print(f"CE={ce.item():.4f}  HybridLoss={se_loss.item():.4f}  "
          f"FeatLoss={feat.item():.4f}  total={total.item():.4f}")
    print(f"DeepVQE params with grad: {n_with_grad}/{n_total}  grad-norm={gnorm:.3e}")
    print(f"Whisper params with grad (should be 0): {w_grads}")
    ok = n_with_grad > 0 and gnorm > 0 and w_grads == 0
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", default="../dataset/LibriSpeech/train-clean-100")
    ap.add_argument("--noise-dir", default="../dataset/musan/noise")
    ap.add_argument("--checkpoint", default="../ckpt/deepvqe/model_240 2.tar")
    ap.add_argument("--whisper", default="openai/whisper-small")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-se", type=float, default=0.02)
    ap.add_argument("--lambda-feat", type=float, default=0.0)
    ap.add_argument("--out-dir", default="runs/exp2")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke_test(args, device)
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor, whisper, se, logmel, hybrid = build(args, device)
    ds = MixDataset(args.librispeech, args.noise_dir, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(processor), num_workers=4,
                        drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW([p for p in se.parameters() if p.requires_grad], lr=args.lr)

    print(f"Training Exp 2: steps={args.steps} bs={args.batch_size} lr={args.lr} "
          f"lambda_ce={args.lambda_ce} lambda_se={args.lambda_se} -> {out}")
    step, t0 = 0, time.time()
    ce_ema = se_ema = feat_ema = None
    log_path = out / "train_log.tsv"
    with open(log_path, "w") as logf:
        logf.write("step\tce\tse\tfeat\ttotal\tit_per_s\n")
    while step < args.steps:
        for batch in loader:
            lr_now = cosine_lr(step, args.steps, args.lr, args.warmup_steps, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr_now
            total, ce, se_loss, feat = loss_step(batch, whisper, se, logmel, hybrid,
                                                 device, args.lambda_ce, args.lambda_se,
                                                 args.lambda_feat)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(se.parameters(), 5.0)
            opt.step()
            ce_ema = ce.item() if ce_ema is None else 0.98 * ce_ema + 0.02 * ce.item()
            se_ema = se_loss.item() if se_ema is None else 0.98 * se_ema + 0.02 * se_loss.item()
            feat_ema = feat.item() if feat_ema is None else 0.98 * feat_ema + 0.02 * feat.item()
            step += 1
            if step % args.log_every == 0:
                ips = step / (time.time() - t0)
                print(f"step {step:5d} | CE {ce_ema:.4f} | HybridLoss {se_ema:.4f} "
                      f"| Feat {feat_ema:.4f} | total {total.item():.4f} | {ips:.2f} it/s")
                with open(log_path, "a") as logf:
                    logf.write(f"{step}\t{ce_ema:.5f}\t{se_ema:.5f}\t{feat_ema:.5f}\t{total.item():.5f}\t{ips:.3f}\n")
            if step % args.save_every == 0 or step >= args.steps:
                ckpt = out / f"deepvqe_exp2_step{step}.tar"
                torch.save({"model": {f"model.{k}": v for k, v in
                                       se.model.state_dict().items()},
                            "step": step}, ckpt)
                print(f"  saved {ckpt}")
            if step >= args.steps:
                break
    print(f"Done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
