"""The ASR-aware composite loss:

    L = lambda_ce   * CE(WhisperDecoder(enh), transcript)      # primary driver
      + lambda_se   * HybridLoss(enh, clean)                   # fidelity stabilizer
      + lambda_feat * L1(WhisperEnc(enh), WhisperEnc(clean))   # feature-match (off by default)

DeepVQE is trainable; Whisper is frozen — gradients flow *through* it into DeepVQE
via the differentiable log-mel. See exps.md for why lambda_ce dominates and
lambda_feat is kept at 0.
"""
import torch
import torch.nn.functional as F

# encoder hidden-state layers for the feature loss (early/mid; lit: Plantinga/
# Kataria found shallow layers transfer best). Indices into encoder hidden_states
# (0 = conv embedding, 1..12 = after each block).
FEAT_LAYERS = (2, 4, 6)


def loss_step(batch, whisper, se, logmel, hybrid, device,
              lambda_ce, lambda_se, lambda_feat=0.0):
    """Return (total_loss, ce_detached, se_detached, feat_detached) for one batch."""
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)
    labels = batch["labels"].to(device)

    enhanced = se(noisy)                    # (B, L) differentiable
    mel = logmel(enhanced)                  # (B, 80, 3000)
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
