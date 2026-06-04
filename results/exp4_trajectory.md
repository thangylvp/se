# Exp 4 result — train longer (15k steps / 4.4 epochs + LR decay)

Whisper-small, test-clean, MUSAN, n=200, seed=1234. WER (%) per SNR across checkpoints.

| model | −10 | −5 | 0 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| clean (upper bound) | 3.30 | 3.30 | 3.30 | 3.30 | 3.30 | 3.30 |
| noisy (baseline) | 40.99 | 13.45 | 6.76 | 4.63 | 3.87 | 3.51 |
| model_240 (old SE) | 35.96 | 18.67 | 10.10 | 5.67 | 4.46 | 3.46 |
| step 3000 | 20.28 | 11.53 | 6.24 | 4.44 | 3.84 | 3.39 |
| step 6000 | 20.80 | 11.48 | 6.02 | 4.70 | 3.91 | 3.46 |
| step 9000 | 20.59 | 10.86 | 5.98 | 4.25 | 3.84 | 3.46 |
| step 12000 | 20.80 | 11.15 | 6.50 | 4.77 | 3.80 | 3.44 |
| step 15000 | 20.28 | 11.29 | 6.57 | 4.91 | 3.84 | 3.44 |

## Verdict: WER plateaus by step 3000 — more training does NOT help.
- All checkpoints 3k→15k are within eval noise (±~0.5%) of each other; no improving trend. CE EMA also flat (~1.28–1.30).
- At 0/5 dB it slightly *worsens* by 12k/15k → mild over-optimization with no WER payoff.
- The recipe converges in <1 epoch; the "train more" lever is exhausted.

## Interpretation
- High SNR (5/10/20): already at the clean floor (noisy ≈ clean), so little to gain — nothing to fix there.
- Gains are concentrated at low SNR: −10 dB ~50% relative WER cut (41→20), −5 ~16% (13.5→11), 0 ~11% (6.8→6).
- The clean floor (3.30%) is the bound at all SNRs but is physically unreachable at low SNR — −10/−5 dB destroys information no front-end can restore. Per-SNR achievable targets are well above 3.30%.
- To push moderate-SNR WER lower we need a DIFFERENT lever, not more steps: loss-weighting (lower λ_se / add λ_feat), or a larger/stronger ASR target (lowers the floor), or more SE capacity.
