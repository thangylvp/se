# Exp 5 result — add encoder feature-matching (λ_feat=2.0)

Whisper-small, test-clean, MUSAN, n=200, greedy. WER (%).

| model | −10 | −5 | 0 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| noisy | 40.99 | 13.45 | 6.76 | 4.63 | 3.87 | 3.51 |
| Exp 4 step9000 (CE+Hybrid) | 20.59 | 10.86 | 5.98 | 4.25 | 3.84 | 3.46 |
| Exp 5 step1500 (+λ_feat) | 29.89 | 12.36 | 7.09 | 4.58 | 3.96 | 3.30 |
| Exp 5 step3000 (+λ_feat) | 24.88 | 11.74 | 6.19 | 4.29 | 3.94 | 3.37 |

## Verdict: NEGATIVE — λ_feat (encoder feature-matching) hurts, mainly at low SNR.
- Worse than CE+HybridLoss at −10 (+4.3), −5 (+0.9), 0 (+0.2); ~neutral at 5/10/20.
- Matches the literature caution (Plantinga/Saddler/Kataria: feature/perceptual loss is unvalidated for WER). Likely the "match clean encoder features" objective conflicts with CE at low SNR — pushing toward an over-suppressed representation that reintroduces the very artifacts CE removed.
- A smaller λ_feat would only reduce the harm toward the Exp 2 baseline, not exceed it → λ_feat is not the lever.

## Standing best model: Exp 2 / Exp 4 (CE + small HybridLoss), ~step 3000.
