# Headroom baseline — go/no-go (Step 1)

Date: 2026-06-03
Config: Whisper-small, LibriSpeech test-clean, MUSAN noise, n=200 utts, seed=1234.
Checkpoint: `ckpt/deepvqe/model_240 2.tar` (teammate's HybridLoss-trained DeepVQE).
Eval: corpus WER, Whisper EnglishTextNormalizer. Same clean+noise per utt across SNRs.

| SNR (dB) | noisy WER | enhanced WER | Δ(enh−noisy) |
|---:|---:|---:|---:|
| clean | 3.30% | — | — |
| −10 | 40.99% | 35.96% | −5.03% (SE helps) |
| −5 | 13.45% | 18.67% | +5.22% (SE hurts) |
| 0 | 6.76% | 10.10% | +3.34% (SE hurts) |
| 5 | 4.63% | 5.67% | +1.04% (SE hurts) |
| 10 | 3.87% | 4.46% | +0.59% (SE hurts) |
| 20 | 3.51% | 3.46% | −0.05% (neutral) |

## Findings
- **Headroom exists**: noise degrades Whisper WER strongly at low SNR (3.30% → 40.99% at −10). PoC is feasible.
- **Current SE actively HURTS WER in the −5…+10 dB band** — the user's complaint, quantified. Its artifacts cost more than the noise it removes once SNR is moderate.
- SE only helps at −10 dB (noise so dominant that removal wins despite artifacts) and is neutral at +20 dB.
- Whisper-small is robust at high SNR (≈clean at +20), so little headroom there.

## Target for ASR-aware fine-tuning
Turn the positive Δ (SE hurts) at −5/0/+5 dB into ≤ 0 (match or beat the noisy baseline),
ideally pulling enhanced WER toward clean. The **−5…+5 dB band is the sweet spot** to
demonstrate the effect. Success = enhanced WER ≤ noisy WER across the band, with the gap
to clean WER shrinking.

## Caveats
- n=200 (fine for go/no-go); use ≥500 for final reported numbers.
- Single noise corpus (MUSAN) and a fixed seed; broaden for robustness claims.
