# Exp 2 result — CE + small HybridLoss, step 3000

Whisper-small, test-clean, MUSAN noise, n=200, seed=1234. Checkpoint: `runs/exp2/deepvqe_exp2_step3000.tar`.
Training: warm-start `model_240`, 3000 steps (~1 epoch train-clean-100), bs=8, lr=1e-4, λ_ce=1.0, λ_se=0.05, SNR weighted toward −5..+5.

| SNR (dB) | clean | noisy | model_240 | **exp2 step3000** | exp2 Δ vs noisy | exp2 vs model_240 |
|---:|---:|---:|---:|---:|---:|---:|
| clean | 3.30% | — | — | — | — | — |
| −10 | | 40.99% | 35.96% | **21.25%** | −19.73 | −14.71 |
| −5 | | 13.45% | 18.67% | **10.84%** | −2.61 | −7.83 |
| 0 | | 6.76% | 10.10% | **6.24%** | −0.52 | −3.86 |
| 5 | | 4.63% | 5.67% | **4.41%** | −0.21 | −1.26 |
| 10 | | 3.87% | 4.46% | **3.87%** | 0.00 | −0.59 |
| 20 | | 3.51% | 3.46% | **3.32%** | −0.19 | −0.14 |

## Verdict: PoC success
- The fine-tuned SE is **≤ noisy at every SNR** and **beats the fidelity-trained `model_240` at every SNR**.
- The −5..+10 dB band where the old SE *hurt* (+1 to +5 WER points) is now **neutral-to-helping**.
- At SNR 20 (near clean), 3.32% ≈ clean 3.30% → no damage to easy audio.
- Largest gain at −10 dB (40.99→21.25), where there is most headroom.

## Caveats (drive the next experiments)
- **MUSAN train AND eval** → the win could be partly MUSAN-noise memorization. Must validate on an unseen noise corpus (Exp 3).
- Whisper-small only; n=200 only (confirm at n≥500).
- HybridLoss rose during training (1.45→2.17) → enhanced output drifted from clean fidelity. Expected (CE dominates), but perceptual quality (DNSMOS/PESQ) was not measured — the model is now optimized for Whisper, may sound worse to humans.
