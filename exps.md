# Experiments — ASR-aware Speech Enhancement (DeepVQE → frozen Whisper)

Goal of the whole line of work: fine-tune the DeepVQE SE front-end so it **lowers Whisper WER**, not just denoises. See `related_works.md` for the literature this is built on and `results/headroom_baseline.md` for the baseline numbers.

## Design constraint (2026-06-04): the SE model must be ASR-INDEPENDENT
We do not own, tune, pick, or decode-trick the deployment ASR — it is unknown. The SE is a standalone front-end. Implications:
- **No ASR-side levers**: no beam search / decoding tweaks, no "switch to a bigger Whisper" as a deployment target.
- Whisper-small stays a **fixed, frozen black box**, used only as (a) a training proxy signal and (b) one evaluator. Greedy decoding only.
- Training still needs *some* ASR signal (pure fidelity = HybridLoss hurt WER), so we train against Whisper as a **proxy** and treat "ASR-independence" as a **transfer property to verify**: an SE trained against Whisper should also lower WER on a *different, unseen* ASR. Signal ASR-specificity: decoder-CE (most Whisper-specific) > encoder feature-match (more generic acoustic/phonetic) > HybridLoss (ASR-agnostic but WER-harmful).
- **Pending validation (backlog Exp G): cross-ASR transfer** — eval our best SE-enhanced audio on a held-out ASR (evaluation only, not a training target) to confirm gains aren't Whisper-overfit.

## Evaluation protocol (fixed across all experiments — change it and you break comparability)
- **ASR**: frozen Whisper-small (`openai/whisper-small`), greedy/beam decode, English.
- **Eval set**: fixed `n=200` utterances from LibriSpeech **test-clean**, seed=1234; same clean utt + same noise clip reused across all SNRs (only SNR varies). Built by `data_utils.build_eval_set`.
- **Noise (eval)**: MUSAN `noise/`. ⚠️ Also report on a **non-MUSAN** noise set once available, to catch overfitting (lit: Plantinga distortion problem).
- **SNRs**: −10, −5, 0, 5, 10, 20 dB.
- **Metric**: corpus WER, Whisper `EnglishTextNormalizer` (`asr_whisper.WhisperASR.wer`).
- **Always report three references**: **clean WER (upper bound)**, **noisy WER (baseline to beat)**, and **current fidelity-SE WER** (`model_240`). A new model is a win only if enhanced WER ≤ noisy across the −5…+5 dB band without regressing the −10 dB case.
- Tooling: `eval_headroom.py` produces the table; reuse it per checkpoint.

---

## Exp 1 — Baseline characterization (DONE)
- **Main goal**: Establish whether the project is even feasible — is there WER headroom, and does the current fidelity-trained DeepVQE help or hurt ASR, as a function of SNR?
- **Change from previous**: N/A (first experiment). Three conditions evaluated: clean / noisy / noisy→`model_240` DeepVQE.
- **Why**: Before training anything, confirm (a) noise actually degrades Whisper WER (else nothing to fix) and (b) quantify the "SE doesn't help WER" complaint.
- **Results** (Whisper-small, n=200, MUSAN, seed=1234):

  | SNR (dB) | noisy WER | enhanced WER (model_240) | Δ(enh−noisy) |
  |---:|---:|---:|---:|
  | clean | 3.30% (upper bound) | — | — |
  | −10 | 40.99% | 35.96% | −5.03% (SE helps) |
  | −5 | 13.45% | 18.67% | +5.22% (SE hurts) |
  | 0 | 6.76% | 10.10% | +3.34% (SE hurts) |
  | 5 | 4.63% | 5.67% | +1.04% (SE hurts) |
  | 10 | 3.87% | 4.46% | +0.59% (SE hurts) |
  | 20 | 3.51% | 3.46% | −0.05% (neutral) |

- **Notes**: Headroom is large at low SNR. The current SE **actively hurts WER in the −5…+10 dB band** (artifacts cost more than the noise removed); only helps at −10 dB; neutral at +20. This is the textbook SE↔ASR misalignment (IRIS, Dissen). Confirms the target band: **−5…+5 dB**.
- **Next step**: First training experiment (Exp 2) — make enhanced WER ≤ noisy in the target band.

---

## Exp 2 — First training run: decoder-CE + small fidelity stabilizer (NEXT — run now)
- **Main goal**: First ASR-aware fine-tune. Turn the positive Δ at −5/0/+5 dB to **≤ 0** (enhanced WER ≤ noisy), without regressing the −10 dB case, and ideally beat `model_240` everywhere.
- **Change from Exp 1**: Exp 1 only *evaluated* the fidelity-trained checkpoint. Exp 2 **trains** DeepVQE (warm-started from `model_240`) with a **frozen Whisper-small decoder cross-entropy** loss as the primary driver, plus a **small HybridLoss** stabilizer:
  `L = λ_ce·CE(WhisperDec(enh), transcript) + λ_se·HybridLoss(enh, clean)`, with **λ_se ≈ λ_ce / 50** (Dissen ratio). **λ_feat = 0** (deferred to Exp 3).
- **Why**:
  - Dissen et al. (Interspeech 2024 / TASLP 2025) validated *exactly* this — decoder-CE through a frozen Whisper as the primary lever — and showed fidelity-only/off-the-shelf SE degrades WER (our Exp 1).
  - CE-alone is unstable (Dissen: "degenerate spectrogram transformations") → keep a small HybridLoss as stabilizer.
  - **λ_feat is unvalidated for WER** (no SE→WER paper showed it helps) → isolate the proven CE lever first; adding it now would confound attribution. One change at a time.
  - Warm-start (not from scratch): IRIS shows joint-from-random fails to converge; init from our pretrained SE converges fast.
- **Setup (planned)**:
  - Train data: LibriSpeech **train-clean-100** + MUSAN `noise/`, on-the-fly mixing. SNR sampled across −10…+20 dB but **weighted toward the −5…+5 dB failure band** (don't break the −10 dB regime).
  - Frozen Whisper-small; trainable DeepVQE; **differentiable torch log-mel** into Whisper (required — we enhance the signal, not the mel).
  - Held-out eval = the fixed Exp-1 eval set (test-clean, never seen in training).
- **Setup (actual)**: warm-start `model_240`; train-clean-100 + MUSAN `noise/`, on-the-fly mix, SNR pool `[-10,-5,-5,0,0,0,5,5,10,20]` (weighted to −5..+5); 3000 steps (~1 epoch), bs=8, lr=1e-4, AdamW, grad-clip 5.0, λ_ce=1.0, λ_se=0.05, differentiable torch log-mel. ~22 min on RTX 5090. Code: `train_exp2.py`, `diff_logmel.py`, `losses.py`. Smoke test (grad flow) passed before the run.
- **Results** (n=200, seed=1234; full table in `results/exp2_step3000.md`):

  | SNR (dB) | noisy | model_240 | **exp2** | Δ vs noisy | vs model_240 |
  |---:|---:|---:|---:|---:|---:|
  | clean | — | — | — | (clean 3.30%) | |
  | −10 | 40.99% | 35.96% | **21.25%** | −19.73 | −14.71 |
  | −5 | 13.45% | 18.67% | **10.84%** | −2.61 | −7.83 |
  | 0 | 6.76% | 10.10% | **6.24%** | −0.52 | −3.86 |
  | 5 | 4.63% | 5.67% | **4.41%** | −0.21 | −1.26 |
  | 10 | 3.87% | 4.46% | **3.87%** | 0.00 | −0.59 |
  | 20 | 3.51% | 3.46% | **3.32%** | −0.19 | −0.14 |

- **Notes / verdict — PoC SUCCESS**: enhanced WER is **≤ noisy at every SNR** and **beats fidelity-trained `model_240` at every SNR**. The −5..+10 dB band where the old SE *hurt* is now neutral-to-helping; biggest win at −10 dB; SNR 20 sits at the clean floor (no harm to easy audio). Training was stable (no CE collapse). HybridLoss rose 1.45→2.17 → enhanced drifted from clean fidelity (expected, CE-dominant); perceptual quality not yet measured. CE EMA 1.50→1.30.
- **Threats to validity**: **MUSAN was used for both train and eval** → the gain may be partly noise-set memorization (the #1 thing to rule out next). Whisper-small only; n=200 only.
- **Next step**: **Exp 3 — generalization to unseen noise** (rule out MUSAN overfitting before anything else). Then Exp 4 (λ_feat ablation), Exp 5 (λ_ce/λ_se & SNR sweep), confirm at n≥500.

---

## Exp 3 — Generalization to unseen noise (NEXT — proposed)
- **Main goal**: Confirm the Exp 2 win is real and not MUSAN-noise memorization, by evaluating the **same Exp 2 checkpoint** on a noise corpus it never saw in training.
- **Change from Exp 2**: no model/training change — only the **eval noise set** changes (MUSAN → a held-out non-MUSAN corpus, e.g. DEMAND / WHAM! / ESC-50). Build a parallel fixed eval set (same test-clean utts + same SNRs, new noise) so the only variable vs Exp 2's eval is the noise distribution.
- **Why**: Train and eval both used MUSAN in Exp 2; a reviewer's first objection is distribution leakage. If the WER gains hold on unseen noise, the PoC is robust; if they shrink, we quantify the overfitting and it motivates broadening the training noise. This is higher-value than adding loss terms (Exp 4) because it validates the result we already have.
- **Setup (planned)**: download a small unseen noise set (≤1–2 GB); add it to `data_utils.build_eval_set` as an alternate `--noise-dir`; run `eval_headroom.py` on `deepvqe_exp2_step3000.tar` and (reference) `model_240` + noisy + clean. Optionally also hold out a MUSAN category for a cheap in-corpus shift check.
- **Results**: _TBD_
- **Notes**: _TBD_
- **Next step**: _TBD_ — if it generalizes → Exp 4 (λ_feat ablation) / scale up; if not → broaden training noise and retrain.

---

## Exp 4 — Train longer (4.4 epochs + LR decay) (RUNNING)
- **Main goal**: Push WER down toward the clean floor (3.30%), especially in the 0/−5/−10 dB bands where headroom remains. Exp 2's gains over *noisy* were small at moderate SNR; test whether that's just under-training.
- **Change from Exp 2**: identical recipe, but **train 15000 steps (~4.4 epochs of train-clean-100) instead of 3000 (~0.84 epoch)**, and add **linear warmup (300) + cosine LR decay** (1e-4 → 1e-5). Same λ_ce=1.0, λ_se=0.05, bs=8, same MUSAN noise + SNR pool. (Noise diversity deliberately deferred per decision 2026-06-04.)
- **Why**: Exp 2 ran <1 epoch and CE was still falling when it stopped — the reported numbers are from an under-trained model. More steps + LR decay is the highest-ROI lever to converge toward clean. One variable changed (training length/schedule) so the effect is attributable.
- **Setup**: `runs/exp4/`, checkpoints every 3000 steps (3k/6k/9k/12k/15k) to plot the WER-vs-steps trajectory. ~1.9 h on RTX 5090.
- **Results** (n=200; full table `results/exp4_trajectory.md`): WER **plateaus by step 3000** and stays flat through 15000 (all within ±~0.5% eval noise); CE EMA flat ~1.28–1.30; at 0/5 dB it slightly *worsens* by 12k/15k.

  | model | −10 | −5 | 0 | 5 | 10 | 20 |
  |---|---:|---:|---:|---:|---:|---:|
  | noisy | 40.99 | 13.45 | 6.76 | 4.63 | 3.87 | 3.51 |
  | step 3000 | 20.28 | 11.53 | 6.24 | 4.44 | 3.84 | 3.39 |
  | step 9000 | 20.59 | 10.86 | 5.98 | 4.25 | 3.84 | 3.46 |
  | step 15000 | 20.28 | 11.29 | 6.57 | 4.91 | 3.84 | 3.44 |

- **Notes / verdict — NEGATIVE: "train more" is exhausted.** The recipe converges in <1 epoch; extra epochs give no WER gain (mild over-opt at 0/5 dB). Gains are real but concentrated at low SNR (−10 ~50% rel.). High SNR is already at the clean floor (nothing to fix). The 3.30% clean bound is physically unreachable at −10/−5 dB (information destroyed), so per-SNR targets are higher.
- **Next step**: change a different lever, not steps. **Exp 5 — loss-weighting**: lower λ_se (let CE drive harder; currently SE-term ≈ 1/13 of CE, looser than Dissen's 1/50) and/or add λ_feat. Cheap (~22 min). If that plateaus too → bigger/stronger ASR target (lowers the floor) or more SE capacity. Generalization (unseen noise) still pending.

---

## Exp 5 — add encoder feature-matching (λ_feat) (RUNNING)
- **Main goal**: Break the Exp 2/4 plateau by adding a *new* gradient signal (Whisper encoder feature-matching), and prefer it because it's more ASR-transfer-friendly than decoder-CE.
- **Change from Exp 2**: add `λ_feat · L1(WhisperEnc(enh), WhisperEnc(clean))` over **early/mid encoder layers (hidden_states 2,4,6)**, λ_feat=2.0; keep λ_ce=1.0, λ_se=0.05. Warm-start `model_240`, 3000 steps (plateau is fast). One change vs Exp 2.
- **Why**: Exp 4 showed CE+HybridLoss plateaus and CE already dominates — lowering λ_se won't help, so we need information CE doesn't provide. Feature-matching adds it; lit (Plantinga/Kataria) says early/mid layers, L1. Bonus: matching the encoder *representation* should transfer across ASRs better than the Whisper-decoder CE.
- **Setup**: `runs/exp5/`, save 1500/3000. Logs CE/HybridLoss/Feat separately.
- **Results** (n=200, greedy; full `results/exp5_feat.md`): **NEGATIVE — λ_feat hurts**, mainly at low SNR (−10: 24.88 vs Exp4 20.59; −5: 11.74 vs 10.86; 0: 6.19 vs 5.98; ~neutral 5/10/20).
- **Notes / verdict**: Matches the lit caution that feature/perceptual loss is unvalidated for WER. Likely matching *clean* encoder features conflicts with CE at low SNR (pushes toward over-suppression → reintroduces artifacts). Smaller λ_feat would only approach the Exp 2 baseline from below, not beat it → **λ_feat is not the lever**.
- **Next step**: drop λ_feat. **Standing best = Exp 2/4 (CE + small HybridLoss).** Remaining ASR-independent SE-side lever to try: **Exp 6 — SNR-targeted training** (weight −10/−5/0 harder, where the WER headroom is). Then Exp G (cross-ASR transfer) to validate the best model.

---

## Planned backlog (one change each; promote to a numbered Exp when run)
- **Exp G — cross-ASR transfer (validates ASR-independence)**: eval best SE on a held-out ASR (e.g. wav2vec2 / a different Whisper size), evaluation-only. Confirms gains transfer to an unseen recognizer.
- **Exp 3 — add λ_feat (ablation)**: + encoder feature-matching `MSE/L1(WhisperEnc(enh), WhisperEnc(clean))`, early/mid layers, per-layer balancing, tiny weight. Tests whether the speculative feature term adds anything over Exp 2.
- **Exp 4 — SNR sampling / loss-ratio sweep**: tune λ_ce:λ_se and the training SNR distribution.
- **Exp 5 — generalization**: eval on a **non-MUSAN** noise set (+ optionally a second ASR size) to check overfitting.
- **Exp 6 — scale**: train-clean-360 / more steps, larger eval (n≥500).
