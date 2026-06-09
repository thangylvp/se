# Related Works — ASR-aware Speech Enhancement

Literature behind our goal: fine-tune a DeepVQE speech-enhancement (SE) front-end so it **lowers WER of a frozen Whisper**, rather than merely denoising. Curated to top venues (Interspeech, ICASSP, WASPAA, TASLP), read paper-by-paper by sub-agents, fact-checked against the source PDFs (several auto-summaries were found to be fabricated and discarded).

Our planned recipe (for reference): trainable DeepVQE, **frozen Whisper-small**, loss
`L = λ_se·HybridLoss(enh, clean) + λ_feat·MSE(WhisperEnc(enh), WhisperEnc(clean)) + λ_ce·CE(WhisperDec(enh), transcript)`,
LibriSpeech clean + MUSAN noise mixed on-the-fly at SNR −10…+20 dB. Measured baseline: the current fidelity-only DeepVQE **hurts** Whisper WER in the −5…+10 dB band.

---

## TL;DR — cross-cutting findings (read this first)

1. **The problem is universal and named.** Every SE→ASR paper states it outright: single-channel SE trained for signal fidelity introduces artifacts that *hurt* recognition. IRIS shows naive SE raising CHiME-4 test-real WER from 4.47 → 12.11; Dissen shows Demucs/SGMSE+ *degrading* Whisper WER. Our −5…+10 dB regression is the textbook symptom, not a bug in our checkpoint.

2. **The closest paper to us (Dissen et al., Interspeech 2024 / TASLP 2025) validates the core architecture:** a trainable front-end before a **frozen Whisper**, trained by **decoder cross-entropy backpropagated through frozen Whisper**. It works and beats fidelity-only enhancers. → **λ_ce is the primary, validated driver.**

3. **CE-through-frozen-ASR alone is unstable.** Dissen report WER "suddenly degrad[ing] due to large gradients … or the model learning degenerate spectrogram transformations." Their fix: a **small reconstruction loss (~1/50 of the CE weight)** as a stabilizer. → **Keep λ_se, but as a small regularizer — not the dominant term.** This refines our earlier "anchor on fidelity" framing: fidelity should be subordinate to CE, not co-equal.

4. **The feature-matching term (λ_feat) is the least-validated of our three.** No SE→**WER** paper used an encoder-feature-MSE term and showed it lowers WER (Dissen get strong gains with CE + L1 only). The deep-feature-loss papers (Germain, Saddler, Kataria) optimized/measured **perceptual quality, never WER**, and Saddler/Kataria explicitly warn perceptual gains ≠ WER gains. IRIS supports *feature-space bridging* conceptually (frozen WavLM as the SE↔ASR bridge). → **Treat λ_feat as an optional, to-be-ablated regularizer; judge it on WER, not PESQ.**

5. **If we use λ_feat, the deep-feature literature is unanimous on how:**
   - **Match early/mid layers, not the deepest** (Plantinga: deep/recurrent/full-context layers *hurt*; Kataria taps first 4–8 blocks; Germain uses first 6 of 15). For Whisper-small (12 encoder blocks) → match early-to-mid blocks, possibly several summed, **not** only the last hidden state.
   - **Use L1, not MSE**, on activations (Germain, Saddler, Plantinga-Voicebank all prefer L1).
   - **Per-layer weight balancing** (inverse of each layer's loss magnitude after a short warm-up) so no layer dominates.
   - **One extractor, not an ensemble** — Kataria found a 6-model ensemble *underperformed* a single well-chosen feature loss; MTL auto-weighting lost to hand-tuned weights.
   - Feature term needs a **tiny weight** relative to the reconstruction term (Kataria: λ_feat ≈ 100× smaller than λ_L1).

6. **Initialize from the pretrained fidelity DeepVQE; do not train the combined objective from scratch.** IRIS: random-init joint training "could not converge … the deep architecture … disturb[s] the gradient back-propagation"; once each module is pretrained, joint fine-tune converges in ~1 epoch. We already have the fidelity-trained DeepVQE — use it as the warm start, then ramp in the ASR terms.

7. **Don't trust signal metrics as a WER proxy.** Subramanian measured PESQ/STOI correlate only ~0.77–0.78 with word accuracy, and configs can win on WER while *losing* on LLR/SRMR. → Track real WER every eval (we already do, via `eval_headroom.py`).

8. **Guard against overfitting to MUSAN.** Plantinga's "distortion-independent training" works by exposing the *recognizer* to enhancer-style distortions and keeping it from being gamed. Since Whisper stays frozen, our analogs are: (a) keep Whisper strictly frozen (it can't be gamed), (b) **evaluate on noise outside MUSAN** to catch overfitting, (c) lean on the frozen-feature/CE signal rather than letting the SE chase fidelity on one noise set.

### Implications — concrete changes to our locked plan

| Term | Literature verdict | Action |
|---|---|---|
| **λ_ce** (decoder CE, frozen Whisper) | Primary, validated driver (Dissen). | Make it the **dominant** term. Expect instability alone → needs a stabilizer. |
| **λ_se** (HybridLoss fidelity) | Necessary **stabilizer**, but must be small (~1/50 of CE in Dissen). | Keep, but **down-weight** well below λ_ce. Reframe from "anchor" to "regularizer." |
| **λ_feat** (encoder feature MSE) | Unvalidated for WER; novel in our recipe. | **Ablate explicitly.** If kept: early/mid Whisper-encoder layers, **L1**, per-layer balancing, tiny weight, single extractor. |

Additional musts:
- **Differentiable log-mel into Whisper** is a real requirement for us (Dissen sidestep it by outputting mel directly; we enhance the *signal*, so we must implement Whisper's mel filterbank + log as torch ops). Confirmed pitfall, not optional.
- **Warm-start from `model_240 2.tar`**; schedule: stabilize with λ_se, then ramp λ_ce (and optionally λ_feat).
- **Whisper size:** small (244M) is in range; Dissen validated base (74M) and large-v2. Their "train-against-base, deploy-on-large" result ("gradients of the base model are easier to handle") hints a smaller frozen model gives cleaner front-end gradients — a useful fallback if Whisper-small CE gradients prove noisy.
- The **gradient-projection/rescaling mechanism** sometimes attributed to Subramanian (2019) is **not in that paper** — do not cite it for that. For genuine multi-objective conflict handling, look to PCGrad-style methods.

---

## Papers

### 1. Dissen, Yonash, Cohen, Keshet — Front-End Adaptation for a Frozen Whisper *(closest match)*
**"Enhanced ASR Robustness to Packet Loss with a Front-End Adaptation Network," Interspeech 2024** (pp. 5008–5012, DOI 10.21437/Interspeech.2024-306; arXiv 2406.18928). Extended journal: **TASLP 2025**, vol. 33 pp. 2175–2188 (adds the noise/reverb experiments). **Top venue: yes** (both).

- **Architecture:** a 7.5M-param fully-conv **U-net** operating **mel-to-mel** — it sits between the noisy log-mel and Whisper's encoder and outputs a corrected mel. Whisper is **fully frozen** (full fine-tune and LoRA tried only as baselines — both hurt out-of-domain WER).
- **Loss:** `λ·L_CE(WhisperDecoder, transcript) + (1−λ)·L1(clean_mel, adapted_mel)`. **CE is the primary driver; L1 is a small stabilizer.** Journal: "weighting the L1 loss at approximately 1/50th of the ASR loss provided the best tradeoff." (The conference text states a conflicting 0.9/0.1 split; trust the journal's CE-dominant value.) **No feature/distillation loss.** Differentiable log-mel is a non-issue for them because the front-end outputs mel directly.
- **Stability finding (key for us):** "training with ASR loss alone sometimes resulted in unstable convergence … the model learning degenerate spectrogram transformations that momentarily reduce loss but ultimately harm ASR." The L1 term fixes it; too much L1 also hurts.
- **Data/models:** trained only on LibriSpeech-960h with on-the-fly corruption (packet loss; +white/babble noise at SNR 2–8 dB and RIR reverb in the journal). Whisper **base (74M)** and **large-v2 (1550M)**; **small not separately reported**.
- **Results:** PLC blind set base 24.0→18.1, large-v2 15.4→14.2 WER; multilingual zero-shot gains up to ~2.5× at high corruption; combined RIR+noise+PL 66%→42%. Full fine-tuning catastrophically forgets (Spanish 57→89). "Train-against-base, deploy-on-large" beat training directly on large ("gradients of the base model are easier to handle").
- **Relevance:** validates **λ_ce as dominant + λ_se as small stabilizer**; **λ_feat unvalidated** (they never used it); flags **differentiable log-mel** as our extra requirement; supports frozen Whisper and a small front-end.
- Quotes: *"we use the frozen transformer ASR loss function to update the weights of the adaptation network."* · *"training with ASR loss alone sometimes resulted in unstable convergence … To mitigate this, we introduced a secondary loss term: an L1 loss."* · *"applying Demucs speech enhancement degrades performance … [L1-only] performing worse than the baseline Whisper model in higher SNRs."*

### 2. Chang, Maekaku, Fujita, Watanabe — IRIS (SE + SSL + ASR)
**"End-to-End Integration of Speech Recognition, Speech Enhancement, and Self-Supervised Learning Representation," Interspeech 2022** (pp. 3819–3823, DOI 10.21437/Interspeech.2022-10839; arXiv 2204.00540). **Top venue: yes.**

- **Method:** pipeline `ASR(SSLR(SE(Y)))` — Conv-TasNet SE → **frozen WavLM-large** (the SE↔ASR bridge) → Transformer joint CTC/attention ASR. WavLM frozen; SE and ASR jointly fine-tuned after separate pretraining.
- **Loss:** multi-task **SI-SNR (signal-level SE) + joint CTC/attention (ASR)**. Explicit λ weights **not reported**.
- **Key results (CHiME-4, 1-ch, WER%):** WavLM no-SE 4.47 test-real; naive SE concat **12.11** (SE *hurts*); fine-tune SE-only 4.90; **joint SE+ASR fine-tune 3.64** (SOTA, beats no-SE). SE helps *only after* joint fine-tuning.
- **Pitfalls:** "monaural SE … produce distortions which deteriorate ASR"; random-init joint training "could not converge … deep architecture … disturb[s] gradient back-propagation" → **pretrain modules, then fine-tune** (converges in ~1 epoch).
- **Relevance:** strongest support for **feature-space bridging** (frozen robust representation) over waveform fidelity; supports **warm-starting from our pretrained DeepVQE**; warns that the recognition gradient must actually reach the SE module (frozen-SE + feature-loss-only would underperform), and that deep frozen stacks can choke gradients.
- Quotes: *"speech enhancement models do not necessarily improve the ASR performance … the training objectives … are not very well aligned."* · *"it is critical to fine-tune both models jointly to eliminate the mismatch."* · *"training the IRIS model from random initialization could not converge to a good point."*

### 3. Plantinga, Bagchi, Fosler-Lussier — Perceptual / Mimic Loss
**"Perceptual Loss with Recognition Model for Single-Channel Enhancement and Robust ASR," arXiv 2112.06068 (Dec 2021).** **Venue: arXiv preprint only — NOT a top venue.** Kept because it is the closest conceptual predecessor; its core idea was published at a top venue earlier: **Bagchi, Plantinga, Stiff, Fosler-Lussier, "Spectral feature mapping with mimic loss for robust speech recognition," ICASSP 2018** (top venue).

- **Method (mimic/perceptual loss):** a **frozen** pretrained acoustic (recognition) model encodes both clean and enhanced features; loss = distance between the two (**L2** on CHiME-2, **L1** on Voicebank) at **shallow/small-context layers** — deep/recurrent/full-utterance layers *hurt*; **phoneme targets beat word-piece**; best ASR model ≠ best perceptual model. Combined as `spectral loss + α·perceptual`, α set so terms match in magnitude.
- **Distortion-independent training:** train the recognizer **separately** from the enhancer with **extra noise added**, so it adapts to enhancer artifacts and the enhancer can't game it. Beat joint training (WER 3.25 vs 3.77).
- **Results:** **2.80% WER on Voicebank+DEMAND** (frozen enhancer + L1 perceptual + distortion-independent recognizer); clean 2.29, noisy 3.43, joint-trained 3.77. Helped even an unadapted recognizer (3.34 vs 3.58).
- **Relevance:** the closest analog of our **λ_feat** → match **early/mid encoder layers, L1, magnitude-balanced**; keep Whisper frozen (the anti-gaming mechanism); its anti-overfitting lesson = **evaluate on non-MUSAN noise**. Caution: their best system used **no decoder-objective gradient into the enhancer** — only frozen-feature loss — so our λ_ce is a different (joint-style) bet that they'd predict could overfit; weight it carefully.
- Quotes: *"Single-channel speech enhancement approaches do not always improve automatic recognition rates … because they can introduce distortions unhelpful for recognition."* · *"the enhancement model can overfit to the training data, weakening the recognition model in the presence of unseen noise."*

### 4. Subramanian et al. — SE Using End-to-End ASR Objectives
**"Speech Enhancement Using End-to-End Speech Recognition Objectives," WASPAA 2019** (pp. 234–238, DOI 10.1109/WASPAA.2019.8937250). **Top venue: yes** (respected IEEE SPS workshop; specialist top-tier).

- **Method:** multichannel WPE→MVDR beamformer→attention ASR, **trained jointly with ASR cross-entropy ONLY** (no signal-level loss; ASR not frozen). Discrete WPE filter-order trained via REINFORCE.
- **No parallel clean data needed** (trains from ASR loss + transcripts alone).
- **Signal-metric↔WER correlations (transferable):** PESQ ≈ 0.78, STOI ≈ 0.77, FWSegSNR ≈ 0.71 with word accuracy; CD/LLR ≈ −0.57, SRMR ≈ 0.48 (weaker).
- **Results:** REVERB real near/far 8.7/11.8 WER, DIRHA 29.1 — large gains over no-processing and a conventional pipeline.
- **Relevance:** supports that an ASR-objective-trained front-end beats fidelity-only SE on WER; warns **signal metrics are imperfect WER proxies**. **Correction:** this paper has **no gradient-projection/rescaling** mechanism (a common mis-attribution) and never combines an SE loss with the ASR loss — don't cite it for either.
- Quotes: *"we can jointly optimize speech enhancement and ASR only with ASR error minimization criteria."* · *"PESQ, FWSegSNR and STOI have high correlation with WAR."*

### 5. Germain, Chen, Koltun — Speech Denoising with Deep Feature Losses
**"Speech Denoising with Deep Feature Losses," Interspeech 2019** (pp. 2723–2727, DOI 10.21437/Interspeech.2019-1924; arXiv 1806.10522). **Top venue: yes.**

- **Method:** time-domain context-aggregation denoiser; loss = **weighted L1 over the first 6 layers** of a **frozen audio-classifier** (DCASE scene/tag net — *not* ASR). Per-layer weights = inverse of each layer's L1 magnitude after a 10-epoch warm-up.
- **Results:** beats Wiener/SEGAN/WaveNet on SIG/BAK/OVL; ablation L2 3.07 → L1 3.11 → **feature 3.22**; perceptual A/B preference up to 83–96%, **largest on the hardest/most-intrusive noise**. No WER eval.
- **Relevance:** supports feature-matching helping **most at low SNR** (our failure band); gives the **per-layer auto-balancing** trick and **early-layer / L1** choices for λ_feat. Caution: classifier features, perceptual metric only — not WER.
- Quotes: *"simple training losses (e.g., L¹) led to noticeably degraded output quality at lower SNRs … improperly process low-energy speech information of perceptual importance."* · *"weights … set to the inverse of the relative values of ‖Φ^m(s)−Φ^m(g(x))‖₁ after 10 training epochs."*

### 6. Saddler, Francl, et al. — Speech Denoising with Auditory Models
**"Speech Denoising with Auditory Models," Interspeech 2021** (pp. 2681–2685, DOI 10.21437/Interspeech.2021-1973; arXiv 2011.10706). **Top venue: yes.**

- **Method:** Wave-U-Net trained to match **multi-layer L1** features of a **frozen** auditory recognition CNN (word-recognizer / AudioSet net) with a cochleagram front-end.
- **Key finding (caution):** a **plain 1-layer cochleagram L1** matched the best deep-feature model in rated naturalness; **random (untrained) features gave no benefit**. Objective SDR disagreed with perceptual ratings. **No ASR/WER eval.**
- **Relevance:** corroborates frozen-recognition-net feature matching + **L1 + per-layer balancing**; but a strong caution that **deep features may give no unique benefit over a cheap mel/spectral distance** → budget a baseline "Whisper-feature loss vs plain mel-L1" ablation. Perceptual gains ≠ WER.
- Quotes: *"the transform trained with this 'cochlear' loss performed just as well as our best model trained with deep feature losses."* · *"no evidence that deep features provide a unique benefit for denoising."*

### 7. Kataria, Villalba, Dehak — PERL (ensemble perceptual loss)
**"Perceptual Loss based Speech Denoising with an Ensemble of Audio Pattern Recognition and Self-Supervised Models," ICASSP 2021** (pp. 7118–7122; arXiv 2010.11860). **Top venue: yes.**

- **Method:** Conformer denoiser, base **STFT-domain L1** + per-model deep-feature losses from 6 **frozen** nets (AudioSet-PANN, DeepSpeech2-ASR, RawNet2 speaker, emotion, PASE+, wav2vec2), each tapped at **early/mid layers**. Weights greedily hand-tuned.
- **Findings:** **single best feature loss (acoustic-event/PANN) + L1 beat the full ensemble**; PASE+ #2, ASR acoustic model only #3. Feature weights ≈100× smaller than λ_L1. "MTL methods cannot outperform greedy hand-tuning."
- **Data/eval:** VCTK-DEMAND, PESQ/CSIG/CBAK/COVL/STOI; **no WER**. Best PERL-AE: PESQ 3.17 vs 3.01 baseline.
- **Relevance:** **one extractor, not an ensemble**; **early/mid layers**; **keep reconstruction loss dominant, feature term tiny**; ASR-model features weren't even the best *perceptually* → reinforces judging λ_feat on **WER** and treating it as a regularizer.
- Quotes: *"acoustic event and self-supervised model PASE+ to be most effective."* · *"state-of-the-art MTL methods cannot outperform greedy hand-tuning based weight selection."* · *"The auxiliary networks are frozen and only help constrain the output space."*

---

## Venue summary

| # | Short name | Venue | Year | Top venue? | Closeness to us |
|---|---|---|---|---|---|
| 1 | Dissen et al. (front-end + frozen Whisper) | Interspeech / TASLP | 2024 / 2025 | ✅ | ★★★★★ |
| 2 | IRIS (Chang et al.) | Interspeech | 2022 | ✅ | ★★★★ |
| 3 | Plantinga (perceptual/mimic loss) | arXiv (orig. ICASSP 2018) | 2021 | ⚠️ preprint (origin top) | ★★★★ |
| 4 | Subramanian (E2E ASR objective) | WASPAA | 2019 | ✅ | ★★★ |
| 5 | Germain (deep feature losses) | Interspeech | 2019 | ✅ | ★★★ |
| 6 | Saddler (auditory models) | Interspeech | 2021 | ✅ | ★★ |
| 7 | Kataria (PERL ensemble) | ICASSP | 2021 | ✅ | ★★ |

## What is novel in our work (relative to this literature)
- **Decoder-CE through a frozen Whisper to train a *waveform/spectral* SE model** (Dissen do mel-to-mel; we enhance the signal → we must add a differentiable log-mel). 
- **Combining decoder-CE with an encoder-feature-matching term** for a frozen Whisper — λ_feat is unproven for WER; our ablation would be a contribution either way.
- Demonstrating the **−5…+10 dB "SE hurts WER" band** explicitly and closing it (our headroom baseline already quantifies it).
