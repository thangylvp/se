# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a **DeepVQE speech-enhancement (SE) front-end so it lowers a downstream ASR's WER**, not just denoises. A teammate's DeepVQE (trained on signal-fidelity loss) denoises well but *hurts* WER when used as an ASR pre-processor; we fix that by fine-tuning it with a **frozen-Whisper ASR loss**. This `dn_asr/` folder is the only thing committed to git — it is meant to be self-sufficient given the setup steps below.

## Repository layout (this folder)

Code (committed):
- `deepvqe.py` — `DeepVQE` model (vendored, verified identical to the teammate's arch).
- `se_model.py` — `STFTWrapper(DeepVQE())` (waveform↔spectrogram) + `load_se_model()` checkpoint loader.
- `diff_logmel.py` — differentiable, pure-torch Whisper log-mel (required for training; see Architecture).
- `losses.py` — `HybridLoss` (vendored from the teammate's `SEtrain`).
- `train_exp2.py` — training loop (ASR-aware fine-tune). `--smoke` for a gradient-flow check.
- `asr_whisper.py` — frozen Whisper-small wrapper for transcription + WER (evaluation).
- `data_utils.py` — LibriSpeech listing, SNR mixing, deterministic eval-set builder.
- `eval_headroom.py` — single-checkpoint WER table (clean/noisy/enhanced × SNRs).
- `eval_traj.py` — multi-checkpoint WER eval with ONE Whisper load (fast comparisons).
- `infer.py` — standalone denoising inference (file or directory).
- `requirements.txt` — conda+pip setup.

Docs (committed): `exps.md` (experiment tracker — read first), `related_works.md` (literature behind the loss design), `report_wer.md` (external-facing WER report), `README.md`, `results/` (saved WER tables).

NOT committed (recreate per "Setup on a new machine"): `.env`, `runs/` (checkpoints), `out/`, `__pycache__/`, and the sibling `../dataset`, `../ckpt`, `../aecdns`, `../deepvqe`, `../SEtrain`.

## External dependencies (siblings of this folder — NOT in git)

Scripts default to relative paths `../ckpt/...` and `../dataset/...`, i.e. they assume this folder sits next to:

```
<parent>/
├── dn_asr/          # this git repo
├── ckpt/deepvqe/    # the DeepVQE checkpoint (from teammate)
├── dataset/         # LibriSpeech + MUSAN (downloaded)
├── aecdns/          # teammate's repo (reference only; arch + HybridLoss already vendored here)
├── deepvqe/         # original DeepVQE arch (reference only; vendored here)
└── SEtrain/         # source of HybridLoss (reference only; vendored here)
```

`deepvqe.py` and `losses.py` are vendored copies, so `../deepvqe`, `../aecdns`, `../SEtrain` are **reference-only** — not needed to run anything here. The two things that ARE needed and are not in git: the **checkpoint** and the **datasets**. All script paths are also overridable via CLI flags (`--checkpoint`, `--librispeech`, `--noise-dir`), so you can point elsewhere instead of recreating the sibling layout.

## Setup on a new machine

```bash
# 1) Python env (conda + pip)
conda create -n dn_asr python=3.11 -y && conda activate dn_asr
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # GPU (Blackwell/5090: cu128)
pip install -r requirements.txt
# (On the original machine the working env is the conda env `torch28`.)

# 2) HF token (NOT committed) — needed to download Whisper
echo "HF_TOKEN=hf_xxx" > .env

# 3) Datasets -> ../dataset  (LibriSpeech = OpenSLR-12, MUSAN = OpenSLR-17, both CC BY 4.0)
mkdir -p ../dataset && cd ../dataset
wget https://www.openslr.org/resources/12/test-clean.tar.gz       && tar xzf test-clean.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz  && tar xzf train-clean-100.tar.gz   # ~6 GB
wget https://www.openslr.org/resources/17/musan.tar.gz            && tar xzf musan.tar.gz musan/noise  # extract only noise/
cd ../dn_asr
# yields ../dataset/LibriSpeech/{test-clean,train-clean-100} and ../dataset/musan/noise (930 files)

# 4) Checkpoint -> ../ckpt/deepvqe/  (obtain from teammate; ~90 MB)
#    expected: "../ckpt/deepvqe/model_240 2.tar"
```

Then run things as `python <script>` (prefix `set -a; source .env; set +a` for anything that loads Whisper). On the original machine, use the existing env: `/home/thang/miniconda3/envs/torch28/bin/python`.

## Common commands

```bash
set -a; source .env; set +a   # load HF_TOKEN

# Denoise a file/folder with any checkpoint
python infer.py --checkpoint "../ckpt/deepvqe/model_240 2.tar" --input <wav_or_dir> --output-dir out

# Gradient-flow smoke test — ALWAYS run before a real training run
python train_exp2.py --smoke --device 0

# Train (ASR-aware fine-tune). Converges fast; ~3000 steps is enough (plateaus after).
python train_exp2.py --steps 3000 --batch-size 8 --lr 1e-4 \
    --lambda-ce 1.0 --lambda-se 0.05 --out-dir runs/expN --device 0

# Eval one checkpoint (full clean/noisy/enhanced table)
python eval_headroom.py --librispeech ../dataset/LibriSpeech/test-clean \
    --noise-dir ../dataset/musan/noise --checkpoint <ckpt.tar> --n-utts 200 --device 0

# Eval many checkpoints in one go (one Whisper load — use this for comparisons)
python eval_traj.py --checkpoints <ckptA> <ckptB> ... --n-utts 200 --device 0
```

GPU note: a training run uses ~27–30 GB of a 32 GB card — **do not run eval while training** on the same GPU (OOM); eval after the run.

## Architecture / how it fits together

Inference and training share the SE stack:
- **`DeepVQE`** (`deepvqe.py`): complex-spectrogram U-Net. Input `(B,F=257,T,2)` → 5 encoder blocks → GRU bottleneck → 5 decoder blocks → CCM complex mask.
- **`STFTWrapper`** (`se_model.py`): waveform→STFT→DeepVQE→iSTFT→waveform. STFT params are **fixed and must match everywhere: 16 kHz, n_fft=512, hop=256, win=512, periodic Hann**. `load_se_model()` strips the `model.` key prefix (checkpoints are saved from the wrapper, so keys look like `model.enblock1...`).
- **`DiffLogMel`** (`diff_logmel.py`): pure-torch Whisper log-mel (n_fft=400, hop=160, 80 mels, 30 s/3000 frames), verified to match the HF processor exactly. **Required** — the stock `WhisperFeatureExtractor` is numpy and detaches, which would silently zero the ASR gradient.
- **`train_exp2.py`**: DeepVQE is trainable (warm-started from `model_240`); **Whisper is frozen** (`eval()` + `requires_grad_(False)`) — gradients still flow *through* it into DeepVQE via the differentiable log-mel. `MixDataset` mixes clean+noise on the fly at a sampled SNR. Loss:
  `L = λ_ce·CE(WhisperDecoder(enh), transcript) + λ_se·HybridLoss(enh, clean) + λ_feat·L1(WhisperEnc(enh), WhisperEnc(clean))`.
- Checkpoints are saved as `{"model": {"model.<k>": v}}` so they load uniformly via `load_se_model`/`infer.py`.

## Loss design — settled; read before changing the recipe

- **λ_ce (decoder CE through frozen Whisper) is the primary, validated driver** (Dissen et al., Interspeech 2024). Already dominates (~12:1 over the SE term).
- **λ_se (HybridLoss) is a small stabilizer** (~0.05), not the anchor — CE-alone can go unstable / degenerate; keep λ_se subordinate.
- **λ_feat (encoder feature-matching) HURT WER** (Exp 5, esp. low SNR) — keep at 0 unless re-investigating; `FEAT_LAYERS=(2,4,6)` if you do.
- **Warm-start from `model_240`** — combined objective from scratch fails to converge (IRIS finding).

## Constraints & gotchas

- **The SE model must be ASR-INDEPENDENT.** We don't own/tune/pick the deployment ASR. No beam-search/decoding tricks, no "switch to a bigger Whisper as a target." Whisper-small is a *fixed frozen proxy* (training signal + one evaluator) only. "Independence" = a transfer property to verify on a held-out ASR, not a reason to drop the ASR from training (pure fidelity loss provably hurts WER).
- **Fixed eval protocol — don't change it or comparability breaks.** `build_eval_set`: 200 `test-clean` utts (seed 1234), each paired with one deterministic MUSAN clip, scored at SNR {−10,−5,0,5,10,20}; same clean+noise across SNRs so SNR is the only variable. Always report **clean (upper bound)**, **noisy (baseline)**, enhanced. Whisper greedy decoding.
- **MUSAN is used for BOTH train and eval** → current results are a PoC, not a generalization claim. Speech is properly held out (disjoint LibriSpeech speakers). Pending: unseen-noise + cross-ASR transfer checks.
- **SNR mixing** (`data_utils.mix_at_snr`): noise rescaled to `RMS_speech / 10^(SNR/20)` (global, whole-utterance RMS); peak-clip guard preserves SNR (scales speech+noise together).

## Data

- Clean: **LibriSpeech** (English, 16 kHz). `train-clean-100` (28,539 utts, 100.6 h) train; `test-clean` (2,620 utts, 5.4 h) eval.
- Noise: **MUSAN** `noise/` subset only (930 files, 6.2 h; free-sound + sound-bible). Music/speech subsets unused.

## Findings so far

- ASR-aware fine-tune **works**: enhanced WER ≤ noisy at every SNR, beats the fidelity-trained `model_240` everywhere (e.g. −10 dB: 40.99 → ~20.6, was 35.96; full table in `report_wer.md`). Current best: `runs/exp4/deepvqe_exp2_step9000.tar`.
- Recipe **converges in <1 epoch**; training longer (Exp 4) gave no gain; λ_feat (Exp 5) hurt; lowering λ_se won't help (CE already dominates). Near the ceiling for DeepVQE + Whisper-small.
- Clean WER (3.30%) is the bound but **unreachable at low SNR** (noise destroys information) — not the per-SNR target.
- Keep `exps.md` and `results/` updated when an experiment finishes.
