# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a **DeepVQE speech-enhancement (SE) front-end so it lowers a downstream ASR's WER**, not just denoises. A teammate's DeepVQE (trained on signal-fidelity loss) denoises well but *hurts* WER when used as an ASR pre-processor; we fix that by fine-tuning it with a **frozen-Whisper ASR loss**. The repo is the importable `afe` package (Audio Front-End) — self-sufficient given the setup below.

## Repository layout (detectron2-lite package)

```
afe/                 # importable package (pip install -e .)
├── modeling/        deepvqe.py (arch), se_model.py (STFTWrapper, build_se_model)
├── data/            audio.py (load/mix/SNR), librispeech.py (list + build_eval_set),
│                    mix_dataset.py (MixDataset + collate), robot.py (build_pairs)
├── losses/          hybrid.py (HybridLoss), asr_aware.py (loss_step + FEAT_LAYERS)
├── asr/             whisper.py (WhisperASR eval), diff_logmel.py (DiffLogMel training)
├── engine/          trainer.py (build, train loop, smoke_test)
├── solver/          lr.py (cosine_lr, build_optimizer)
├── evaluation/      headroom.py, trajectory.py, robot.py (all return metric dicts)
├── checkpoint/      io.py (load_se_model, save_se_checkpoint)
├── config/          defaults.py (TrainConfig/EvalConfig dataclasses + load_config YAML)
└── utils/           constants.py (SR/STFT params — single source), text.py (WER/CER, normalizers)
configs/             train/{exp2,exp4,exp5}.yaml, eval/librispeech.yaml
tools/               thin argparse CLIs, run as `python -m tools.<name>`
tests/               pytest suite (metrics, config, pairing, modeling, logmel-matches-HF)
docs/                exps.md (READ FIRST), related_works.md, report_wer.md
results/             curated WER reports (.md only)        checkpoints/  model_D.tar (released best)
outputs/             generated CSV/JSON artifacts (gitignored — regenerable, not curated)
```

NOT committed (recreate per "Setup"): `.env`, `runs/` (training checkpoints), `out/`, the sibling `../dataset`, `../ckpt`, and the reference-only `../aecdns`, `../deepvqe`, `../SEtrain`.

## External data (siblings of this repo — NOT in git)

Defaults assume this repo sits next to `../ckpt/deepvqe/model_240.tar` (warm-start checkpoint, from teammate, ~90 MB) and `../dataset/` (LibriSpeech + MUSAN). `deepvqe.py` and `losses/hybrid.py` are vendored, so `../aecdns`, `../deepvqe`, `../SEtrain` are reference-only. All paths are overridable via CLI flags / configs.

## Setup on a new machine

```bash
conda create -n afe python=3.11 -y && conda activate afe
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # GPU (Blackwell/50xx: cu128)
pip install -e ".[dev]"
cp .env.example .env   # add HF_TOKEN (Whisper is public; token avoids rate limits)

# Datasets -> ../dataset  (LibriSpeech = OpenSLR-12, MUSAN = OpenSLR-17, both CC BY 4.0)
mkdir -p ../dataset && cd ../dataset
wget https://www.openslr.org/resources/12/test-clean.tar.gz      && tar xzf test-clean.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz && tar xzf train-clean-100.tar.gz  # ~6 GB, train only
wget https://www.openslr.org/resources/17/musan.tar.gz           && tar xzf musan.tar.gz musan/noise # noise/ only
cd -
# -> ../dataset/LibriSpeech/{test-clean,train-clean-100} and ../dataset/musan/noise (930 files)
# Checkpoint -> ../ckpt/deepvqe/model_240.tar
```

Prefix anything that loads Whisper with `set -a; source .env; set +a`.

## Common commands

```bash
set -a; source .env; set +a

python -m tools.infer --checkpoint checkpoints/model_D.tar --input <wav_or_dir> --output-dir out  # denoise only
python -m tools.transcribe <wav_or_dir> [--model openai/whisper-large-v3 --fp16]                  # ASR only
python -m tools.test_one <wav>                                  # one file: without vs with denoise + WER/CER

python -m tools.train --smoke --librispeech ../dataset/LibriSpeech/test-clean   # ALWAYS run before training
python -m tools.train --config configs/train/exp2.yaml --device 0               # train (CLI flags override YAML)

python -m tools.eval_headroom --checkpoint <ckpt.tar> --device 0                # one-checkpoint SNR table
python -m tools.eval_trajectory --checkpoints <a> <b> ... --device 0            # many checkpoints, one Whisper load
python -m tools.eval_robot --model openai/whisper-large-v3 --fp16 --detail \    # real Vietnamese robot test set
    --per-file-csv outputs/robot_per_file.csv --folder-csv outputs/robot_folder.csv

pytest -q
```

GPU note: a bs=8 training run uses ~27–30 GB — don't eval while training on the same card.

## Architecture / how it fits together

- **`DeepVQE`** (`afe/modeling/deepvqe.py`): complex-spectrogram U-Net. `(B,F=257,T,2)` → 5 encoder blocks → GRU bottleneck → 5 decoder blocks → CCM complex mask.
- **`STFTWrapper`** (`afe/modeling/se_model.py`): waveform→STFT→DeepVQE→iSTFT→waveform. STFT params are **fixed in `afe/utils/constants.py` and must match everywhere: 16 kHz, n_fft=512, hop=256, win=512, periodic Hann**.
- **checkpoint IO** (`afe/checkpoint/io.py`): `load_se_model` strips the `model.` key prefix; `save_se_checkpoint` restores it. Checkpoints are `{"model": {"model.<k>": v}}`.
- **`DiffLogMel`** (`afe/asr/diff_logmel.py`): pure-torch Whisper log-mel (n_fft=400, hop=160, 80 mels, 3000 frames), verified to match the HF processor (`tests/test_logmel.py`). **Required** — the stock `WhisperFeatureExtractor` is numpy and detaches, silently zeroing the ASR gradient.
- **training** (`afe/engine/trainer.py` + `afe/losses/asr_aware.py`): DeepVQE trainable (warm-started from `model_240`); **Whisper frozen** (`eval()` + `requires_grad_(False)`) — gradients flow *through* it via the differentiable log-mel. `MixDataset` mixes clean+noise on the fly. Loss: `L = λ_ce·CE(WhisperDec(enh), txt) + λ_se·HybridLoss(enh, clean) + λ_feat·L1(WhisperEnc(enh), WhisperEnc(clean))`.

## Loss design — settled; read before changing the recipe

- **λ_ce (decoder CE through frozen Whisper) is the primary, validated driver** (Dissen et al., Interspeech 2024). Dominates (~12:1 over the SE term).
- **λ_se (HybridLoss) is a small stabilizer** (~0.05), not the anchor — CE-alone can go degenerate; keep λ_se subordinate.
- **λ_feat (encoder feature-matching) HURT WER** (Exp 5, esp. low SNR) — keep at 0 unless re-investigating; `FEAT_LAYERS=(2,4,6)` in `afe/losses/asr_aware.py`.
- **Warm-start from `model_240`** — combined objective from scratch fails to converge (IRIS finding).

## Constraints & gotchas

- **The SE model must be ASR-INDEPENDENT.** We don't own/tune/pick the deployment ASR. No beam-search/decoding tricks, no "switch to a bigger Whisper as a target." Whisper-small is a *fixed frozen proxy* (training signal + one evaluator) only. "Independence" = a transfer property to verify on a held-out ASR.
- **Fixed eval protocol — don't change it (`configs/eval/librispeech.yaml`, `afe.data.librispeech.build_eval_set`).** 200 `test-clean` utts (seed 1234), each paired with one deterministic MUSAN clip, scored at SNR {−10,−5,0,5,10,20}; same clean+noise across SNRs. Always report clean (upper bound), noisy (baseline), enhanced. Whisper greedy.
- **MUSAN is used for BOTH train and eval** → results are a PoC, not a generalization claim. Speech is held out (disjoint speakers).
- **SNR mixing** (`afe.data.audio.mix_at_snr`): noise rescaled to `RMS_speech / 10^(SNR/20)`; peak-clip guard preserves SNR.
- **Whisper hallucination on hard/short/low-info audio** (silence, very noisy, or wrong-language clips): produces repetition loops or stock phrases ("thank you for watching", subscribe prompts). `WhisperASR.transcribe` defaults to `no_repeat_ngram_size=3` (constant `DEFAULT_NO_REPEAT_NGRAM`) to suppress loops; single-phrase hallucinations still need correct `language=`.

## Data

- Clean: **LibriSpeech** (English, 16 kHz). `train-clean-100` (28,539 utts) train; `test-clean` (2,620 utts) eval.
- Noise: **MUSAN** `noise/` subset only (930 files; free-sound + sound-bible).
- Real test set: **`../data_test_robot`** — 247 real **Vietnamese** robot voice-assistant clips (no clean ref), organized by noise condition. Eval with `python -m tools.eval_robot` (lang=vi, CER reported alongside WER).

## Findings so far

- ASR-aware fine-tune **works on the English/Whisper-small proxy**: enhanced WER ≤ noisy at every SNR, beats `model_240` everywhere (−10 dB: 40.99 → ~20.6; `docs/report_wer.md`). Released best: `checkpoints/model_D.tar` (= exp4 step 9000). Recipe converges in <1 epoch; Exp 4 (longer) no gain; Exp 5 (λ_feat) hurt.
- **Real Vietnamese robot data (`docs`/`results/robot_real_data*.md`): denoising HURTS overall** — raw noisy beats both SE models except in the loudest condition (`robot_di_chuyen`). Cause: out-of-distribution (English-trained SE + English-Whisper loss applied to Vietnamese with different noise). Holds under whisper-large-v3 too, so it's not a small-model artifact; and the model_D-over-model_240 advantage does NOT transfer to large-v3 (effectively the pending cross-ASR check, Exp G).
- Keep `docs/exps.md` and `results/` updated when an experiment finishes.
