# afe — Audio Front-End for ASR-aware Speech Enhancement

Fine-tune a **DeepVQE speech-enhancement front-end so it lowers a downstream ASR's
WER**, not just denoises. A DeepVQE trained on signal-fidelity loss denoises well
but *hurts* WER as an ASR pre-processor; we fix that by fine-tuning it with a
**frozen-Whisper ASR loss** (decoder cross-entropy as the primary driver).

See `docs/exps.md` for the experiment log, `docs/related_works.md` for the
literature, and `docs/report_wer.md` for the headline WER results.

## Layout (detectron2-lite)

```
afe/                 # the installable package
├── modeling/        # DeepVQE arch + STFTWrapper
├── data/            # audio ops, LibriSpeech, on-the-fly mixing, robot test set
├── losses/          # HybridLoss (fidelity) + asr_aware composite loss
├── asr/             # frozen-Whisper wrapper + differentiable log-mel
├── engine/          # training loop
├── solver/          # optimizer + LR schedule
├── evaluation/      # headroom / trajectory / robot evaluators
├── checkpoint/      # checkpoint load/save
├── config/          # config dataclasses + YAML loader
└── utils/           # constants (SR/STFT params) + text metrics (WER/CER)
configs/             # YAML experiment configs (train/, eval/)
tools/               # thin CLIs, run as `python -m tools.<name>`
tests/  docs/  results/  checkpoints/
```

## Install

```bash
conda create -n afe python=3.11 -y && conda activate afe
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128  # GPU (cu128)
pip install -e ".[dev]"
cp .env.example .env && $EDITOR .env   # add HF_TOKEN (Whisper is public; token avoids rate limits)
```

Datasets and the warm-start checkpoint are not in git — see "External data" below.

## Quickstart

```bash
set -a; source .env; set +a   # load HF_TOKEN

# denoise a file/dir -> *_enhanced.wav
python -m tools.infer --checkpoint checkpoints/model_D.tar --input clip.wav --output-dir out

# transcribe a file (ASR only, no denoise)
python -m tools.transcribe out/clip_enhanced.wav --model openai/whisper-large-v3 --fp16

# one file: without vs with denoising, + WER/CER
python -m tools.test_one clip.wav

# gradient-flow smoke test (always run before real training)
python -m tools.train --smoke --librispeech ../dataset/LibriSpeech/test-clean

# train the ASR-aware fine-tune (config-driven; CLI flags override YAML)
python -m tools.train --config configs/train/exp2.yaml --device 0

# LibriSpeech WER eval
python -m tools.eval_headroom --checkpoint checkpoints/model_D.tar --device 0
python -m tools.eval_trajectory --checkpoints runs/exp4/deepvqe_step3000.tar runs/exp4/deepvqe_step9000.tar

# real Vietnamese robot test set (no-denoise vs denoise), with CSV reports
python -m tools.eval_robot --model openai/whisper-large-v3 --fp16 --detail \
    --per-file-csv outputs/robot_per_file.csv --folder-csv outputs/robot_folder.csv
```

## External data (not in git)

Scripts default to siblings of this repo: `../dataset/` (LibriSpeech + MUSAN) and
`../ckpt/deepvqe/model_240.tar` (warm-start checkpoint). All paths are overridable
via CLI flags / configs. Download steps are in `docs/` and the dataset constants in
`afe/utils/constants.py`. The released best checkpoint `checkpoints/model_D.tar` is
committed.

## Config

Experiments are YAML over a dataclass (`afe/config/defaults.py`); no yacs/registry.
CLI flags override YAML, which overrides dataclass defaults. The fixed LibriSpeech
eval protocol lives in `configs/eval/librispeech.yaml` — do not change it (breaks
comparability with `results/`).

## Tests

```bash
pytest -q
```
Covers metrics, config loading, the robot pairing, SE forward/checkpoint round-trip,
and that `DiffLogMel` matches the HF Whisper feature extractor (the invariant the
training gradient depends on).
