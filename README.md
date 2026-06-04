# dn_asr

Goal: train a speech-enhancement (SE) front-end that actually helps our ASR model
(lower WER), not just one that sounds clean. The teammate's DeepVQE denoiser is
great at denoising but doesn't improve WER as an ASR pre-processor — so we plan to
fine-tune it with an ASR-aware loss.

**Step 1 (done): reproduce the teammate's denoiser inference and confirm the
architecture.** This README documents how the checkpoint is run.

## What the checkpoint is

`ckpt/deepvqe/model_240 2.tar` — a **DeepVQE** denoiser, epoch 240, `best_score`
2.273. It's a `torch.save` dict:

```
{ 'model': <state_dict, 156 tensors>, 'epoch': 240, 'optimizer', 'scheduler', 'best_score' }
```

The `model` state_dict keys are prefixed `model.` (e.g. `model.enblock1.conv.weight`)
because training wrapped the net as `STFTWrapper(self.model = DeepVQE())`. To load
into a bare `DeepVQE`, strip the `model.` prefix.

### Architecture (verified by exact shape match)

DeepVQE — complex-spectrogram U-Net, defined identically in `deepvqe/deepvqe.py`
and `aecdns/models/deepvqe.py` (byte-for-byte identical; vendored here as
`deepvqe.py`):

- Input: STFT real/imag, shape `(B, F=257, T, 2)`
- `FE`: power-law feature compression (divide by `mag^0.7`, c=0.3)
- Encoder: 5 blocks `(2→64→128→128→128→128)`, freq-stride 2 each → `(B,128,T,9)`
- Bottleneck: GRU `1152→576` + FC `576→1152` (1152 = 128·9, 576 = 64·9)
- Decoder: 5 sub-pixel blocks with skip connections → `(B,27,T,257)`
- `CCM`: complex convolving mask (3×3) applied to the noisy STFT → `(B,257,T,2)`

### Exact STFT pipeline (must be matched, incl. for training)

| param | value |
|-------|-------|
| sample rate | **16000 Hz** (resample in/out if input differs) |
| n_fft | 512 |
| hop_length | 256 |
| win_length | 512 |
| window | `torch.hann_window(512)` (periodic) |
| stft | `onesided=True, return_complex=True` → `view_as_real` |
| istft | same params; pad/trim to original length |

No input gain/RMS normalization is applied — the model runs directly on STFT
coefficients (the only "normalization" is the internal `FE` magnitude compression).

Source of truth: `aecdns/models/modules/wrapper.py::STFTWrapper`,
`aecdns/inference/infer_file.py`, `aecdns/utils/checkpoint_utils.py`.

## How to run inference (this folder, self-contained)

Files here vendor everything needed — no dependency on the `aecdns` tree:
- `deepvqe.py` — DeepVQE architecture (exact copy)
- `se_model.py` — `STFTWrapper` + `load_se_model()` checkpoint loader
- `infer.py` — CLI

```bash
conda activate dn_asr   # see requirements.txt for env setup
python infer.py \
    --checkpoint "../ckpt/deepvqe/model_240 2.tar" \
    --input ../noisy_audio/audio_8c3e9f2b_1773733142.wav \
    --output-dir out
# directory input + GPU:
python infer.py --checkpoint <ckpt.tar> --input <wav_dir> --output-dir out --device 0
```

**Verified:** output is bit-identical (max abs diff `0.0`) to the teammate's
`aecdns/inference/infer_file.py` on the same input, and `load_state_dict` is
`strict=True` (any arch mismatch fails loud). On a noisy sample RMS dropped
0.063 → 0.032 with no NaNs and length preserved.

## Environment

Verified stack: `python 3.11`, `torch 2.8.0+cu128`, `librosa 0.11`,
`soundfile 0.13`, `einops 0.8`, `numpy 1.26`. See `requirements.txt` for the
conda + pip setup. (The teammate's `aecdns` pins `torch==1.11`, but 2.8 runs the
model identically — the STFT/iSTFT API is unchanged.)

## Next step — training with an ASR loss

The teammate's SE training loss is **HybridLoss** (spectral RI + magnitude MSE):
`SEtrain/loss_factory.py::HybridLoss` (λ_ri=30, λ_mag=70, same STFT params as
above). It optimizes signal fidelity, which explains good denoising but no WER
gain. Plan: keep enhancement output as waveform and add an ASR-feature / CTC loss
from a frozen ASR model on the enhanced audio, so gradients push the SE front-end
toward what the recognizer needs. ASR backbone + deps to be added to
`requirements.txt` once chosen.
