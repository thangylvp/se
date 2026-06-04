# checkpoints/

## `model_D.tar` — best ASR-aware DeepVQE (the model in `report_wer.md`)

The DeepVQE speech-enhancement model fine-tuned with the frozen-Whisper ASR loss
(`L = λ_ce·CE + λ_se·HybridLoss`, warm-started from the teammate's `model_240`).
This is the **"model D"** column in `../report_wer.md`.

- Provenance: training run `runs/exp4`, step 9000 (`deepvqe_exp2_step9000.tar`).
  Recipe: λ_ce=1.0, λ_se=0.05, bs=8, lr=1e-4 (cosine), MUSAN noise, SNR weighted to −5…+5.
- Format: `{"model": {"model.<k>": v}, "step": 9000}` — same as all checkpoints here;
  `se_model.load_se_model()` strips the `model.` prefix automatically.

### WER (Whisper-small, LibriSpeech test-clean, MUSAN, n=200, greedy)

| SNR (dB) | noisy | clean | model D |
|---:|---:|---:|---:|
| −10 | 40.99 | 3.30 | 20.59 |
| −5 | 13.45 | 3.30 | 10.86 |
| 0 | 6.76 | 3.30 | 5.98 |
| 5 | 4.63 | 3.30 | 4.25 |
| 10 | 3.87 | 3.30 | 3.84 |
| 20 | 3.51 | 3.30 | 3.46 |

### Use

```bash
python infer.py --checkpoint checkpoints/model_D.tar --input <wav_or_dir> --output-dir out
# or, to reproduce the table:
python eval_headroom.py --librispeech ../dataset/LibriSpeech/test-clean \
    --noise-dir ../dataset/musan/noise --checkpoint checkpoints/model_D.tar --n-utts 200 --device 0
```
