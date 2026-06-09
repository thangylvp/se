# Speech Enhancement — WER Evaluation

| Test: English · LibriSpeech `test-clean` (200 utts) · noise: MUSAN · ASR: Whisper-small (frozen, greedy) · values = WER % | | | |
|---|---|---|---|
| **SNR (dB)** | **noisy** | **clean** | **model D** |
| −10 | 40.99 | 3.30 | 20.59 |
| −5 | 13.45 | 3.30 | 10.86 |
| 0 | 6.76 | 3.30 | 5.98 |
| 5 | 4.63 | 3.30 | 4.25 |
| 10 | 3.87 | 3.30 | 3.84 |
| 20 | 3.51 | 3.30 | 3.46 |

> `noisy` = ASR on noisy input (no enhancement) · `clean` = ASR on clean reference (upper bound) · `model D` = ASR on our enhanced output.
