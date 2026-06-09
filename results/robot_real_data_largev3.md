# Real-data test with a STRONGER ASR — Whisper-large-v3 vs Whisper-small

Same 247 Vietnamese robot utterances, same protocol (lang=vi, greedy + no_repeat_ngram=3),
swapping only the recognizer. large-v3 loaded in fp16 (3.1 GB weights, ~4.1 GB peak → fits 8 GB).
Reproduce: `python -m tools.eval_robot --model openai/whisper-large-v3 --fp16 --batch-size 8 --detail --per-file-csv outputs/robot_per_file_largev3.csv --folder-csv outputs/robot_folder_summary_largev3.csv`

## Overall (corpus WER/CER %)
| ASR | noisy WER | noisy CER | model_240 WER | model_D WER | model_240 ΔWER | model_D ΔWER |
|---|---:|---:|---:|---:|---:|---:|
| whisper-small | 60.66 | 40.83 | 66.10 | 65.84 | +5.43 | +5.18 |
| **whisper-large-v3** | **41.98** | **29.83** | 45.91 | 46.12 | +3.93 | +4.14 |

## Two conclusions

### 1. Most of the "bad absolute WER" was the weak recognizer — confirmed.
large-v3 cuts noisy WER 60.7 → 42.0 and CER 40.8 → 29.8. The low-noise folders drop hugely:
`silence` 35→13 WER (6.4 CER), `park` 59→26, `washing_dishes` 53→33. So Whisper-small's weak
Vietnamese was a big part of the story.

### 2. Denoising STILL hurts — even with the strong ASR. This is the real finding.
Overall, a SOTA Vietnamese-capable recognizer *still prefers the raw noisy audio*: model_D +4.14 WER,
model_240 +3.93 WER. So the denoiser's harm on this data is **not** a whisper-small artifact — it is
genuinely removing speech information (hits down, deletions up). On easy/low-noise clips it can only
hurt; it helps only where noise is overwhelming.

### 3. Bonus — this is effectively the pending cross-ASR transfer check (Exp G), and model_D FAILS it.
Per-utterance win/loss vs noisy (by CER):
| ASR | model_240 net | model_D net |
|---|---:|---:|
| whisper-small | −58 | **−32** (model_D better) |
| whisper-large-v3 | **−20** | −36 (model_240 better) |

Under whisper-small, the ASR-aware model_D beat the fidelity model_240 (as designed — it was tuned to
whisper-small's decoder CE). Under the held-out large-v3, that advantage **reverses**: the plain
fidelity model_240 transfers better. The whisper-small-specific gains do not generalize to a different
recognizer — exactly the ASR-overfitting risk flagged in `exps.md` (Exp G). large-v3 here plays the
role of the held-out ASR.

## Per-folder, large-v3 (mean per-file WER/CER; ΔWER = with model_D − without)
| folder | n | WER w/o | WER w/ D | ΔWER | CER w/o | ΔCER |
|---|--:|--:|--:|--:|--:|--:|
| silence | 27 | 12.6 | 14.5 | +1.9 | 7.0 | +2.1 |
| robot_dung_yen | 12 | 21.5 | 17.8 | −3.7 | 10.9 | +0.2 |
| piano | 26 | 26.8 | 41.8 | +14.9 | 15.7 | +12.6 |
| crowd | 27 | 27.6 | 35.1 | +7.6 | 17.4 | +5.3 |
| park | 30 | 27.9 | 31.2 | +3.4 | 15.3 | +3.0 |
| washing_dishes | 27 | 32.6 | 40.8 | +8.2 | 21.3 | +10.1 |
| audio_1 | 27 | 34.7 | 53.7 | +19.0 | 23.6 | +13.6 |
| audio_2 | 18 | 72.5 | 62.5 | −10.0 | 56.3 | −9.5 |
| audio_3 | 24 | 81.7 | 90.1 | +8.5 | 60.9 | +7.5 |
| robot_di_chuyen | 29 | 167.4 | 161.2 | −6.2 | 135.5 | −4.7 |
| **ALL** | 247 | 51.9 | 57.0 | +5.2 | 37.6 | +4.6 |

(Note: large-v3 hallucinates badly on the destroyed `robot_di_chuyen` audio → WER >100% from huge
insertions, despite no_repeat_ngram=3. That folder is near-unintelligible regardless of ASR.)

Artifacts (gitignored, regenerable): `outputs/robot_per_file_largev3.csv`,
`outputs/robot_folder_summary_largev3.csv`, `outputs/robot_detail_largev3.json`.
