# Real-data test — DeepVQE denoising vs downstream ASR (Vietnamese robot recordings)

- **Data**: `../data_test_robot`, 247 real noisy utterances (no clean reference), **Vietnamese**
  (robot voice-assistant commands), 16 kHz mono. Categories = noise conditions.
- **ASR**: frozen Whisper-small, `language="vi"`, greedy + `no_repeat_ngram_size=3`
  (needed — without it Whisper falls into repetition-loop hallucinations on the noisy/short
  clips, blowing WER past 100–600% and making the metric meaningless).
- **Metric**: corpus WER and CER, Vietnamese normalization (NFC, lowercase, strip punctuation).
- **Conditions**: `noisy` (no enhancement) · `model_240` (fidelity-trained DeepVQE) ·
  `model_D` (our ASR-aware fine-tune).
- Reproduce: `python -m tools.eval_robot --detail --per-file-csv outputs/robot_per_file.csv --folder-csv outputs/robot_folder_summary.csv`

## WER (%)
| category | n | noisy | model_240 | model_D |
|---|---:|---:|---:|---:|
| recording_2404/robot_di_chuyen | 29 | 109.96 | 107.97 | **100.40** |
| recording_2404/robot_dung_yen | 12 | **52.11** | 60.56 | 57.04 |
| vmo_1703/audio_1 | 27 | **47.71** | 54.58 | **47.71** |
| vmo_1703/audio_2 | 18 | 81.94 | **80.09** | 89.35 |
| vmo_1703/audio_3 | 24 | **91.44** | 94.14 | 102.25 |
| vmo_2305/crowd | 27 | **50.87** | 54.33 | 52.25 |
| vmo_2305/park | 30 | **55.56** | 64.20 | 69.38 |
| vmo_2305/piano | 26 | **55.22** | 59.89 | 58.52 |
| vmo_2305/silence | 27 | **32.34** | 35.97 | 40.59 |
| vmo_2305/washing_dishes | 27 | **50.17** | 66.45 | 58.47 |
| **OVERALL** | **247** | **60.66** | 66.10 | 65.84 |

## CER (%)
| category | n | noisy | model_240 | model_D |
|---|---:|---:|---:|---:|
| recording_2404/robot_di_chuyen | 29 | 87.78 | 84.34 | **81.45** |
| recording_2404/robot_dung_yen | 12 | **30.08** | 34.40 | 34.24 |
| vmo_1703/audio_1 | 27 | **31.14** | 39.23 | 31.44 |
| vmo_1703/audio_2 | 18 | 59.56 | **59.03** | 64.63 |
| vmo_1703/audio_3 | 24 | **67.56** | 72.62 | 76.24 |
| vmo_2305/crowd | 27 | **31.82** | 34.09 | 34.01 |
| vmo_2305/park | 30 | **33.54** | 38.31 | 46.31 |
| vmo_2305/piano | 26 | **34.05** | 37.16 | 36.12 |
| vmo_2305/silence | 27 | **18.13** | 21.14 | 21.67 |
| vmo_2305/washing_dishes | 27 | **33.21** | 46.85 | 38.97 |
| **OVERALL** | **247** | **40.83** | 45.09 | 44.94 |

## Findings
- **Denoising slightly HURTS overall** here: noisy is best (WER 60.7 vs ~66; CER 40.8 vs ~45).
  Among the two SE models, the ASR-aware **model_D ≥ model_240** (65.84 vs 66.10 WER;
  44.94 vs 45.09 CER) — consistent with the project thesis, but the absolute SE effect is negative.
- **SE helps only in the loudest category** — `robot_di_chuyen` (robot-motor self-noise):
  WER 109.96 → 100.40, CER 87.78 → 81.45 with model_D. This mirrors the English finding
  (Exp 1): SE helps WER only at very low SNR and hurts elsewhere.
- **Why this is expected**: out-of-distribution. DeepVQE was trained on *English* LibriSpeech +
  MUSAN with an *English*-Whisper ASR loss. This set is *Vietnamese* with different noise
  (motor, piano, washing). The denoiser's artifacts cost more than the noise it removes.
- **Absolute WER is high because Whisper-small is weak on Vietnamese**, not just noise: even the
  low-noise `silence` category is 32% WER / 18% CER — that's the ASR's Vietnamese floor here.
- **CER < WER everywhere** (Vietnamese is monosyllabic; one wrong syllable = one whole word error),
  so CER is the more informative metric for this data.

---

# Detailed breakdown (`python -m tools.eval_robot --detail`; per-utt dump in `outputs/robot_detail.json`)

## By source subset — WER/CER with Δ vs noisy
| subset | n | noisy WER/CER | model_240 ΔWER/ΔCER | model_D ΔWER/ΔCER |
|---|---:|---:|---:|---:|
| recording_2404 (robot self-noise) | 41 | 89.06 / 66.94 | +1.78 / −0.64 | **−4.33 / −2.54** |
| vmo_1703 | 69 | 70.70 / 50.34 | +3.09 / +4.66 | +5.38 / +4.20 |
| vmo_2305 (park/crowd/piano/silence/dishes) | 137 | 49.46 / 30.51 | +7.34 / +5.24 | +7.34 / +5.64 |

→ **model_D only helps on the loud robot-self-noise subset**; it hurts on the ambient-noise bulk.

## By category — ΔWER vs noisy (negative = denoise helps)
| category | n | noisy WER | model_240 Δ | model_D Δ |
|---|---:|---:|---:|---:|
| robot_di_chuyen (robot moving) | 29 | 109.96 | −1.99 | **−9.56** |
| audio_1 | 27 | 47.71 | +6.86 | **0.00** |
| audio_2 | 18 | 81.94 | −1.85 | +7.41 |
| robot_dung_yen | 12 | 52.11 | +8.45 | +4.93 |
| crowd | 27 | 50.87 | +3.46 | +1.38 |
| piano | 26 | 55.22 | +4.67 | +3.30 |
| washing_dishes | 27 | 50.17 | +16.28 | +8.31 |
| silence | 27 | 32.34 | +3.63 | +8.25 |
| audio_3 | 24 | 91.44 | +2.70 | +10.81 |
| park | 30 | 55.56 | +8.64 | +13.83 |

## Error structure (overall, words)
| cond | hits | subs | dels | ins |
|---|---:|---:|---:|---:|
| noisy | **1296** | 1303 | 200 | 195 |
| model_240 | 1189 | 1354 | 256 | 240 |
| model_D | 1163 | 1342 | **294** | 207 |

→ Denoising **lowers hits and raises deletions** — it is erasing/distorting real speech, not just
noise. (Insertions are NOT the problem after the `no_repeat_ngram` fix.)

## Per-utterance win/loss vs noisy (by CER)
| model | improved | worse | tie | net |
|---|---:|---:|---:|---:|
| model_240 | 77/247 | 135 | 35 | −58 |
| model_D | **89/247** | 121 | 37 | **−32** |

→ **model_D beats model_240** (more per-utt wins, smaller net loss) — the ASR-aware fine-tune is the
better denoiser for ASR, exactly as the project claims — but on this OOD Vietnamese data it is still
net-negative vs leaving the audio alone.

## Illustrative transcriptions
- `silence` (low noise): noisy ≈ correct, denoise adds small slips → *"trả lời **ngu** quá"* →
  noisy "**mua**", model_D "**mút**". SE can only hurt when there's little noise to remove.
- `audio_2` (near-silent): noisy Whisper-hallucinates YouTube boilerplate
  *"Cảm ơn các bạn đã đón xem video này"*; model_240 partially recovers the real command.
- `robot_di_chuyen` (loudest): everything is wrong, but model_D shortens the garbage → fewer
  insertions/subs, hence the only clear win.
