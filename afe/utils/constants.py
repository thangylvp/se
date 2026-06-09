"""Single source of truth for sample rate, STFT params, and Whisper defaults.

These MUST stay consistent across the SE model, the training loss, and the
differentiable log-mel — historically they were duplicated across files.
"""

SR = 16000

# ---- SE STFT (DeepVQE / STFTWrapper / HybridLoss) ----
# 16 kHz, periodic Hann. Used for waveform<->spectrogram and the fidelity loss.
SE_N_FFT = 512
SE_HOP = 256
SE_WIN = 512

# ---- Whisper log-mel front-end (DiffLogMel) ----
# whisper-small and all <= medium: 80 mels, n_fft=400, hop=160, 30 s window.
WHISPER_N_FFT = 400
WHISPER_HOP = 160
WHISPER_N_MELS = 80
WHISPER_N_SAMPLES = 480000   # 30 s @ 16 kHz
WHISPER_N_FRAMES = 3000

# ---- Whisper decoding defaults ----
DEFAULT_WHISPER = "openai/whisper-small"
# suppress repetition-loop hallucinations on noisy/short audio (0 = off)
DEFAULT_NO_REPEAT_NGRAM = 3

# ---- fixed LibriSpeech eval protocol (do not change — breaks comparability) ----
EVAL_SNRS = (-10, -5, 0, 5, 10, 20)
EVAL_N_UTTS = 200
EVAL_SEED = 1234
