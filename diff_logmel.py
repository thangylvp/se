"""
Differentiable Whisper log-mel front-end (pure torch).

Replicates transformers/openai-whisper feature extraction exactly, but with
torch ops so gradients flow from Whisper's input features back into the
upstream SE model. The stock WhisperFeatureExtractor uses numpy and detaches —
using it in the training loop would silently zero the ASR gradient.

Constants for whisper-small (and all <=medium): 80 mels, n_fft=400, hop=160,
16 kHz, 30 s = 480000 samples = 3000 frames.
"""
import torch
import torch.nn.functional as F

N_FFT = 400
HOP = 160
N_SAMPLES = 480000
N_FRAMES = 3000


class DiffLogMel(torch.nn.Module):
    def __init__(self, processor, device):
        super().__init__()
        # mel_filters from the processor: shape (n_freq=201, n_mels=80) -> (80, 201)
        filters = torch.as_tensor(
            processor.feature_extractor.mel_filters.T, dtype=torch.float32
        )
        self.register_buffer("filters", filters.to(device))
        self.register_buffer("window", torch.hann_window(N_FFT).to(device))

    def forward(self, wav):
        """wav: (B, L) @ 16 kHz, float32. Returns (B, 80, 3000) log-mel."""
        # pad/trim each example to exactly 30 s (Whisper encoder needs 3000 frames)
        if wav.shape[1] < N_SAMPLES:
            wav = F.pad(wav, (0, N_SAMPLES - wav.shape[1]))
        elif wav.shape[1] > N_SAMPLES:
            wav = wav[:, :N_SAMPLES]

        stft = torch.stft(wav, N_FFT, HOP, window=self.window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2          # (B, 201, 3000)
        mel = torch.matmul(self.filters, magnitudes)     # (B, 80, 3000)

        log_spec = torch.clamp(mel, min=1e-10).log10()
        # per-sample dynamic-range floor, then Whisper's [-1,1]-ish normalization
        log_spec = torch.maximum(
            log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0
        )
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
