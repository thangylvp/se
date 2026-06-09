"""DeepVQE speech-enhancement model wrapped with the fixed STFT front/back end.

Wraps the vendored `DeepVQE` (verified identical to the teammate's arch) with the
exact STFT pipeline used by aecdns/models/modules/wrapper.py::STFTWrapper:

    waveform (B, L) @ 16 kHz
      -> torch.stft(n_fft=512, hop=256, win=512, periodic Hann, onesided, complex)
      -> view_as_real -> (B, F=257, T, 2)
      -> DeepVQE  -> (B, 257, T, 2)
      -> complex  -> torch.istft(same params)
      -> pad/trim to original length -> (B, L)

Checkpoint loading/saving lives in `afe.checkpoint.io`.
"""
import torch
import torch.nn as nn

from afe.modeling.deepvqe import DeepVQE
from afe.utils.constants import SE_HOP, SE_N_FFT, SE_WIN


class STFTWrapper(nn.Module):
    """Waveform -> STFT -> DeepVQE -> iSTFT -> waveform. Mirrors aecdns STFTWrapper."""

    def __init__(self, model, n_fft=SE_N_FFT, hop_len=SE_HOP, win_len=SE_WIN):
        super().__init__()
        self.model = model
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len

    def forward(self, x):
        """x: (B, L) waveform. Returns (B, L) waveform."""
        device = x.device
        n_samples = x.shape[1]
        window = torch.hann_window(self.win_len).to(device)

        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len,
            window=window, onesided=True, return_complex=True,
        )  # (B, F, T) complex
        spec_ri = torch.view_as_real(spec)  # (B, F, T, 2)

        enh_ri = self.model(spec_ri)  # (B, F, T, 2)

        enh = torch.complex(enh_ri[..., 0], enh_ri[..., 1])  # (B, F, T) complex
        output = torch.istft(
            enh, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len,
            window=window,
        )  # (B, L)

        if output.shape[1] < n_samples:
            output = torch.nn.functional.pad(output, (0, n_samples - output.shape[1]))
        elif output.shape[1] > n_samples:
            output = output[:, :n_samples]
        return output


def build_se_model(n_fft=SE_N_FFT, hop_len=SE_HOP, win_len=SE_WIN) -> STFTWrapper:
    """Fresh STFTWrapper(DeepVQE()) with default STFT params."""
    return STFTWrapper(DeepVQE(), n_fft=n_fft, hop_len=hop_len, win_len=win_len)
