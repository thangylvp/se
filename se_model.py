"""
Self-contained DeepVQE speech-enhancement model + checkpoint loader.

This wraps the vendored `deepvqe.DeepVQE` (identical to aecdns/models/deepvqe.py
and deepvqe/deepvqe.py, verified byte-for-byte) with the exact STFT front/back
end used by the teammate's training & inference pipeline (aecdns).

Pipeline (must match aecdns/models/modules/wrapper.py::STFTWrapper):
    waveform (B, L) @ 16 kHz
      -> torch.stft(n_fft=512, hop=256, win=512, periodic Hann, onesided, complex)
      -> view_as_real -> (B, F=257, T, 2)
      -> DeepVQE  -> (B, 257, T, 2)
      -> complex  -> torch.istft(same params)
      -> pad/trim to original length -> (B, L)

The checkpoint state_dict keys are prefixed `model.` because training saved an
`STFTWrapper` whose `self.model = DeepVQE()`. We strip that prefix and load the
DeepVQE body directly (matches aecdns/utils/checkpoint_utils.py::load_checkpoint).
"""
import torch
import torch.nn as nn

from deepvqe import DeepVQE


class STFTWrapper(nn.Module):
    """Waveform -> STFT -> DeepVQE -> iSTFT -> waveform. Mirrors aecdns STFTWrapper."""

    def __init__(self, model, n_fft=512, hop_len=256, win_len=512):
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


def load_se_model(checkpoint_path, device="cpu", n_fft=512, hop_len=256, win_len=512):
    """Build STFTWrapper(DeepVQE()) and load a `.tar` checkpoint into it.

    Handles both the full training checkpoint ({'model': sd, 'epoch': ...}) and a
    raw state_dict, stripping the `model.` prefix so the DeepVQE body loads cleanly.
    """
    device = torch.device(device)
    model = STFTWrapper(DeepVQE(), n_fft=n_fft, hop_len=hop_len, win_len=win_len)

    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}

    # strict=True: any mismatch means the architecture is wrong — we want it to fail loud.
    model.model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    return model
