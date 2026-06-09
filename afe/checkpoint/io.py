"""Checkpoint IO for the SE model.

Checkpoints are saved as {"model": {"model.<k>": v}} because training saves an
STFTWrapper whose `self.model = DeepVQE()`. We strip/restore that `model.` prefix
so the DeepVQE body loads uniformly (matches aecdns checkpoint_utils).
"""
import torch

from afe.modeling.se_model import STFTWrapper, build_se_model


def load_se_model(checkpoint_path, device="cpu", **stft_kwargs) -> STFTWrapper:
    """Build STFTWrapper(DeepVQE()) and load a `.tar` checkpoint into it.

    Handles both a full training checkpoint ({'model': sd, 'step': ...}) and a
    raw state_dict, stripping the `model.` prefix. strict=True so an arch mismatch
    fails loud.
    """
    device = torch.device(device)
    model = build_se_model(**stft_kwargs)

    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}

    model.model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def save_se_checkpoint(se_model: STFTWrapper, path, **extra):
    """Save an STFTWrapper's DeepVQE body with the `model.` prefix (+ any extra keys)."""
    payload = {"model": {f"model.{k}": v for k, v in se_model.model.state_dict().items()}}
    payload.update(extra)
    torch.save(payload, path)
