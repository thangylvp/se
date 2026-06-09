"""SE model build + checkpoint round-trip. Forward must preserve waveform length."""
import os

import pytest
import torch

from afe.checkpoint.io import load_se_model, save_se_checkpoint
from afe.modeling.se_model import build_se_model

CKPT = "checkpoints/model_D.tar"


def test_forward_preserves_length():
    se = build_se_model().eval()
    x = torch.randn(1, 16000)
    with torch.inference_mode():
        y = se(x)
    assert y.shape == x.shape


def test_checkpoint_roundtrip(tmp_path):
    se = build_se_model()
    p = tmp_path / "rt.tar"
    save_se_checkpoint(se, str(p), step=5)
    se2 = load_se_model(str(p), device="cpu")
    for (k, a), (_, b) in zip(se.model.state_dict().items(), se2.model.state_dict().items()):
        assert torch.equal(a, b), k


@pytest.mark.skipif(not os.path.exists(CKPT), reason="model_D checkpoint not present")
def test_load_released_checkpoint():
    se = load_se_model(CKPT, device="cpu")
    assert sum(p.numel() for p in se.parameters()) == 7509996
