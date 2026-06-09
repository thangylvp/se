"""DiffLogMel must match the stock HF WhisperFeatureExtractor (CLAUDE.md invariant:
if it drifts, the training ASR gradient is computed on the wrong features)."""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def processor():
    try:
        return transformers.WhisperProcessor.from_pretrained("openai/whisper-small")
    except Exception as e:  # offline / not cached
        pytest.skip(f"whisper processor unavailable: {e}")


def test_diff_logmel_matches_hf(processor):
    from afe.asr.diff_logmel import DiffLogMel
    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal(16000 * 4)).astype(np.float32)

    hf = processor(wav, sampling_rate=16000, return_tensors="pt").input_features  # (1,80,3000)
    diff = DiffLogMel(processor, torch.device("cpu"))(torch.from_numpy(wav).unsqueeze(0))

    assert diff.shape == hf.shape == (1, 80, 3000)
    assert torch.allclose(diff, hf, atol=1e-4), \
        f"max abs diff {(diff - hf).abs().max().item():.2e}"
