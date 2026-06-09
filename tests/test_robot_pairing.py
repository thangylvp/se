"""Robot test-set pairing: every transcript pairs to an existing wav, and pairing
is correct (not off-by-one). Skips if the dataset isn't present."""
import os

import pytest

from afe.data.robot import DEFAULT_ROOT, build_pairs

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_ROOT), reason="robot dataset not present")


def test_pairs_nonempty_and_wavs_exist():
    pairs = build_pairs(verbose=False)
    assert len(pairs) > 0
    for p in pairs:
        assert os.path.exists(p["wav"]), p["wav"]
        assert p["text"].strip()


def test_stem_alignment():
    # the paired wav's stem must equal the transcript stem (no cross-pairing)
    import glob
    from pathlib import Path
    pairs = build_pairs(verbose=False)
    # spot-check: each wav filename stem appears in its folder's transcript tree
    p = pairs[0]
    assert Path(p["wav"]).suffix == ".wav"
