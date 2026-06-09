"""Text normalization + WER/CER metrics."""
from afe.utils.text import cer, vi_norm, wer, wer_cer_breakdown


def test_vi_norm_keeps_diacritics_strips_punct():
    assert vi_norm("Xin chào, Việt Nam!") == "xin chào việt nam"
    assert vi_norm("HÀ\xa0NỘI.") == "hà nội"


def test_wer_cer_basic():
    assert wer(["a b c"], ["a b d"], vi_norm) == 1 / 3
    assert abs(cer(["abc"], ["abd"], vi_norm) - 1 / 3) < 1e-9


def test_empty_ref_dropped():
    # ref normalizes to empty -> pair dropped, remaining pair is perfect
    assert wer(["", "a b"], ["x", "a b"], vi_norm) == 0.0


def test_breakdown_counts():
    b = wer_cer_breakdown(["a b c"], ["a b d"], vi_norm)
    assert b["n"] == 1 and b["H"] == 2 and b["S"] == 1 and b["ref_w"] == 3
