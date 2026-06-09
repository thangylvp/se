"""Config dataclass + YAML loader."""
import pytest

from afe.config.defaults import EvalConfig, TrainConfig, load_config


def test_defaults():
    c = TrainConfig()
    assert c.lambda_ce == 1.0 and c.lambda_se == 0.05 and c.lambda_feat == 0.0


def test_yaml_load_and_override(tmp_path):
    y = tmp_path / "t.yaml"
    y.write_text("steps: 1234\nlambda_feat: 2.0\n")
    c = load_config(TrainConfig, str(y))
    assert c.steps == 1234 and c.lambda_feat == 2.0
    c2 = load_config(TrainConfig, str(y), steps=7)  # CLI override wins
    assert c2.steps == 7


def test_unknown_key_rejected(tmp_path):
    y = tmp_path / "bad.yaml"
    y.write_text("nonsense_key: 1\n")
    with pytest.raises(ValueError):
        load_config(TrainConfig, str(y))


def test_eval_defaults():
    e = EvalConfig()
    assert e.n_utts == 200 and e.seed == 1234 and e.snrs[0] == -10
