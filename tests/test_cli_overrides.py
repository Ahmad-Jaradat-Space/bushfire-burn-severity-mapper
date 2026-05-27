"""Verify that train scripts accept OmegaConf dot-overrides on the CLI.

This is the smoke-test for the M10 fan-out (`scripts/run_all_events.sh`)
which passes `experiment.split_mode=event_wise` as a positional argument.
"""
import sys

import pytest

torch = pytest.importorskip("torch")


def test_train_segmenter_accepts_overrides(monkeypatch):
    # Build args as if invoked from the CLI; signature parsing must accept them.
    from src.models.train_segmenter import main as seg_main
    monkeypatch.setattr(sys, "argv",
                        ["train_segmenter", "--config", "configs/experiments/smoke_unet.yaml",
                         "--fast-mode",
                         "experiment.name=cli_override_test", "train.batch_size=2"])
    # We don't actually invoke training — just verify the argparse accepts the
    # positional overrides without error. We monkey-patch the train function to
    # short-circuit the actual training step.
    from src.models import train_segmenter as ts
    captured = {}

    def fake_train(config_path, fast_mode=False, overrides=None):
        captured["config"] = config_path
        captured["fast_mode"] = fast_mode
        captured["overrides"] = overrides
        return {}

    monkeypatch.setattr(ts, "train", fake_train)
    seg_main()
    assert captured["fast_mode"] is True
    assert captured["overrides"] == ["experiment.name=cli_override_test", "train.batch_size=2"]


def test_train_rf_accepts_overrides(monkeypatch):
    from src.models import train_rf as tr
    captured = {}

    def fake_train_rf(config_path, overrides=None):
        captured["config"] = config_path
        captured["overrides"] = overrides
        return {}

    monkeypatch.setattr(tr, "train_rf", fake_train_rf)
    monkeypatch.setattr(sys, "argv",
                        ["train_rf", "--config", "configs/experiments/rf_multiclass.yaml",
                         "experiment.split_mode=event_wise"])
    tr.main()
    assert captured["overrides"] == ["experiment.split_mode=event_wise"]


def test_config_loader_applies_overrides():
    """Sanity that the loader actually applies overrides (not just parses them)."""
    from src.utils.config import load_config
    cfg = load_config(
        "configs/experiments/rf_multiclass.yaml",
        overrides=["experiment.split_mode=event_wise", "rf.n_estimators=42"],
    )
    assert cfg.experiment.split_mode == "event_wise"
    assert cfg.rf.n_estimators == 42
