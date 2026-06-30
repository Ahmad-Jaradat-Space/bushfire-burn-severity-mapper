"""Tracking must be safe whether or not MLflow is installed/active."""

from src.utils import tracking


def test_no_active_run_is_noop():
    # No start_run context -> every call is a harmless no-op.
    tracking.log_params({"a": 1, "nested": {"b": 2.0}})
    tracking.log_metrics({"x": 0.5, "nan_skipped": float("nan")})
    tracking.log_artifact("/no/such/file.txt")


def test_flatten_keeps_scalars_only():
    flat = tracking._flatten({"a": 1, "d": {"b": "s", "c": [1, 2]}, "n": None})
    assert flat == {"a": 1, "d.b": "s", "n": None}


def test_start_run_yields_context():
    import pytest

    pytest.importorskip("mlflow")
    with tracking.start_run(run_name="unit-test", experiment="unit") as run:
        # either a real run (mlflow present) or None (degraded) — both fine
        tracking.log_params({"k": "v"})
        tracking.log_metrics({"m": 1.0}, step=0)
    assert run is None or hasattr(run, "info")
