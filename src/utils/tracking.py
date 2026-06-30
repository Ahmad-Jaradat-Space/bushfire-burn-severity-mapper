"""Experiment tracking that degrades to a no-op when MLflow is absent.

Every function here is safe to call whether or not ``mlflow`` is installed or a
tracking server is configured — so the training scripts can log params, metrics
and artifacts without taking a hard dependency, and CI (which has neither)
keeps running unchanged. Point it at a server with ``MLFLOW_TRACKING_URI``;
otherwise it logs to a local ``outputs/mlruns`` file store.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path

from src.utils.geo import REPO_ROOT


def _mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
    return out


def _default_uri() -> str:
    # MLflow 3 deprecated the plain file store; default to a sqlite backend.
    return (
        os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'outputs' / 'mlflow.db'}"
    )


@contextmanager
def start_run(run_name: str | None = None, experiment: str = "burn-severity"):
    mf = _mlflow()
    if mf is None:
        yield None
        return
    try:
        mf.set_tracking_uri(_default_uri())
        mf.set_experiment(experiment)
        with mf.start_run(run_name=run_name) as run:
            yield run
    except Exception as exc:  # tracking must never break the pipeline
        import logging

        logging.getLogger(__name__).warning("MLflow tracking disabled: %s", exc)
        yield None


def log_params(params: dict) -> None:
    mf = _mlflow()
    if mf is None or not mf.active_run():
        return
    try:
        mf.log_params(_flatten(params))
    except Exception:
        pass


def log_metrics(metrics: dict, step: int | None = None) -> None:
    mf = _mlflow()
    if mf is None or not mf.active_run():
        return
    clean = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not math.isnan(float(v))
    }
    try:
        if clean:
            mf.log_metrics(clean, step=step)
    except Exception:
        pass


def log_artifact(path: str | Path) -> None:
    mf = _mlflow()
    if mf is None or not mf.active_run() or not Path(path).exists():
        return
    try:
        mf.log_artifact(str(path))
    except Exception:
        pass
