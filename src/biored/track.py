# src/biored/track.py
import json
import os
import subprocess
import time
from pathlib import Path

import mlflow


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or os.environ.get("BIORED_COMMIT", "unknown")


def log_run(
    name: str, params: dict, metrics: dict, out_dir: str = "results/runs"
) -> None:
    record = {
        "name": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": git_commit(),
        "params": params,
        "metrics": metrics,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(record, indent=2))
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
