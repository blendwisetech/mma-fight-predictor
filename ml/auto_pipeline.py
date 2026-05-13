"""
Batch maintenance: merge predictions with outcomes, retrain, evaluate.

Run from ``mma_predictor``: ``python -m ml.auto_pipeline``
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_MODULES = [
    "ml.merge_training_data_mma",
    "ml.train_win_model_mma",
    "ml.evaluate_models_mma",
]


def run_pipeline() -> list[tuple[str, int, str]]:
    results: list[tuple[str, int, str]] = []
    for mod in PIPELINE_MODULES:
        r = subprocess.run(
            [sys.executable, "-m", mod],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
        text = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        results.append((mod, int(r.returncode), text))
    return results


def append_pipeline_log(results: list[tuple[str, int, str]]) -> None:
    log_path = ROOT / "data" / "processed" / "auto_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = [f"\n--- {stamp} ---"]
    for mod, code, text in results:
        block.append(f"{mod}  exit={code}")
        block.append(text[:4000] if text else "(no output)")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")


def main() -> None:
    res = run_pipeline()
    append_pipeline_log(res)
    for mod, code, _ in res:
        print(f"{mod} -> {code}")


if __name__ == "__main__":
    main()
