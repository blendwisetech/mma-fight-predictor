"""
Evaluate logged home-win probabilities on merged rows; optional sklearn replay.

Run from ``nba_predictor``: ``python -m ml.evaluate_models_nba``
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ml.feature_config_nba import dataframe_X, enriched_row_to_feature_vector
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_registry, save_eval_report


def _rolling_time_metrics(df: pd.DataFrame, y_col: str, p_col: str, n_bins: int = 5) -> list[dict[str, Any]]:
    if df.empty or y_col not in df.columns or p_col not in df.columns:
        return []
    sub = df[[y_col, p_col]].copy()
    if "logged_at" in df.columns:
        sub["_dt"] = pd.to_datetime(df["logged_at"], errors="coerce")
    elif "slate_date" in df.columns:
        sub["_dt"] = pd.to_datetime(df["slate_date"], errors="coerce")
    else:
        return []
    sub = sub.dropna(subset=["_dt"])
    if len(sub) < n_bins * 3:
        return []
    try:
        sub["_bin"] = pd.qcut(sub["_dt"], q=n_bins, duplicates="drop")
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for name, grp in sub.groupby("_bin", observed=True):
        if len(grp) < 3:
            continue
        y = grp[y_col].astype(int)
        p = grp[p_col].astype(float).clip(1e-6, 1 - 1e-6)
        out.append(
            {
                "time_bin": str(name),
                "n": int(len(grp)),
                "brier": float(brier_score_loss(y, p)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
            }
        )
    return out


def _replay_sklearn(df: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any] | None:
    if "home_win" not in df.columns or bundle.get("pipeline") is None:
        return None
    rows: list[dict[str, float]] = []
    for _, r in df.iterrows():
        try:
            rows.append(enriched_row_to_feature_vector(r))
        except Exception:
            return {"skipped": True, "reason": "feature_vector_error"}
    X = dataframe_X(pd.DataFrame(rows))
    try:
        p = np.clip(bundle["pipeline"].predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    except Exception as e:
        return {"skipped": True, "reason": f"predict_error:{e}"}
    y = df["home_win"].astype(int).values
    tmp = df.copy()
    tmp["_p_replay"] = p
    return {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "rolling_by_time": _rolling_time_metrics(tmp, "home_win", "_p_replay"),
    }


def main() -> None:
    ensure_dirs()
    if not PATH_MERGED.exists():
        print("No merged data.")
        return
    df = pd.read_parquet(PATH_MERGED)
    report: dict[str, Any] = {"n_rows": int(len(df))}

    if "home_win" in df.columns and "pred_home_win_prob" in df.columns:
        y = df["home_win"].astype(int)
        p = df["pred_home_win_prob"].astype(float).clip(1e-6, 1 - 1e-6)
        report["win_logged"] = {
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "rolling_by_time": _rolling_time_metrics(df.copy(), "home_win", "pred_home_win_prob"),
        }

    reg = load_registry()
    win_path = (reg.get("production") or {}).get("win_model_path")
    if win_path and (ROOT / win_path).exists():
        try:
            raw = joblib.load(ROOT / win_path)
            bundle = raw if isinstance(raw, dict) else {"pipeline": raw}
            rep = _replay_sklearn(df, bundle)
            if rep:
                report["win_model_replay_on_merged"] = rep
        except Exception as e:
            report["win_model_replay_error"] = str(e)

    save_eval_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
