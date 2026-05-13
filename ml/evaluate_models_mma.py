"""
Evaluate logged predictions and replay the production win model on merged rows.

Run: ``python -m ml.evaluate_models_mma``
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ml.calibration_utils import apply_marginal_shrink, apply_registry_tail_calibration, apply_temperature
from ml.feature_config_mma import GAME_FEATURE_NAMES, dataframe_X
from ml.time_split import resolve_sort_date, sort_by_time
from models.ml_predict_mma import raw_fighter_a_win_prob_batch
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_registry, save_eval_report


def _rolling_time_metrics(df: pd.DataFrame, y_col: str, p_col: str, n_bins: int = 5) -> list[dict[str, Any]]:
    if df.empty or y_col not in df.columns or p_col not in df.columns:
        return []
    dts, _ = resolve_sort_date(df)
    sub = df[[y_col, p_col]].copy()
    sub["_dt"] = pd.to_datetime(dts, errors="coerce")
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


def _replay_win_model(df: pd.DataFrame, bundle: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any] | None:
    if "fighter_a_win" not in df.columns:
        return None
    miss = [c for c in GAME_FEATURE_NAMES if c not in df.columns]
    if miss:
        return {"skipped": True, "reason": f"missing_feature_columns_{len(miss)}"}
    df_s = sort_by_time(df.reset_index(drop=True))
    X = dataframe_X(df_s)
    y = df_s["fighter_a_win"].astype(int).values
    try:
        raw = raw_fighter_a_win_prob_batch(bundle, X)
    except Exception as e:
        return {"skipped": True, "reason": f"predict_error:{e}"}
    iso = bundle.get("iso")
    T = float(meta.get("win_temperature", meta.get("temperature", 1.0)) or 1.0)
    lam = float(meta.get("win_marginal_lambda", 0.0) or 0.0)
    gamma = float(meta.get("win_marginal_gamma", 0.5) or 0.5)
    if iso is not None:
        p_iso = np.clip(iso.predict(raw), 1e-6, 1.0 - 1e-6)
    else:
        p_iso = np.clip(raw, 1e-6, 1.0 - 1e-6)
    p_ml = np.asarray(apply_temperature(p_iso, T), dtype=float).ravel()
    p_ml = np.asarray(apply_marginal_shrink(p_ml, lam, gamma), dtype=float).ravel()
    p_ml = np.clip(p_ml, 1e-6, 1.0 - 1e-6)
    w = float(meta.get("blend_weight_val", 0.0) or 0.0)
    thresh = float(meta.get("home_threshold_val", 0.5) or 0.5)
    if "pred_fighter_a_win_prob_heur" in df_s.columns:
        ph = df_s["pred_fighter_a_win_prob_heur"].astype(float).values
        ph = np.clip(np.nan_to_num(ph, nan=0.5), 1e-6, 1.0 - 1e-6)
        p = (1.0 - w) * p_ml + w * ph
    else:
        p = p_ml
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    reg = load_registry()
    p = np.asarray(apply_registry_tail_calibration(p, reg), dtype=float).ravel()
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    rep_out: dict[str, Any] = {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_tuned_threshold": float(accuracy_score(y, (p >= thresh).astype(int))),
    }
    tmp = df_s.copy()
    tmp["_p_replay"] = p
    rep_out["rolling_replay_by_time"] = _rolling_time_metrics(tmp, "fighter_a_win", "_p_replay")
    return rep_out


def main() -> None:
    ensure_dirs()
    if not PATH_MERGED.exists():
        print("No merged data.")
        return
    df = pd.read_parquet(PATH_MERGED)
    report: dict[str, Any] = {"n_rows": len(df)}

    if "fighter_a_win" in df.columns and "pred_fighter_a_win_prob" in df.columns:
        y = df["fighter_a_win"].astype(int)
        p = df["pred_fighter_a_win_prob"].astype(float).clip(1e-6, 1 - 1e-6)
        report["win_logged"] = {
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy_at_0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "rolling_by_time": _rolling_time_metrics(sort_by_time(df.copy()), "fighter_a_win", "pred_fighter_a_win_prob"),
        }

    reg = load_registry()
    win_path = (reg.get("production") or {}).get("win_model_path")
    if win_path and (ROOT / win_path).exists():
        try:
            bundle = joblib.load(ROOT / win_path)
            meta = bundle.get("meta") or {}
            rep = _replay_win_model(df, bundle, meta)
            if rep:
                report["win_model_replay_on_merged"] = rep
        except Exception as e:
            report["win_model_replay_error"] = str(e)

    save_eval_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
