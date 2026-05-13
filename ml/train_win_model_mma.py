"""
Train **Fighter A** win classifier (chronological splits + isotonic + temperature + marginal shrink).

Run from ``mma_predictor``: ``python -m ml.train_win_model_mma``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from ml.calibration_utils import apply_marginal_shrink, apply_temperature, tune_marginal_shrink_brier, tune_temperature_brier
from ml.feature_config_mma import GAME_FEATURE_NAMES, dataframe_X
from ml.time_split import sort_by_time, time_train_val_test_masks
from ml.win_prob_utils import tune_blend_weight, tune_home_threshold
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_registry, save_registry


def _build_base_pipeline(*, n_train_rows: int) -> Pipeline:
    hgb_kw: dict = dict(
        random_state=42,
        max_depth=6,
        max_iter=400,
        learning_rate=0.06,
        min_samples_leaf=15,
        l2_regularization=1.0,
    )
    if n_train_rows >= 150:
        hgb_kw["early_stopping"] = True
        hgb_kw["validation_fraction"] = 0.12
        hgb_kw["n_iter_no_change"] = 25
    else:
        hgb_kw["early_stopping"] = False
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced", **hgb_kw)),
        ]
    )


def main() -> None:
    ensure_dirs()
    if not PATH_MERGED.exists():
        print("No merged training file. Run: python -m ml.seed_training_from_history_mma")
        return
    df0 = pd.read_parquet(PATH_MERGED)
    if "fighter_a_win" not in df0.columns:
        print("Merged data missing fighter_a_win label.")
        return
    df = df0.dropna(subset=["fighter_a_win"]).copy()
    df = sort_by_time(df)
    reg = load_registry()
    reg.setdefault("training", {})
    reg["training"].setdefault("min_rows_win", 400)
    min_rows = int(reg["training"]["min_rows_win"])
    if len(df) < min_rows:
        print(f"Only {len(df)} rows; need >= {min_rows}. Seed or merge more outcomes.")
        return

    n = len(df)
    train_m, val_m, test_m = time_train_val_test_masks(n)
    X = dataframe_X(df)
    y = df["fighter_a_win"].astype(int).values

    va_ix = np.flatnonzero(val_m)
    te_ix = np.flatnonzero(test_m)
    X_va, y_va = X.iloc[va_ix].values, y[val_m]
    X_te, y_te = X.iloc[te_ix].values, y[test_m]

    fit_m = train_m | val_m
    fit_ix = np.flatnonzero(fit_m)
    X_fit = X.iloc[fit_ix].values
    y_fit = y[fit_m]
    base_final = _build_base_pipeline(n_train_rows=len(fit_ix))
    base_final.fit(X_fit, y_fit)

    p_va2 = base_final.predict_proba(X_va)[:, 1]

    iso_final: IsotonicRegression | None = None
    if val_m.sum() >= 8 and len(np.unique(y_va)) >= 2:
        try:
            iso_final = IsotonicRegression(out_of_bounds="clip")
            iso_final.fit(p_va2, y_va)
        except Exception:
            iso_final = None

    def calibrate_final(raw: np.ndarray) -> np.ndarray:
        if iso_final is None:
            return np.clip(raw, 1e-6, 1.0 - 1e-6)
        return np.clip(iso_final.predict(raw), 1e-6, 1.0 - 1e-6)

    gamma_a = float(np.mean(y[train_m])) if train_m.sum() > 0 else 0.5

    p_va_iso = calibrate_final(p_va2)
    brier_va_pre_temp = float(brier_score_loss(y_va, np.clip(p_va_iso, 1e-6, 1.0 - 1e-6)))

    T = 1.0
    if val_m.sum() >= 8 and len(np.unique(y_va)) >= 2:
        T = tune_temperature_brier(p_va_iso, y_va)
    T_floor = float(reg.get("training", {}).get("win_temperature_floor", 1.0))
    T = max(float(T), T_floor)
    p_va_ml = np.asarray(apply_temperature(p_va_iso, T), dtype=float).ravel()
    brier_va_post_temp = float(brier_score_loss(y_va, np.clip(p_va_ml, 1e-6, 1.0 - 1e-6)))

    lam_shrink = 0.0
    if val_m.sum() >= 8 and len(np.unique(y_va)) >= 2:
        lam_shrink = tune_marginal_shrink_brier(p_va_ml, y_va, gamma_a)
    lam_floor = float(reg.get("training", {}).get("win_marginal_lambda_floor", 0.08))
    lam_shrink = max(lam_shrink, lam_floor)
    p_va_ml = np.asarray(apply_marginal_shrink(p_va_ml, lam_shrink, gamma_a), dtype=float).ravel()
    brier_va_post_shrink = float(brier_score_loss(y_va, np.clip(p_va_ml, 1e-6, 1.0 - 1e-6)))

    p_heur_va = None
    if "pred_fighter_a_win_prob_heur" in df.columns:
        p_heur_va = df.iloc[va_ix]["pred_fighter_a_win_prob_heur"].astype(float).values
        if not np.isfinite(p_heur_va).any():
            p_heur_va = None

    blend_w = tune_blend_weight(p_va_ml, p_heur_va, y_va)
    if p_heur_va is None:
        p_va_blend = p_va_ml.copy()
    else:
        ph = np.clip(np.nan_to_num(p_heur_va, nan=0.5), 1e-6, 1.0 - 1e-6)
        p_va_blend = (1.0 - blend_w) * p_va_ml + blend_w * ph
    thresh = tune_home_threshold(p_va_blend, y_va)

    p_te_raw_f = base_final.predict_proba(X_te)[:, 1]
    p_te_iso = calibrate_final(p_te_raw_f)
    p_te_ml = np.asarray(apply_temperature(p_te_iso, T), dtype=float).ravel()
    p_te_ml = np.asarray(apply_marginal_shrink(p_te_ml, lam_shrink, gamma_a), dtype=float).ravel()
    p_heur_te = None
    if "pred_fighter_a_win_prob_heur" in df.columns:
        p_heur_te = df.iloc[te_ix]["pred_fighter_a_win_prob_heur"].astype(float).values
        if not np.isfinite(p_heur_te).any():
            p_heur_te = None
    if p_heur_te is None:
        p_te_blend = p_te_ml
    else:
        ph = np.clip(np.nan_to_num(p_heur_te, nan=0.5), 1e-6, 1.0 - 1e-6)
        p_te_blend = (1.0 - blend_w) * p_te_ml + blend_w * ph

    p_te_clip = np.clip(p_te_blend, 1e-6, 1.0 - 1e-6)
    bri_te = float(brier_score_loss(y_te, p_te_clip))
    ll_te = float(log_loss(y_te, p_te_clip, labels=[0, 1]))
    acc_te = float(accuracy_score(y_te, (p_te_clip >= thresh).astype(int)))

    out_path = ROOT / "data" / "models" / "win_mma.joblib"
    meta = {
        "type": "sklearn_hgbm_fighter_a_win_calibrated",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "features": GAME_FEATURE_NAMES,
        "time_split": {"train": int(train_m.sum()), "val": int(val_m.sum()), "test": int(test_m.sum())},
        "test_brier": bri_te,
        "test_log_loss": ll_te,
        "test_accuracy_at_tuned_threshold": acc_te,
        "blend_weight_val": blend_w,
        "home_threshold_val": thresh,
        "calibrator": "isotonic" if iso_final is not None else "none",
        "win_temperature": T,
        "win_temperature_floor": T_floor,
        "win_marginal_lambda": lam_shrink,
        "win_marginal_gamma": gamma_a,
        "win_marginal_lambda_floor": lam_floor,
        "val_brier_pre_temperature": brier_va_pre_temp,
        "val_brier_post_temperature": brier_va_post_temp,
        "val_brier_post_shrink": brier_va_post_shrink,
    }
    bundle = {
        "base_pipeline": base_final,
        "iso": iso_final,
        "meta": meta,
    }
    joblib.dump(bundle, out_path)
    print(json.dumps(meta, indent=2))

    reg.setdefault("production", {})
    reg["production"]["win_model_path"] = str(out_path.relative_to(ROOT)).replace("\\", "/")
    reg["production"]["win_model_version"] = f"hgbm_mma_a_{meta['trained_at'][:10]}"
    reg["production"]["win_blend_weight"] = blend_w
    reg["production"]["fighter_a_win_threshold"] = thresh
    reg["production"]["home_win_threshold"] = thresh
    reg["production"]["fighter1_win_threshold"] = thresh
    reg["production"]["win_temperature"] = T
    reg["production"]["win_marginal_lambda"] = lam_shrink
    reg["production"]["win_marginal_gamma"] = gamma_a
    if "event_date" in df.columns:
        reg["production"]["trained_on_through"] = str(df["event_date"].max())
    save_registry(reg)
    print(f"Updated registry win model -> {reg['production']['win_model_path']}")


if __name__ == "__main__":
    main()
