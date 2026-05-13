"""
Train a home-win classifier from ``data/processed/training_games_wnba.parquet``.

Same feature layout as NBA (advanced-team slice + rest + ESPN injury load).

Run from ``nba_predictor``:
  python -m ml.merge_training_data_wnba
  python -m ml.train_win_model_wnba
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.feature_config_nba import dataframe_X, enriched_row_to_feature_vector
from utils.data_io import PATH_MERGED_WNBA, ROOT, ensure_dirs, load_registry, save_registry


def _build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append(enriched_row_to_feature_vector(r))
    X = pd.DataFrame(rows)
    X = dataframe_X(X)
    y = df["home_win"].astype(int).values
    return X, y


def main() -> None:
    ensure_dirs()
    p = Path(PATH_MERGED_WNBA)
    if not p.exists():
        print(
            "No merged WNBA training file. Log WNBA predictions, fetch WNBA outcomes, "
            "run merge_training_data_wnba."
        )
        return
    df0 = pd.read_parquet(p)
    if "home_win" not in df0.columns:
        print("Merged WNBA data missing home_win label.")
        return
    df = df0.dropna(subset=["home_win"]).copy()
    if "logged_at" in df.columns:
        df["_sort"] = pd.to_datetime(df["logged_at"], errors="coerce")
        df = df.sort_values("_sort", na_position="last").drop(columns=["_sort"], errors="ignore")
    elif "slate_date" in df.columns:
        df["_sort"] = pd.to_datetime(df["slate_date"], errors="coerce")
        df = df.sort_values("_sort", na_position="last").drop(columns=["_sort"], errors="ignore")

    reg = load_registry()
    train_cfg = reg.get("training") or {}
    min_rows = int(train_cfg.get("min_rows_win_wnba", train_cfg.get("min_rows_win", 40)))

    if len(df) < min_rows:
        print(f"Only {len(df)} WNBA rows; need >= {min_rows}. Log more + ingest outcomes.")
        return

    X, y = _build_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=200, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    p_val = pipe.predict_proba(X_val)[:, 1]
    metrics = {
        "league": "wnba",
        "rows": int(len(df)),
        "val_logloss": float(log_loss(y_val, p_val)),
        "val_brier": float(brier_score_loss(y_val, p_val)),
        "val_auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else None,
    }
    print(json.dumps(metrics, indent=2))

    model_dir = ROOT / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rel = "data/models/wnba_win_logreg.joblib"
    bundle = {"pipeline": pipe, "version": "logreg_wnba_v1", "metrics": metrics}
    joblib.dump(bundle, ROOT / rel)

    reg = load_registry()
    prod = reg.setdefault("production", {})
    prod["win_model_path_wnba"] = rel
    prod["win_model_version_wnba"] = "logreg_wnba_v1"
    prod["notes_wnba"] = "Trained from WNBA prediction+outcome logs; tune ml_blend_weight_wnba in Streamlit."
    save_registry(reg)
    print(f"Wrote {rel} and updated registry (production.win_model_path_wnba).")


if __name__ == "__main__":
    main()
