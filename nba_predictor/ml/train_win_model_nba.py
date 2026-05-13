"""
Train a simple home-win classifier from ``data/processed/training_games.parquet``.

Run from ``nba_predictor``:
  python -m ml.merge_training_data_nba
  python -m ml.train_win_model_nba
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
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_registry, save_registry


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
    p = Path(PATH_MERGED)
    if not p.exists():
        print("No merged training file. Log predictions, ingest outcomes, run merge_training_data_nba.")
        return
    df0 = pd.read_parquet(p)
    if "home_win" not in df0.columns:
        print("Merged data missing home_win label.")
        return
    df = df0.dropna(subset=["home_win"]).copy()
    if "logged_at" in df.columns:
        df["_sort"] = pd.to_datetime(df["logged_at"], errors="coerce")
        df = df.sort_values("_sort", na_position="last").drop(columns=["_sort"], errors="ignore")
    elif "slate_date" in df.columns:
        df["_sort"] = pd.to_datetime(df["slate_date"], errors="coerce")
        df = df.sort_values("_sort", na_position="last").drop(columns=["_sort"], errors="ignore")

    reg = load_registry()
    min_rows = int(reg.get("training", {}).get("min_rows_win", 40))
    if len(df) < min_rows:
        print(f"Only {len(df)} rows; need >= {min_rows}. Log more predictions + ingest outcomes.")
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
        "rows": int(len(df)),
        "val_logloss": float(log_loss(y_val, p_val)),
        "val_brier": float(brier_score_loss(y_val, p_val)),
        "val_auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else None,
    }
    print(json.dumps(metrics, indent=2))

    model_dir = ROOT / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rel = "data/models/nba_win_logreg.joblib"
    bundle = {"pipeline": pipe, "version": "logreg_nba_v1", "metrics": metrics}
    joblib.dump(bundle, ROOT / rel)

    reg = load_registry()
    prod = reg.setdefault("production", {})
    prod["win_model_path"] = rel
    prod["win_model_version"] = "logreg_nba_v1"
    prod["notes"] = "Trained from merged prediction+outcome logs; tune blend in Streamlit registry."
    save_registry(reg)
    print(f"Wrote {rel} and updated registry.")


if __name__ == "__main__":
    main()
