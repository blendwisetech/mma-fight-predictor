"""
CSV / JSON paths for prediction logs, outcomes, and model registry (same pattern as baseball_predictor).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH_PREDICTIONS = ROOT / "data" / "predictions" / "game_predictions.csv"
PATH_OUTCOMES = ROOT / "data" / "outcomes" / "game_outcomes.csv"
PATH_MERGED = ROOT / "data" / "processed" / "training_games.parquet"
PATH_MERGED_WNBA = ROOT / "data" / "processed" / "training_games_wnba.parquet"
PATH_REGISTRY = ROOT / "data" / "models" / "registry.json"
PATH_EVAL_REPORT = ROOT / "data" / "processed" / "eval_report.json"
PATH_EVAL_REPORT_WNBA = ROOT / "data" / "processed" / "eval_report_wnba.json"


def ensure_dirs() -> None:
    for p in (
        PATH_PREDICTIONS.parent,
        PATH_OUTCOMES.parent,
        PATH_MERGED.parent,
        PATH_REGISTRY.parent,
    ):
        p.mkdir(parents=True, exist_ok=True)


def load_registry() -> dict[str, Any]:
    ensure_dirs()
    if not PATH_REGISTRY.exists():
        return {}
    return json.loads(PATH_REGISTRY.read_text(encoding="utf-8"))


def save_registry(data: dict[str, Any]) -> None:
    ensure_dirs()
    PATH_REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_predictions_df(df: pd.DataFrame) -> None:
    ensure_dirs()
    if PATH_PREDICTIONS.exists() and PATH_PREDICTIONS.stat().st_size > 0:
        old = pd.read_csv(PATH_PREDICTIONS)
        combined = pd.concat([old, df], ignore_index=True, sort=False)
    else:
        combined = df
    combined.to_csv(PATH_PREDICTIONS, index=False)


def load_predictions() -> pd.DataFrame:
    if not PATH_PREDICTIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(PATH_PREDICTIONS)


def load_outcomes() -> pd.DataFrame:
    if not PATH_OUTCOMES.exists():
        return pd.DataFrame()
    return pd.read_csv(PATH_OUTCOMES)


def upsert_outcomes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    ensure_dirs()
    old = load_outcomes()
    df2 = df.copy()
    if "league" not in df2.columns:
        df2["league"] = "nba"
    if not old.empty:
        if "league" not in old.columns:
            old = old.copy()
            old["league"] = "nba"
        all_df = pd.concat([old, df2], ignore_index=True)
    else:
        all_df = df2
    all_df["league"] = all_df["league"].fillna("nba").astype(str).str.lower()
    all_df = all_df.drop_duplicates(subset=["league", "game_id"], keep="last")
    all_df.to_csv(PATH_OUTCOMES, index=False)
    return len(df2)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_eval_report(payload: dict[str, Any], *, league: str = "nba") -> None:
    ensure_dirs()
    path = PATH_EVAL_REPORT_WNBA if league.strip().lower() == "wnba" else PATH_EVAL_REPORT
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_eval_report(*, league: str = "nba") -> dict[str, Any] | None:
    path = PATH_EVAL_REPORT_WNBA if league.strip().lower() == "wnba" else PATH_EVAL_REPORT
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
