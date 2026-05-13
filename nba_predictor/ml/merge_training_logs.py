"""
Join prediction CSV + outcomes CSV for a single league → labeled Parquet for training.

Shared by ``merge_training_data_nba`` and ``merge_training_data_wnba``.
"""

from __future__ import annotations

import pandas as pd

from utils.data_io import PATH_MERGED, PATH_MERGED_WNBA, ensure_dirs, load_outcomes, load_predictions


def merge_league_logs(league: str) -> pd.DataFrame:
    """
    Inner-join predictions to outcomes for ``league`` (``\"nba\"`` or ``\"wnba\"``).

    Uses ``[\"game_id\", \"league\"]`` when both sides carry ``league``; otherwise ``game_id`` only.
    """
    lg = league.strip().lower()
    if lg not in ("nba", "wnba"):
        raise ValueError(f"unsupported league: {league!r}")

    pred = load_predictions()
    out = load_outcomes()

    if not pred.empty:
        pred = pred.copy()
        if "league" not in pred.columns:
            pred["league"] = "nba"
        pred["league"] = pred["league"].fillna("nba").astype(str).str.lower()
        pred = pred.loc[pred["league"] == lg].copy()
    if not out.empty:
        out = out.copy()
        if "league" not in out.columns:
            out["league"] = "nba"
        out["league"] = out["league"].fillna("nba").astype(str).str.lower()
        out = out.loc[out["league"] == lg].copy()

    if pred.empty or out.empty:
        return pd.DataFrame()

    if "logged_at" in pred.columns:
        pred = pred.sort_values("logged_at").drop_duplicates(subset=["game_id"], keep="last")
    else:
        pred = pred.drop_duplicates(subset=["game_id"], keep="last")

    merge_keys: list[str] = ["game_id"]
    if "league" in pred.columns and "league" in out.columns:
        merge_keys = ["game_id", "league"]

    m = pred.merge(out, on=merge_keys, how="inner", suffixes=("", "_oc"))
    ensure_dirs()
    dest = PATH_MERGED if lg == "nba" else PATH_MERGED_WNBA
    m.to_parquet(dest, index=False)
    return m
