"""
Build per-game feature rows from schedule + MLB Stats API season aggregates.

FanGraphs / pybaseball leaders often return HTTP 403 for automated clients; MLB JSON is reliable.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from utils.mlb_season_stats import pitcher_row_for_model, team_hitting_mlb_table
from utils.mlb_team_context import attach_high_impact_context
from utils.park_factors import park_factors_for_venue
from utils.team_map import fg_abbr_from_mlb_name


def _safe_int(x: Any) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def enrich_games_with_features(games: pd.DataFrame, season: int, slate_date: date | None = None) -> pd.DataFrame:
    """
    Input: merge_schedule_with_probables output.
    Joins team offense by MLB team id; probable starters by MLB people id.
    Keeps home_fg / away_fg abbreviations for park-factor lookup only.
    """
    off = team_hitting_mlb_table(season)
    if off.empty:
        off_idx = pd.DataFrame({"team_id": []}).astype({"team_id": "int64"}).set_index("team_id")
    else:
        off_idx = off.set_index("team_id")
    out_rows: list[dict[str, Any]] = []

    for _, g in games.iterrows():
        row = g.to_dict()
        row["home_fg"] = fg_abbr_from_mlb_name(g.get("home_name"))
        row["away_fg"] = fg_abbr_from_mlb_name(g.get("away_name"))

        def attach_off(prefix: str, tid_raw: Any) -> None:
            tid = _safe_int(tid_raw)
            if tid is None or tid not in off_idx.index:
                return
            s = off_idx.loc[tid]
            for col in ["wRC+", "OBP", "SLG", "OPS", "BB%", "K%", "Barrel%", "Hard%", "wOBA"]:
                val = s[col] if col in s.index else np.nan
                if pd.notna(val) and np.isscalar(val):
                    row[f"{prefix}_team_{col}"] = float(val)

        attach_off("home", g.get("home_id"))
        attach_off("away", g.get("away_id"))

        hp = pitcher_row_for_model(_safe_int(g.get("home_probable_id")), season)
        ap = pitcher_row_for_model(_safe_int(g.get("away_probable_id")), season)
        for label, pr in (("home_sp", hp), ("away_sp", ap)):
            for k, v in pr.items():
                row[f"{label}_{k}"] = v

        vid = _safe_int(g.get("venue_id"))
        abbr = str(row.get("home_fg") or "NYY")
        _hr_f, run_fac = park_factors_for_venue(vid, abbr)
        row["park_runs_factor"] = float(run_fac)

        out_rows.append(row)

    return attach_high_impact_context(pd.DataFrame(out_rows), season, slate_date)
