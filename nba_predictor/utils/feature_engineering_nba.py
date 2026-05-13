"""
Per-game basketball feature rows (NBA or WNBA): team advanced season rates + fatigue flags.

Uses official **LeagueDashTeamStats** (Advanced) joined by ``TEAM_ID``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from utils.espn_injuries import fetch_injury_weights_by_team, injury_load_for_display
from utils.nba_core import (
    NBA_LEAGUE_ID,
    back_to_back_flags,
    batch_rest_days_before_slate,
    fetch_team_advanced_stats,
)


def _safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _attach_team_row(row: dict[str, Any], prefix: str, team_row: pd.Series | None) -> None:
    if team_row is None:
        return
    for col in (
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "E_OFF_RATING",
        "E_DEF_RATING",
        "E_NET_RATING",
        "PACE",
        "TS_PCT",
        "EFG_PCT",
        "REB_PCT",
        "AST_PCT",
        "TM_TOV_PCT",
        "W_PCT",
        "GP",
    ):
        if col in team_row.index:
            row[f"{prefix}_team_{col}"] = _safe_float(team_row[col])


def _league_rating_means(tbl: pd.DataFrame) -> dict[str, float]:
    means: dict[str, float] = {}
    for col in ("NET_RATING", "OFF_RATING", "DEF_RATING"):
        if col not in tbl.columns:
            continue
        s = pd.to_numeric(tbl[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        means[col] = float(s.mean())
    return means


def _apply_rating_stabilization(
    row: dict[str, Any],
    *,
    league_means: dict[str, float],
    expected_rating_blend: float,
    rating_shrink_gp_prior: float,
) -> None:
    """
    Pull noisy early-season ratings toward league average and (when present) tracking ``E_*`` estimates.

    Mutates ``home_team_*`` / ``away_team_*`` rating columns in ``row``.
    """
    eb = float(np.clip(expected_rating_blend, 0.0, 1.0))
    k = float(max(0.0, rating_shrink_gp_prior))

    pairs = [
        ("NET_RATING", "E_NET_RATING"),
        ("OFF_RATING", "E_OFF_RATING"),
        ("DEF_RATING", "E_DEF_RATING"),
    ]

    for prefix in ("home", "away"):
        gp_raw = row.get(f"{prefix}_team_GP")
        gpf = _safe_float(gp_raw)
        if not np.isfinite(gpf) or gpf <= 0:
            gpf = float("nan")

        for base_col, e_col in pairs:
            key = f"{prefix}_team_{base_col}"
            ekey = f"{prefix}_team_{e_col}"
            raw = _safe_float(row.get(key))
            if not np.isfinite(raw):
                continue
            adj = raw
            if eb > 0:
                ev = _safe_float(row.get(ekey))
                if np.isfinite(ev):
                    adj = (1.0 - eb) * raw + eb * ev
            mu = league_means.get(base_col)
            if k > 0 and np.isfinite(mu) and np.isfinite(gpf):
                w = gpf / (gpf + k)
                adj = w * adj + (1.0 - w) * mu
            row[key] = float(adj)


def enrich_games_nba(
    games: pd.DataFrame,
    season: str,
    slate_date: date,
    *,
    season_type: str = "Regular Season",
    timeout: int = 120,
    rest_scan_days: int = 7,
    league_id: str = NBA_LEAGUE_ID,
    injury_league: str = "nba",
    expected_rating_blend: float = 0.2,
    rating_shrink_gp_prior: float = 8.0,
) -> pd.DataFrame:
    """
    ``games``: output of ``fetch_scoreboard_games`` (non-empty).

    Adds home/away advanced stats, **rest days** since last game (calendar scan),
    legacy **B2B** flags, and **ESPN injury** load (Out/Doubtful weighted heavier).

    ``league_id`` selects NBA vs WNBA on stats.nba.com. ``injury_league`` is ``\"nba\"`` or ``\"wnba\"`` for ESPN’s JSON.

    ``expected_rating_blend`` mixes box-score ratings with ``E_*`` tracking estimates (0 = off).
    ``rating_shrink_gp_prior`` is an additive pseudo games prior shrinking toward league mean (0 = off).
    """
    tbl = fetch_team_advanced_stats(
        season, season_type=season_type, timeout=timeout, league_id=league_id
    )
    league_means = _league_rating_means(tbl) if not tbl.empty else {}
    if tbl.empty:
        by_id = pd.DataFrame()
    else:
        by_id = tbl.set_index("TEAM_ID")

    team_ids: set[int] = set()
    for _, g in games.iterrows():
        team_ids.add(int(g["home_team_id"]))
        team_ids.add(int(g["away_team_id"]))

    rest_map = batch_rest_days_before_slate(
        slate_date,
        team_ids,
        max_scan=max(2, int(rest_scan_days)),
        timeout=timeout,
        league_id=league_id,
    )

    inj_weights: dict[str, float] = {}
    try:
        inj_weights = fetch_injury_weights_by_team(
            timeout=min(timeout, 60), league=injury_league
        )
    except Exception:
        inj_weights = {}

    out_rows: list[dict[str, Any]] = []
    for _, g in games.iterrows():
        row: dict[str, Any] = g.to_dict()
        hid = int(g["home_team_id"])
        aid = int(g["away_team_id"])
        h_row = by_id.loc[hid] if hid in by_id.index else None
        a_row = by_id.loc[aid] if aid in by_id.index else None
        if h_row is not None and isinstance(h_row, pd.DataFrame):
            h_row = h_row.iloc[0]
        if a_row is not None and isinstance(a_row, pd.DataFrame):
            a_row = a_row.iloc[0]
        _attach_team_row(row, "home", h_row)
        _attach_team_row(row, "away", a_row)
        if league_means:
            _apply_rating_stabilization(
                row,
                league_means=league_means,
                expected_rating_blend=expected_rating_blend,
                rating_shrink_gp_prior=rating_shrink_gp_prior,
            )
        hb, ab = back_to_back_flags(slate_date, hid, aid, timeout=timeout, league_id=league_id)
        row["home_b2b"] = bool(hb)
        row["away_b2b"] = bool(ab)
        row["neutral_site"] = bool(g.get("neutral"))

        hr = rest_map.get(hid)
        ar = rest_map.get(aid)
        row["home_rest_days"] = float(hr) if hr is not None else float("nan")
        row["away_rest_days"] = float(ar) if ar is not None else float("nan")

        hdisp = str(row.get("home_display") or "")
        adisp = str(row.get("away_display") or "")
        row["home_injury_load"] = float(injury_load_for_display(hdisp, inj_weights))
        row["away_injury_load"] = float(injury_load_for_display(adisp, inj_weights))

        out_rows.append(row)

    return pd.DataFrame(out_rows)
