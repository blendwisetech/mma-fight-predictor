"""
Ordered tabular features for NBA game-level models.

Focus: efficiency margins (NET, TS), pace context, record strength, fatigue.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

GAME_FEATURE_NAMES: list[str] = [
    "f_home_net",
    "f_away_net",
    "f_net_diff",
    "f_home_off",
    "f_away_off",
    "f_off_diff",
    "f_away_def",
    "f_home_def",
    "f_def_diff",
    "f_home_pace",
    "f_away_pace",
    "f_pace_mean",
    "f_home_ts",
    "f_away_ts",
    "f_ts_diff",
    "f_home_wpct",
    "f_away_wpct",
    "f_wpct_diff",
    "f_home_reb_pct",
    "f_away_reb_pct",
    "f_reb_pct_diff",
    "f_home_tov_pct",
    "f_away_tov_pct",
    "f_tov_pct_diff",
    "f_home_b2b",
    "f_away_b2b",
    "f_b2b_diff",
    "f_home_rest_days",
    "f_away_rest_days",
    "f_rest_diff",
    "f_home_inj_load",
    "f_away_inj_load",
    "f_inj_adv_home",
    "f_neutral",
]


def _f(row: pd.Series, key: str, default: float = float("nan")) -> float:
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def enriched_row_to_feature_vector(row: pd.Series) -> dict[str, float]:
    hn = _f(row, "home_team_NET_RATING")
    an = _f(row, "away_team_NET_RATING")
    ho = _f(row, "home_team_OFF_RATING")
    ao = _f(row, "away_team_OFF_RATING")
    hd = _f(row, "home_team_DEF_RATING")
    ad = _f(row, "away_team_DEF_RATING")
    hp = _f(row, "home_team_PACE")
    ap = _f(row, "away_team_PACE")
    hts = _f(row, "home_team_TS_PCT")
    ats = _f(row, "away_team_TS_PCT")
    hwp = _f(row, "home_team_W_PCT")
    awp = _f(row, "away_team_W_PCT")
    hr = _f(row, "home_team_REB_PCT")
    ar = _f(row, "away_team_REB_PCT")
    htov = _f(row, "home_team_TM_TOV_PCT")
    atov = _f(row, "away_team_TM_TOV_PCT")
    hb = 1.0 if bool(row.get("home_b2b")) else 0.0
    ab = 1.0 if bool(row.get("away_b2b")) else 0.0
    neu = 1.0 if bool(row.get("neutral_site")) else 0.0
    hrest = _f(row, "home_rest_days")
    arest = _f(row, "away_rest_days")
    hinj = _f(row, "home_injury_load", 0.0)
    ainj = _f(row, "away_injury_load", 0.0)
    if not np.isfinite(hinj):
        hinj = 0.0
    if not np.isfinite(ainj):
        ainj = 0.0
    pace_mean = float((hp + ap) / 2.0) if np.isfinite(hp) and np.isfinite(ap) else float("nan")
    return {
        "f_home_net": hn,
        "f_away_net": an,
        "f_net_diff": _diff(hn, an),
        "f_home_off": ho,
        "f_away_off": ao,
        "f_off_diff": _diff(ho, ao),
        "f_away_def": ad,
        "f_home_def": hd,
        "f_def_diff": _diff(ad, hd),
        "f_home_pace": hp,
        "f_away_pace": ap,
        "f_pace_mean": pace_mean,
        "f_home_ts": hts,
        "f_away_ts": ats,
        "f_ts_diff": _diff(hts, ats),
        "f_home_wpct": hwp,
        "f_away_wpct": awp,
        "f_wpct_diff": _diff(hwp, awp),
        "f_home_reb_pct": hr,
        "f_away_reb_pct": ar,
        "f_reb_pct_diff": _diff(hr, ar),
        "f_home_tov_pct": htov,
        "f_away_tov_pct": atov,
        "f_tov_pct_diff": _diff(atov, htov),
        "f_home_b2b": hb,
        "f_away_b2b": ab,
        "f_b2b_diff": float(ab - hb),
        "f_home_rest_days": hrest,
        "f_away_rest_days": arest,
        "f_rest_diff": _diff(hrest, arest),
        "f_home_inj_load": hinj,
        "f_away_inj_load": ainj,
        "f_inj_adv_home": float(ainj - hinj),
        "f_neutral": neu,
    }


def dataframe_X(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in GAME_FEATURE_NAMES})
    return out.astype(float)
