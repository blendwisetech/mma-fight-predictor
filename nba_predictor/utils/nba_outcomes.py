"""Build outcomes rows from a scoreboard day (Final games only)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from utils.nba_core import NBA_LEAGUE_ID, fetch_scoreboard_games


def scoreboard_to_outcomes(
    game_date: date,
    *,
    timeout: int = 120,
    league_id: str = NBA_LEAGUE_ID,
    league_slug: str = "nba",
) -> pd.DataFrame:
    df = fetch_scoreboard_games(game_date, timeout=timeout, league_id=league_id)
    if df.empty:
        return df
    mask = df["game_status_text"].astype(str).str.lower() == "final"
    df = df.loc[mask].copy()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        hs = int(r["home_score"])
        aw = int(r["away_score"])
        rows.append(
            {
                "game_id": str(r["game_id"]),
                "league": league_slug,
                "home_score": hs,
                "away_score": aw,
                "home_win": int(hs > aw),
            }
        )
    return pd.DataFrame(rows)
