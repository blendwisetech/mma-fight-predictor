"""
NBA and WNBA schedules + team tables via **nba_api** (stats.nba.com JSON with browser-like headers).

Timeouts default high because the league endpoint can be slow from some networks.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, scoreboardv3

# stats.nba.com league ids (see nba_api ScoreboardV3 / LeagueDashTeamStats)
NBA_LEAGUE_ID = "00"
WNBA_LEAGUE_ID = "10"


def season_str_for_date(d: date) -> str:
    """NBA season label e.g. May 2026 → ``2025-26``."""
    y, m = d.year, d.month
    if m >= 10:
        return f"{y}-{str(y + 1)[2:]}"
    return f"{y - 1}-{str(y)[2:]}"


def season_str_for_date_wnba(d: date) -> str:
    """
    WNBA season label for **LeagueDashTeamStats** (calendar year of the campaign).

    May–December map to ``year``; January–April map to ``year - 1`` (off-season / tail).
    """
    y, m = d.year, d.month
    return str(y) if m >= 5 else str(y - 1)


def season_str_for_league(league_id: str, d: date) -> str:
    """Season string for advanced team tables (NBA vs WNBA naming differs on stats.nba.com)."""
    if league_id == WNBA_LEAGUE_ID:
        return season_str_for_date_wnba(d)
    return season_str_for_date(d)


def _full_team_name(team_blob: dict[str, Any]) -> str:
    city = str(team_blob.get("teamCity") or "").strip()
    name = str(team_blob.get("teamName") or "").strip()
    return f"{city} {name}".strip()


def fetch_scoreboard_games(
    game_date: date, *, timeout: int = 120, league_id: str = NBA_LEAGUE_ID
) -> pd.DataFrame:
    """
    One row per game from ScoreboardV3 for ``game_date`` (calendar day in North America;
    pass the date you care about in local planning — API uses ``YYYY-MM-DD``).
    """
    iso = game_date.isoformat()
    sb = scoreboardv3.ScoreboardV3(game_date=iso, league_id=league_id, timeout=timeout)
    payload = sb.get_dict()
    games = (payload.get("scoreboard") or {}).get("games") or []
    rows: list[dict[str, Any]] = []
    for g in games:
        home = g.get("homeTeam") or {}
        away = g.get("awayTeam") or {}
        rows.append(
            {
                "game_id": str(g.get("gameId") or ""),
                "gameTimeUTC": str(g.get("gameTimeUTC") or ""),
                "game_date": iso,
                "game_status_text": str(g.get("gameStatusText") or ""),
                "game_status": int(g.get("gameStatus") or 0),
                "home_team_id": int(home.get("teamId") or 0),
                "away_team_id": int(away.get("teamId") or 0),
                "home_tricode": str(home.get("teamTricode") or ""),
                "away_tricode": str(away.get("teamTricode") or ""),
                "home_display": _full_team_name(home),
                "away_display": _full_team_name(away),
                "home_score": int(home.get("score") or 0),
                "away_score": int(away.get("score") or 0),
                "neutral": bool(g.get("isNeutral")),
                "series_text": str(g.get("seriesText") or ""),
            }
        )
    return pd.DataFrame(rows)


def team_ids_playing_on(
    game_date: date, *, timeout: int = 120, league_id: str = NBA_LEAGUE_ID
) -> set[int]:
    df = fetch_scoreboard_games(game_date, timeout=timeout, league_id=league_id)
    if df.empty:
        return set()
    ids: set[int] = set()
    for _, r in df.iterrows():
        ids.add(int(r["home_team_id"]))
        ids.add(int(r["away_team_id"]))
    return ids


def batch_rest_days_before_slate(
    slate_date: date,
    team_ids: set[int],
    *,
    max_scan: int = 7,
    timeout: int = 120,
    league_id: str = NBA_LEAGUE_ID,
) -> dict[int, int | None]:
    """
    For each team id, calendar days since its **most recent** game strictly before ``slate_date``.

    Scans ``slate_date - 1`` … ``slate_date - max_scan`` (one scoreboard fetch per day).
    ``None`` means no game found in the window (early season edge or ``max_scan`` too small).
    """
    found: dict[int, int | None] = {int(t): None for t in team_ids}
    remaining = {int(t) for t in team_ids}
    for off in range(1, max_scan + 1):
        if not remaining:
            break
        d = slate_date - timedelta(days=off)
        played = team_ids_playing_on(d, timeout=timeout, league_id=league_id)
        for tid in list(remaining):
            if tid in played:
                found[tid] = off
                remaining.discard(tid)
    return found


def back_to_back_flags(
    slate_date: date,
    home_id: int,
    away_id: int,
    *,
    timeout: int = 120,
    league_id: str = NBA_LEAGUE_ID,
) -> tuple[bool, bool]:
    """True if that team played the **previous** calendar day (legacy; see ``batch_rest_days_before_slate``)."""
    prev = slate_date - timedelta(days=1)
    played = team_ids_playing_on(prev, timeout=timeout, league_id=league_id)
    return home_id in played, away_id in played


def fetch_team_advanced_stats(
    season: str,
    *,
    season_type: str = "Regular Season",
    timeout: int = 120,
    league_id: str = NBA_LEAGUE_ID,
) -> pd.DataFrame:
    """
    Advanced team rates for ``season`` (e.g. ``2025-26``).

    During playoffs you can switch ``season_type`` to ``Playoffs`` in the UI for
    playoff-only samples (noisier, smaller *n*).
    """
    raw = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        league_id_nullable=league_id,
        timeout=timeout,
    )
    df = raw.get_data_frames()[0]
    if df.empty:
        return df
    keep = [
        "TEAM_ID",
        "TEAM_NAME",
        "GP",
        "W",
        "L",
        "W_PCT",
        "E_OFF_RATING",
        "E_DEF_RATING",
        "E_NET_RATING",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "PACE",
        "TS_PCT",
        "EFG_PCT",
        "REB_PCT",
        "AST_PCT",
        "TM_TOV_PCT",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()
