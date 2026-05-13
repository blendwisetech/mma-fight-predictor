"""
Aggregate injury counts from ESPN’s public NBA / WNBA injuries JSON (no key).

Used as a **rough** availability signal — not official injury reports.
"""

from __future__ import annotations

from typing import Any

import requests

ESPN_NBA_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_WNBA_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"

# ESPN sometimes abbreviates city; NBA.com scoreboard uses full city + nickname.
_ESPN_TO_NBA_DISPLAY: dict[str, str] = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _nba_display_name(espn_team_name: str) -> str:
    return _ESPN_TO_NBA_DISPLAY.get(espn_team_name.strip(), espn_team_name.strip())


def _severity_weight(status: str) -> float:
    s = (status or "").strip().lower()
    if s in ("out", "doubtful"):
        return 1.0
    if "day-to-day" in s or s in ("questionable", "gtd"):
        return 0.35
    return 0.0


def fetch_injury_weights_by_team(*, timeout: int = 45, league: str = "nba") -> dict[str, float]:
    """
    Map **league** full team display name (e.g. ``Los Angeles Lakers``) → weighted injury load.

    ``league`` is ``\"nba\"`` or ``\"wnba\"`` (ESPN public JSON, no key).

    Weight: Out/Doubtful = 1.0 each, Questionable/Day-to-day/GTD = 0.35 (tunable).
    """
    url = ESPN_WNBA_INJURIES_URL if league.strip().lower() == "wnba" else ESPN_NBA_INJURIES_URL
    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-predictor/1.0)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    payload = r.json()
    out: dict[str, float] = {}
    for bucket in payload.get("injuries") or []:
        espn_name = str(bucket.get("displayName") or "").strip()
        if not espn_name:
            continue
        nba_name = _nba_display_name(espn_name)
        key = _norm(nba_name)
        load = 0.0
        for inj in bucket.get("injuries") or []:
            st = str(inj.get("status") or "")
            load += _severity_weight(st)
        # Avoid one exploded bucket dominating when ESPN lists many minor injuries.
        out[key] = min(load, 12.0)
    return out


def injury_load_for_display(team_display: str, weights: dict[str, float]) -> float:
    """Lookup by normalized ``team_display`` (``City Name`` from NBA scoreboard)."""
    return float(weights.get(_norm(team_display), 0.0))
