"""
Upcoming MMA / UFC matchups from `The Odds API <https://the-odds-api.com/>`_.

Requires **THE_ODDS_API_KEY** or **ODDS_API_KEY** from the environment, or the same keys in **Streamlit secrets** (Community Cloud / local ``secrets.toml``).
Fights are matched to the selected calendar day using **America/New_York** (typical US card date).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from utils.feature_engineering_mma import normalize_name

NY = ZoneInfo("America/New_York")
ODDS_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
PREFERRED_BOOKMAKERS = (
    "draftkings",
    "fanduel",
    "betmgm",
    "bovada",
    "betonlineag",
    "lowvig",
    "mybookieag",
    "williamhill_us",
)


def get_odds_api_key() -> str | None:
    for name in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    try:
        import streamlit as st

        sec = getattr(st, "secrets", None)
        if sec is not None:
            for name in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
                if name in sec:
                    out = str(sec[name]).strip()
                    if out:
                        return out
    except Exception:
        pass
    return None


def _pick_bookmaker(bookmakers: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not bookmakers:
        return None
    by_key = {str(b.get("key") or ""): b for b in bookmakers}
    for pref in PREFERRED_BOOKMAKERS:
        if pref in by_key:
            return by_key[pref]
    return bookmakers[0]


def _favourite_underdog_from_h2h(
    bm: dict[str, Any], home_team: str, away_team: str
) -> tuple[str, str, float, float] | None:
    hn, an = normalize_name(home_team), normalize_name(away_team)
    for m in bm.get("markets", []):
        if str(m.get("key") or "") != "h2h":
            continue
        outs: list[dict[str, Any]] = []
        for o in m.get("outcomes", []):
            nm_raw = str(o.get("name") or "").strip()
            if not nm_raw:
                continue
            if normalize_name(nm_raw) in ("draw", "no contest"):
                continue
            nn = normalize_name(nm_raw)
            if nn not in (hn, an):
                # Some books add extra H2H outcomes; keep only the two fighters.
                continue
            try:
                price = float(o["price"])
            except (TypeError, ValueError, KeyError):
                continue
            outs.append((nm_raw, price))
        if len(outs) < 2:
            continue
        outs.sort(key=lambda x: x[1])
        fav_n, d_fav = outs[0][0], outs[0][1]
        dog_n, d_und = outs[-1][0], outs[-1][1]
        return fav_n, dog_n, float(d_fav), float(d_und)
    return None


def fetch_mma_odds_events(api_key: str, *, timeout: int = 45) -> list[dict[str, Any]]:
    r = requests.get(
        ODDS_URL,
        params={
            "apiKey": api_key.strip(),
            "regions": "us,us2",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected Odds API response shape (expected list).")
    return data


def mma_events_to_slate_dataframe(events: list[dict[str, Any]], pick: date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ix, ev in enumerate(events):
        ct_raw = ev.get("commence_time")
        if not ct_raw:
            continue
        ct = pd.Timestamp(ct_raw)
        if ct.tz is None:
            ct = ct.tz_localize("UTC")
        else:
            ct = ct.tz_convert("UTC")
        card_day = ct.tz_convert(NY).date()
        if card_day != pick:
            continue

        home = str(ev.get("home_team") or "").strip()
        away = str(ev.get("away_team") or "").strip()
        if not home or not away:
            continue

        bm = _pick_bookmaker(ev.get("bookmakers") or [])
        if bm is None:
            continue

        sides = _favourite_underdog_from_h2h(bm, home, away)
        if sides is None:
            continue
        fav_n, dog_n, d_fav, d_und = sides
        ed_str = card_day.strftime("%Y-%m-%d")
        title = str(ev.get("sport_title") or "MMA")
        rows.append(
            {
                "event_date": ed_str,
                "event_name": f"{title} (Odds API · {str(bm.get('title') or bm.get('key') or 'book')})",
                "weight_class": "Unknown",
                "fighter1": home,
                "fighter2": away,
                "favourite": fav_n,
                "underdog": dog_n,
                "favourite_odds": d_fav,
                "underdog_odds": d_und,
                "outcome": float("nan"),
                "_idx": ix,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_mma_slate_for_event_date(api_key: str, pick: date) -> pd.DataFrame:
    """Return slate rows for ``pick`` (US/Eastern card date) or an empty DataFrame."""
    events = fetch_mma_odds_events(api_key)
    return mma_events_to_slate_dataframe(events, pick)
