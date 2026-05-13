"""
NBA lines from **Bovada.lv** public coupon JSON (no API key).

Parses **moneyline** (Head-to-Head), **main full-game spread** (Main Dynamic Asian Handicap),
and **main full-game total** (Main Dynamic Over/Under).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

BOVADA_NBA_COUPON_URL = (
    "https://www.bovada.lv/services/sports/event/coupon/events/A/description/basketball/nba"
)
BOVADA_WNBA_COUPON_URL = (
    "https://www.bovada.lv/services/sports/event/coupon/events/A/description/basketball/wnba"
)


def _norm_team(s: str) -> str:
    return (s or "").strip().lower()


def _teams_exact(a1: str, h1: str, a2: str, h2: str) -> bool:
    return _norm_team(a1) == _norm_team(a2) and _norm_team(h1) == _norm_team(h2)


def _teams_loose(a1: str, h1: str, a2: str, h2: str) -> bool:
    def lo(x: str, y: str) -> bool:
        a, b = _norm_team(x), _norm_team(y)
        if not a or not b:
            return False
        if a == b:
            return True
        return len(a) >= 4 and (a in b or b in a)

    return lo(a1, a2) and lo(h1, h2)


def _parse_american(raw: Any) -> float | None:
    if raw is None:
        return None
    t = str(raw).strip().replace("\u2212", "-").replace("−", "-")
    if t.upper() == "EVEN":
        return 100.0
    try:
        return float(t)
    except ValueError:
        return None


def _parse_handicap(raw: Any) -> float | None:
    if raw is None:
        return None
    t = str(raw).strip().replace("\u2212", "-").replace("−", "-")
    try:
        return float(t)
    except ValueError:
        return None


def _event_et_date(start_ms: Any) -> date | None:
    if start_ms is None:
        return None
    try:
        ms = float(start_ms)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(ms / 1000.0, tz=_ET)
    return dt.date()


def _competitor_names(ev: dict[str, Any]) -> tuple[str | None, str | None]:
    away_n = home_n = None
    for c in ev.get("competitors") or []:
        name = c.get("name")
        if not name:
            continue
        if c.get("home") is True:
            home_n = str(name)
        else:
            away_n = str(name)
    return away_n, home_n


def _competitor_home_by_id(ev: dict[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for c in ev.get("competitors") or []:
        cid = c.get("id")
        if cid is None:
            continue
        out[str(cid)] = bool(c.get("home"))
    return out


def _is_main_full_game_market(mkt: dict[str, Any]) -> bool:
    per = mkt.get("period") or {}
    if per.get("live"):
        return False
    if not per.get("main", False):
        return False
    abbr = str(per.get("abbreviation") or "")
    desc = str(per.get("description") or "").lower()
    return abbr == "G" or desc == "game"


def _pick_head_to_head_market(ev: dict[str, Any]) -> dict[str, Any] | None:
    found: list[dict[str, Any]] = []
    for dg in ev.get("displayGroups") or []:
        for mkt in dg.get("markets") or []:
            if mkt.get("descriptionKey") != "Head To Head":
                continue
            outs = mkt.get("outcomes") or []
            if len(outs) != 2:
                continue
            found.append(mkt)
    if not found:
        return None
    non_live = [m for m in found if not (m.get("period") or {}).get("live")]
    pool = non_live if non_live else found
    pool.sort(key=lambda m: not (m.get("period") or {}).get("main", False))
    return pool[0]


def _pick_main_dynamic_market(ev: dict[str, Any], description_key: str) -> dict[str, Any] | None:
    cands: list[dict[str, Any]] = []
    for dg in ev.get("displayGroups") or []:
        for mkt in dg.get("markets") or []:
            if mkt.get("descriptionKey") != description_key:
                continue
            if not _is_main_full_game_market(mkt):
                continue
            outs = mkt.get("outcomes") or []
            if len(outs) < 2:
                continue
            cands.append(mkt)
    if not cands:
        return None
    cands.sort(key=lambda m: str(m.get("id") or ""))
    return cands[0]


def _moneylines_from_h2h(ev: dict[str, Any], mkt: dict[str, Any]) -> tuple[float | None, float | None]:
    id_to_side = _competitor_home_by_id(ev)
    away_ml = home_ml = None
    for o in mkt.get("outcomes") or []:
        cid = o.get("competitorId")
        price = _parse_american((o.get("price") or {}).get("american"))
        if cid is None or price is None:
            continue
        is_home = id_to_side.get(str(cid))
        if is_home is True:
            home_ml = price
        elif is_home is False:
            away_ml = price
    return away_ml, home_ml


def _spread_lines_from_market(ev: dict[str, Any], mkt: dict[str, Any]) -> dict[str, Any]:
    id_to_side = _competitor_home_by_id(ev)
    out: dict[str, Any] = {
        "away_spread_line": None,
        "home_spread_line": None,
        "away_spread_am": None,
        "home_spread_am": None,
    }
    for o in mkt.get("outcomes") or []:
        cid = o.get("competitorId")
        if cid is None:
            continue
        side = id_to_side.get(str(cid))
        pr = o.get("price") or {}
        am = _parse_american(pr.get("american"))
        hc = _parse_handicap(pr.get("handicap"))
        if side is False:
            out["away_spread_line"] = hc
            out["away_spread_am"] = am
        elif side is True:
            out["home_spread_line"] = hc
            out["home_spread_am"] = am
    return out


def _total_from_market(mkt: dict[str, Any]) -> dict[str, Any]:
    total_line: float | None = None
    over_am: float | None = None
    under_am: float | None = None
    for o in mkt.get("outcomes") or []:
        if str(o.get("type") or "").upper() != "O":
            continue
        pr = o.get("price") or {}
        hc = _parse_handicap(pr.get("handicap"))
        if hc is None:
            continue
        total_line = hc
        over_am = _parse_american(pr.get("american"))
        break
    for o in mkt.get("outcomes") or []:
        if str(o.get("type") or "").upper() != "U":
            continue
        pr = o.get("price") or {}
        hc = _parse_handicap(pr.get("handicap"))
        if hc is None or total_line is None:
            continue
        if abs(hc - float(total_line)) > 0.05:
            continue
        under_am = _parse_american(pr.get("american"))
        break
    return {"total_line": total_line, "over_am": over_am, "under_am": under_am}


def _fetch_bovada_slate_from_url(slate_date: date, coupon_url: str) -> list[dict[str, Any]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-predictor/1.0)"}
    r = requests.get(coupon_url, timeout=60, headers=headers)
    r.raise_for_status()
    payload = r.json()
    if not payload or not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for league in payload:
        for ev in league.get("events") or []:
            ed = _event_et_date(ev.get("startTime"))
            if ed != slate_date:
                continue
            away_n, home_n = _competitor_names(ev)
            if not away_n or not home_n:
                continue
            mkt = _pick_head_to_head_market(ev)
            if mkt is None:
                continue
            aml, hml = _moneylines_from_h2h(ev, mkt)
            if aml is None or hml is None:
                continue
            row: dict[str, Any] = {
                "source_id": str(ev.get("id") or ""),
                "away_team": away_n,
                "home_team": home_n,
                "away_ml": float(aml),
                "home_ml": float(hml),
                "provider": "Bovada",
            }
            sm = _pick_main_dynamic_market(ev, "Main Dynamic Asian Handicap")
            if sm:
                row.update(_spread_lines_from_market(ev, sm))
            tm = _pick_main_dynamic_market(ev, "Main Dynamic Over/Under")
            if tm:
                row.update(_total_from_market(tm))
            rows.append(row)
    return rows


def fetch_nba_bovada_slate(slate_date: date) -> list[dict[str, Any]]:
    """
    Full-game Bovada rows for ``slate_date`` (Eastern calendar date on the event).

    Each dict includes moneylines plus spread/total when present.
    """
    return _fetch_bovada_slate_from_url(slate_date, BOVADA_NBA_COUPON_URL)


def fetch_wnba_bovada_slate(slate_date: date) -> list[dict[str, Any]]:
    """Same parsing as NBA, using Bovada’s **WNBA** coupon."""
    return _fetch_bovada_slate_from_url(slate_date, BOVADA_WNBA_COUPON_URL)


def fetch_moneylines_for_slate_date(slate_date: date) -> list[dict[str, Any]]:
    """Backward-compatible alias — returns Bovada rows including ML (and spread/total when parsed)."""
    return fetch_nba_bovada_slate(slate_date)


def lookup_bovada_row(
    rows: list[dict[str, Any]],
    away_name: str,
    home_name: str,
) -> dict[str, Any] | None:
    for row in rows:
        a2, h2 = row.get("away_team") or "", row.get("home_team") or ""
        if _teams_exact(away_name, home_name, a2, h2):
            return row
    for row in rows:
        a2, h2 = row.get("away_team") or "", row.get("home_team") or ""
        if _teams_loose(away_name, home_name, a2, h2):
            return row
    return None


def lookup_moneylines_pair(
    rows: list[dict[str, Any]],
    away_name: str,
    home_name: str,
) -> tuple[float | None, float | None]:
    row = lookup_bovada_row(rows, away_name, home_name)
    if not row:
        return None, None
    return row.get("away_ml"), row.get("home_ml")


def moneyline_for_pick(
    rows: list[dict[str, Any]],
    away_name: str,
    home_name: str,
    pick_name: str,
) -> float | None:
    aml, hml = lookup_moneylines_pair(rows, away_name, home_name)
    if aml is None or hml is None:
        return None
    pn = _norm_team(pick_name)
    if pn == _norm_team(home_name):
        return float(hml)
    if pn == _norm_team(away_name):
        return float(aml)
    hn, an = _norm_team(home_name), _norm_team(away_name)
    if len(pn) >= 4 and (pn in hn or hn in pn):
        return float(hml)
    if len(pn) >= 4 and (pn in an or an in pn):
        return float(aml)
    return None
