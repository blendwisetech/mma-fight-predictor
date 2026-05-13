"""
Scrape completed bouts from `ufcstats.com` (free public site) for dates missing
from the static jansen88 CSV.

Respect rate limits: insert sleeps between requests. Use for personal / research only.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.feature_engineering_mma import normalize_name

BASE = "http://www.ufcstats.com"
COMPLETED_EVENTS_URL = f"{BASE}/statistics/events/completed?page=all"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mma-predictor/1.0; research/education)"}


def _landed_from_of_expr(side: str | None) -> float:
    if not side:
        return float("nan")
    m = re.match(r"^(\d+)\s+of\s+(\d+)$", str(side).strip())
    return float(m.group(1)) if m else float("nan")


@dataclass
class ListedEvent:
    url: str
    title: str
    event_date: date
    location: str


def _split_two_fighter_sides(cell_val: str) -> tuple[str, str] | None:
    s = str(cell_val).strip()
    if not s:
        return None
    if "|" in s:
        a, b = [x.strip() for x in s.split("|", 1)]
        return a, b
    found = re.findall(r"\d+\s+of\s+\d+", s)
    if len(found) >= 2:
        return found[0], found[1]
    toks = s.split()
    if len(toks) >= 2 and re.match(r"^\d+$", toks[0]) and re.match(r"^\d+$", toks[1]):
        return toks[0], toks[1]
    return None


def fetch_soup(session: requests.Session, url: str, *, sleep_s: float = 0.0) -> BeautifulSoup:
    if sleep_s > 0:
        time.sleep(sleep_s)
    r = session.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def parse_completed_event_rows(session: requests.Session, *, sleep_s: float = 0.35) -> list[ListedEvent]:
    soup = fetch_soup(session, COMPLETED_EVENTS_URL, sleep_s=sleep_s)
    out: list[ListedEvent] = []
    for tr in soup.select("tr.b-statistics__table-row"):
        tds = tr.find_all("td")
        if not tds:
            continue
        line = tds[0].get_text(" ", strip=True)
        link = tr.find("a", href=lambda h: h and "event-details" in str(h))
        if not link or not line.strip() or "Name/date" in line:
            continue
        title = link.get_text(strip=True)
        rest = line[len(title) :].strip() if line.startswith(title) else ""
        ed_ts = pd.to_datetime(rest, errors="coerce")
        if pd.isna(ed_ts):
            continue
        ed = ed_ts.date()
        loc = ""
        if len(tds) > 1:
            loc = tds[1].get_text(" ", strip=True)
        href = urljoin(BASE, link.get("href", ""))
        out.append(ListedEvent(url=href, title=title, event_date=ed, location=loc))
    out.sort(key=lambda e: (e.event_date, e.title))
    return out


def parse_event_date_from_event_page(soup: BeautifulSoup) -> date | None:
    for li in soup.select("li.b-list__box-list-item"):
        t = li.get_text(" ", strip=True)
        if t.lower().startswith("date:"):
            ds = t.split(":", 1)[1].strip()
            try:
                return pd.to_datetime(ds, errors="coerce").date()
            except Exception:
                return None
    return None


def parse_event_fight_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Rows from event page (fight table)."""
    table = soup.select_one("table.b-fight-details__table")
    if not table:
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []
    rows: list[dict[str, Any]] = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue
        res_td, name_td = tds[0], tds[1]
        fight_a = tr.find("a", href=lambda h: h and "/fight-details/" in str(h))
        if not fight_a:
            continue
        furl = urljoin(BASE, fight_a.get("href", ""))
        name_links = name_td.find_all("a", href=lambda h: h and "fighter-details" in str(h))
        if len(name_links) >= 2:
            fighter1 = name_links[0].get_text(strip=True)
            fighter2 = name_links[1].get_text(strip=True)
        else:
            names_raw = name_td.get_text(" ", strip=True)
            parts = [x.strip() for x in names_raw.split("|")]
            if len(parts) != 2:
                continue
            fighter1, fighter2 = parts[0], parts[1]
        wc = tds[6].get_text(" ", strip=True)
        method = tds[7].get_text(" ", strip=True)
        rnd = tds[8].get_text(" ", strip=True)
        tim = tds[9].get_text(" ", strip=True)
        rows.append(
            {
                "fight_url": furl,
                "fighter1": fighter1,
                "fighter2": fighter2,
                "weight_class": wc or "Unknown",
                "method": method,
                "round": rnd,
                "time": tim,
            }
        )
    return rows


def _winner_loser_from_fight_page(soup: BeautifulSoup) -> tuple[str | None, str | None, str]:
    """
    Returns (winner_name, loser_name, outcome_kind).
    outcome_kind: ``decisive`` | ``draw`` | ``unknown``
    """
    people = soup.select("div.b-fight-details__person")
    wn: str | None = None
    ln: str | None = None
    had_d = False
    for div in people:
        st = div.select_one("i.b-fight-details__person-status")
        nm = div.select_one("h3.b-fight-details__person-name")
        if not st or not nm:
            continue
        status = st.get_text(strip=True)
        name = nm.get_text(strip=True)
        if status == "W":
            wn = name
        elif status == "L":
            ln = name
        elif status == "D":
            had_d = True
    if had_d and wn is None:
        return None, None, "draw"
    if wn and ln:
        return wn, ln, "decisive"
    return None, None, "unknown"


def _outcome_fighter_key(fighter1: str, fighter2: str, winner: str | None) -> str | None:
    if not winner:
        return None
    wn = normalize_name(winner)
    if wn == normalize_name(fighter1):
        return "fighter1"
    if wn == normalize_name(fighter2):
        return "fighter2"
    return None


def parse_fight_totals_for_fighters(
    soup: BeautifulSoup, fighter1: str, fighter2: str
) -> dict[str, float] | None:
    """Map fight totals (first statistics table) onto fighter1 / fighter2 order from the event."""
    tables = soup.find_all("table")
    if not tables:
        return None
    tb = tables[0]
    thead = tb.find("thead")
    body = tb.find("tbody")
    if not thead or not body:
        return None
    headers = [h.get_text(" ", strip=True) for h in thead.find_all("th")]
    row = body.find("tr")
    if not row:
        return None
    cells = row.find_all("td")
    if len(cells) != len(headers):
        return None

    ps0 = cells[0].select("p.b-fight-details__table-text")
    if len(ps0) >= 2:
        left_n = ps0[0].get_text(strip=True)
        right_n = ps0[1].get_text(strip=True)
        fighter_cell = f"{left_n} | {right_n}"
    else:
        raw = cells[0].get_text(" ", strip=True)
        if "|" not in raw:
            return None
        left_n, right_n = [x.strip() for x in raw.split("|", 1)]
        fighter_cell = raw

    data = [fighter_cell] + [c.get_text(" ", strip=True) for c in cells[1:]]
    if len(data) != len(headers):
        return None

    def col(name: str) -> str | None:
        for i, h in enumerate(headers):
            if h == name:
                return data[i]
        return None

    fcell = col("Fighter")
    if not fcell or "|" not in fcell:
        return None
    left_n, right_n = [x.strip() for x in fcell.split("|", 1)]

    def pick_side(cell_val: str | None, tgt: str) -> str | None:
        if not cell_val:
            return None
        pair = _split_two_fighter_sides(cell_val)
        if not pair:
            return None
        left_s, right_s = pair
        nt = normalize_name(tgt)
        nl, nr = normalize_name(left_n), normalize_name(right_n)
        if nt == nl:
            return left_s
        if nt == nr:
            return right_s
        return None

    sig = col("Sig. str.")
    kd_c = col("KD")
    td_c = col("Td")
    sub_c = col("Sub. att")

    out: dict[str, float] = {}
    for tgt, key_land in [
        (fighter1, "f1_sig_str_landed"),
        (fighter2, "f2_sig_str_landed"),
    ]:
        side = pick_side(sig, tgt)
        out[key_land] = _landed_from_of_expr(side)

    for tgt, key in [(fighter1, "f1_kd"), (fighter2, "f2_kd")]:
        side = pick_side(kd_c, tgt)
        if side is None:
            out[key] = float("nan")
            continue
        try:
            out[key] = float(int(side.strip()))
        except ValueError:
            out[key] = float("nan")

    for tgt, key in [(fighter1, "f1_td"), (fighter2, "f2_td")]:
        side = pick_side(td_c, tgt)
        out[key] = _landed_from_of_expr(side)

    for tgt, key in [(fighter1, "f1_sub_att"), (fighter2, "f2_sub_att")]:
        side = pick_side(sub_c, tgt)
        if side is None:
            out[key] = float("nan")
            continue
        try:
            out[key] = float(int(side.strip()))
        except ValueError:
            out[key] = float("nan")

    return out


def bout_duration_from_round_time(round_s: str, time_s: str) -> float:
    from utils.events_raw_merge import bout_duration_minutes

    try:
        rnd = int(float(round_s))
    except (TypeError, ValueError):
        rnd = float("nan")
    return bout_duration_minutes(rnd, time_s)


def scrape_one_fight_row(
    session: requests.Session,
    fight_url: str,
    fighter1: str,
    fighter2: str,
    *,
    sleep_s: float,
) -> dict[str, Any] | None:
    soup = fetch_soup(session, fight_url, sleep_s=sleep_s)
    wn, ln, kind = _winner_loser_from_fight_page(soup)
    if kind == "draw":
        outcome = "draw"
    elif kind == "decisive" and wn:
        oc = _outcome_fighter_key(fighter1, fighter2, wn)
        outcome = oc if oc else "unknown"
    else:
        outcome = "unknown"

    stats = parse_fight_totals_for_fighters(soup, fighter1, fighter2)
    if stats is None:
        stats = {}
    return {"outcome": outcome, **stats}
