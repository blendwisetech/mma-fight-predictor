"""
Merge jansen88 ``events_raw.csv`` (per-bout totals) into the wide ``complete_ufc_data`` frame.

Joined on ``event_date`` + ``fighter1`` + ``fighter2`` (normalized). Adds parsed:

- ``bout_duration_min`` — length of the bout in minutes (5-minute rounds).
- ``f1_sig_str_landed`` / ``f2_sig_str_landed`` — significant strikes (``Str`` column).
- ``f1_kd`` / ``f2_kd``, ``f1_td`` / ``f2_td``, ``f1_sub_att`` / ``f2_sub_att``
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EVENTS_RAW_URL = "https://raw.githubusercontent.com/jansen88/ufc-data/master/data/events_raw.csv"
EVENTS_RAW_PATH = ROOT / "data" / "raw" / "events_raw.csv"


def ensure_events_raw_downloaded(*, timeout: int = 120) -> Path:
    EVENTS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EVENTS_RAW_PATH.exists() and EVENTS_RAW_PATH.stat().st_size > 100_000:
        return EVENTS_RAW_PATH
    r = requests.get(EVENTS_RAW_URL, timeout=timeout)
    r.raise_for_status()
    EVENTS_RAW_PATH.write_bytes(r.content)
    return EVENTS_RAW_PATH


def parse_two_nonnegative_ints(cell: object) -> tuple[int | None, int | None]:
    nums = [int(x) for x in re.findall(r"\d+", str(cell))]
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], None
    return None, None


def bout_duration_minutes(round_val: object, time_str: object) -> float:
    try:
        rnd = int(float(round_val))
    except (TypeError, ValueError):
        return float("nan")
    if rnd < 1:
        return float("nan")
    if pd.isna(time_str):
        return float("nan")
    ts = str(time_str).strip()
    if ":" not in ts:
        return float("nan")
    parts = ts.split(":")
    try:
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
        else:
            return float("nan")
    except ValueError:
        return float("nan")
    last_round_sec = m * 60 + s
    completed_full_rounds = max(0, rnd - 1)
    total_sec = completed_full_rounds * 5 * 60 + last_round_sec
    return float(total_sec / 60.0)


def _norm_join_key(event_date: pd.Timestamp, f1: str, f2: str) -> str:
    def nn(s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    if pd.isna(event_date):
        return ""
    return f"{pd.Timestamp(event_date).strftime('%Y-%m-%d')}|{nn(f1)}|{nn(f2)}"


def load_events_raw_parsed(*, refresh: bool = False) -> pd.DataFrame:
    if refresh and EVENTS_RAW_PATH.exists():
        EVENTS_RAW_PATH.unlink(missing_ok=True)
    ensure_events_raw_downloaded()
    e = pd.read_csv(EVENTS_RAW_PATH, low_memory=False)
    e["event_date"] = pd.to_datetime(e["event_date"], errors="coerce")

    def row_dur(r: pd.Series) -> float:
        return bout_duration_minutes(r.get("Round"), r.get("Time"))

    e["bout_duration_min"] = e.apply(row_dur, axis=1)

    def split_stats(row: pd.Series) -> pd.Series:
        kd1, kd2 = parse_two_nonnegative_ints(row.get("Kd"))
        s1, s2 = parse_two_nonnegative_ints(row.get("Str"))
        t1, t2 = parse_two_nonnegative_ints(row.get("Td"))
        sb1, sb2 = parse_two_nonnegative_ints(row.get("Sub"))
        return pd.Series(
            {
                "f1_kd": kd1 if kd1 is not None else np.nan,
                "f2_kd": kd2 if kd2 is not None else np.nan,
                "f1_sig_str_landed": float(s1) if s1 is not None else np.nan,
                "f2_sig_str_landed": float(s2) if s2 is not None else np.nan,
                "f1_td": float(t1) if t1 is not None else np.nan,
                "f2_td": float(t2) if t2 is not None else np.nan,
                "f1_sub_att": float(sb1) if sb1 is not None else np.nan,
                "f2_sub_att": float(sb2) if sb2 is not None else np.nan,
            }
        )

    stats = e.apply(split_stats, axis=1)
    out = pd.concat([e, stats], axis=1)
    out["_jk"] = out.apply(
        lambda r: _norm_join_key(r["event_date"], str(r.get("fighter1", "")), str(r.get("fighter2", ""))),
        axis=1,
    )
    return out


def merge_bout_stats_into_complete(complete: pd.DataFrame, *, refresh_events: bool = False) -> pd.DataFrame:
    """Left-join per-bout stat columns onto the wide historical table."""
    c = complete.copy()
    c["event_date"] = pd.to_datetime(c["event_date"], errors="coerce")
    c["_jk"] = c.apply(
        lambda r: _norm_join_key(r["event_date"], str(r.get("fighter1", "")), str(r.get("fighter2", ""))),
        axis=1,
    )
    ev = load_events_raw_parsed(refresh=refresh_events)
    keep = [
        "_jk",
        "bout_duration_min",
        "f1_kd",
        "f2_kd",
        "f1_sig_str_landed",
        "f2_sig_str_landed",
        "f1_td",
        "f2_td",
        "f1_sub_att",
        "f2_sub_att",
    ]
    sub = ev[keep].drop_duplicates(subset=["_jk"], keep="last")
    m = c.merge(sub, on="_jk", how="left", suffixes=("", "_ev"))
    m = m.drop(columns=["_jk"], errors="ignore")
    return m
