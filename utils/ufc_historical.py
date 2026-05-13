"""
Download and cache UFC historical tables from jansen88/ufc-data.

- ``complete_ufc_data.csv`` — wide row per bout (physicals, odds, career fields we *do not* use as rates).
- ``events_raw.csv`` — per-bout totals (sig strikes, TD, KD, sub attempts, round/time) merged for **true**
  pre-fight cumulative rates (see ``utils.events_raw_merge``).

Run: ``python -m utils.ufc_historical download``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://raw.githubusercontent.com/jansen88/ufc-data/master/data/complete_ufc_data.csv"
)
RAW_PATH = ROOT / "data" / "raw" / "complete_ufc_data.csv"


def raw_csv_path() -> Path:
    return RAW_PATH


def ensure_downloaded(*, url: str = DEFAULT_URL, timeout: int = 120) -> Path:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 1_000_000:
        return RAW_PATH
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    RAW_PATH.write_bytes(r.content)
    return RAW_PATH


def load_complete_data(*, refresh: bool = False) -> pd.DataFrame:
    if refresh and RAW_PATH.exists():
        RAW_PATH.unlink(missing_ok=True)
    ensure_downloaded()
    return pd.read_csv(RAW_PATH, low_memory=False)


def _dedup_bout_key(r: pd.Series) -> str:
    from utils.events_raw_merge import _norm_join_key

    ed = pd.to_datetime(r.get("event_date"), errors="coerce")
    if pd.isna(ed):
        return ""
    return _norm_join_key(ed, str(r.get("fighter1", "")), str(r.get("fighter2", "")))


def load_merged_bout_history(*, refresh: bool = False, refresh_events: bool = False) -> pd.DataFrame:
    """``complete_ufc_data`` left-joined with parsed ``events_raw`` per-bout stats, plus optional UFC Stats extension."""
    from utils.events_raw_merge import merge_bout_stats_into_complete

    c = load_complete_data(refresh=refresh)
    c = merge_bout_stats_into_complete(c, refresh_events=refresh_events)

    ext_path = ROOT / "data" / "raw" / "ufcstats_extension.parquet"
    if not ext_path.exists() or ext_path.stat().st_size < 32:
        return c

    try:
        ext = pd.read_parquet(ext_path)
    except Exception:
        return c

    if ext is None or ext.empty:
        return c

    # Prefer extension rows when the same bout exists (newer stats + outcomes).
    ext = ext.drop(columns=["fight_id"], errors="ignore")
    for col in c.columns:
        if col not in ext.columns:
            ext[col] = np.nan
    ext = ext[[*c.columns]].copy()
    ext["_dedup"] = ext.apply(_dedup_bout_key, axis=1)
    c = c.copy()
    c["_dedup"] = c.apply(_dedup_bout_key, axis=1)
    drop_keys = set(ext.loc[ext["_dedup"].astype(str).str.len() > 0, "_dedup"].astype(str))
    c = c[~c["_dedup"].isin(drop_keys)]
    out = pd.concat([c, ext.drop(columns=["_dedup"])], ignore_index=True)
    out = out.drop(columns=["_dedup"], errors="ignore")
    if "event_date" in out.columns:
        out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="UFC historical CSV helper")
    p.add_argument("cmd", nargs="?", default="download", choices=("download", "info"))
    args = p.parse_args()
    if args.cmd == "download":
        path = ensure_downloaded()
        from utils.events_raw_merge import ensure_events_raw_downloaded

        ep = ensure_events_raw_downloaded()
        print(f"complete: {path} ({path.stat().st_size // 1024} KiB)")
        print(f"events_raw: {ep} ({ep.stat().st_size // 1024} KiB)")
    elif args.cmd == "info":
        df = load_complete_data()
        print(f"rows={len(df)} cols={len(df.columns)} date_min={df['event_date'].min()} date_max={df['event_date'].max()}")


if __name__ == "__main__":
    main()
