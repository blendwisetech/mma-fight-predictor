"""
Download / refresh UFC bout history from ufcstats.com into ``data/raw/ufcstats_extension.parquet``.

This fills the gap after the frozen jansen88 CSV. Run periodically:

    python -m utils.ufcstats_sync

First run can take a long time (many HTTP requests). Use ``--max-events`` for testing.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from utils.feature_engineering_mma import fight_id_for_row
from utils.ufc_historical import RAW_PATH, load_complete_data
from utils.ufcstats_scrape import (
    ListedEvent,
    bout_duration_from_round_time,
    fetch_soup,
    parse_completed_event_rows,
    parse_event_date_from_event_page,
    parse_event_fight_rows,
    scrape_one_fight_row,
)
from utils.events_raw_merge import merge_bout_stats_into_complete

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = ROOT / "data" / "raw" / "ufcstats_extension.parquet"


def reference_merged_columns() -> list[str]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Need jansen88 CSV at {RAW_PATH} (run utils.ufc_historical download).")
    c = load_complete_data()
    ref = merge_bout_stats_into_complete(c.head(1))
    return ref.columns.tolist()


def _empty_row(cols: list[str]) -> dict:
    return {c: float("nan") for c in cols}


def build_merged_shaped_row(
    ev: ListedEvent,
    event_date_str: str,
    fight: dict[str, str],
    stats: dict,
    cols: list[str],
) -> dict[str, object]:
    r = _empty_row(cols)
    r["event_date"] = event_date_str
    r["event_name"] = ev.title
    r["weight_class"] = fight["weight_class"]
    r["fighter1"] = fight["fighter1"]
    r["fighter2"] = fight["fighter2"]
    r["method"] = fight.get("method", "")
    try:
        r["round"] = int(float(str(fight.get("round", ""))))
    except (TypeError, ValueError):
        r["round"] = float("nan")

    oc = stats.get("outcome", "")
    if oc in ("fighter1", "fighter2", "draw"):
        r["outcome"] = oc
    else:
        r["outcome"] = float("nan")

    dur = bout_duration_from_round_time(str(fight.get("round", "")), str(fight.get("time", "")))
    r["bout_duration_min"] = dur

    for k in (
        "f1_kd",
        "f2_kd",
        "f1_sig_str_landed",
        "f2_sig_str_landed",
        "f1_td",
        "f2_td",
        "f1_sub_att",
        "f2_sub_att",
    ):
        v = stats.get(k)
        r[k] = float(v) if v is not None and pd.notna(v) else float("nan")

    return r


def load_extension_parquet() -> pd.DataFrame | None:
    if not EXTENSION_PATH.exists() or EXTENSION_PATH.stat().st_size < 32:
        return None
    return pd.read_parquet(EXTENSION_PATH)


def run_sync(
    *,
    since: date,
    sleep_fight: float = 0.35,
    sleep_event: float = 0.5,
    max_events: int | None = None,
    rebuild: bool = False,
) -> int:
    cols = reference_merged_columns()
    session = requests.Session()
    listed = parse_completed_event_rows(session, sleep_s=sleep_event)
    listed = [e for e in listed if e.event_date >= since]
    if max_events is not None:
        listed = listed[: int(max_events)]

    seen: set[str] = set()
    prior_rows: list[dict] = []
    if not rebuild:
        old = load_extension_parquet()
        if old is not None and not old.empty and "fight_id" in old.columns:
            seen = set(old["fight_id"].astype(str))
            prior_rows = old.to_dict("records")

    new_rows: list[dict] = []
    for i, ev in enumerate(listed):
        esoup = fetch_soup(session, ev.url, sleep_s=sleep_event)
        d_page = parse_event_date_from_event_page(esoup)
        ed = d_page or ev.event_date
        event_date_str = ed.strftime("%Y-%m-%d")

        fights = parse_event_fight_rows(esoup)
        print(
            f"[{i + 1}/{len(listed)}] {event_date_str} — {ev.title} ({len(fights)} bouts)",
            flush=True,
        )
        for f in fights:
            ser_id = pd.Series(
                {
                    "event_date": event_date_str,
                    "fighter1": f["fighter1"],
                    "fighter2": f["fighter2"],
                    "weight_class": f["weight_class"],
                }
            )
            fid = str(fight_id_for_row(ser_id))
            if fid in seen:
                continue
            try:
                st = scrape_one_fight_row(
                    session,
                    f["fight_url"],
                    f["fighter1"],
                    f["fighter2"],
                    sleep_s=sleep_fight,
                )
            except Exception as ex:
                print(
                    f"  skip fight (fetch error): {f['fighter1']} vs {f['fighter2']} — {ex}",
                    flush=True,
                )
                continue
            if not st or st.get("outcome") not in ("fighter1", "fighter2", "draw"):
                print(
                    f"  skip fight (no outcome): {f['fighter1']} vs {f['fighter2']}",
                    flush=True,
                )
                continue
            ev_adj = ListedEvent(url=ev.url, title=ev.title, event_date=ed, location=ev.location)
            row = build_merged_shaped_row(ev_adj, event_date_str, f, st, cols)
            row["fight_id"] = fid
            new_rows.append(row)
            seen.add(fid)

    if not new_rows:
        print("No new bouts scraped.", flush=True)
        return 0

    df_new = pd.DataFrame(new_rows)
    if rebuild or not prior_rows:
        df_all = df_new
    else:
        df_all = pd.concat([pd.DataFrame(prior_rows), df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["fight_id"], keep="last", inplace=True)
    EXTENSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(EXTENSION_PATH, index=False)
    print(
        f"Wrote {len(df_new)} new rows ({len(df_all)} total) -> {EXTENSION_PATH}",
        flush=True,
    )
    return len(df_new)


def main() -> None:
    p = argparse.ArgumentParser(description="Sync UFC extension from ufcstats.com")
    p.add_argument("--since", default="2023-09-17", help="ISO min event date (default fills post-jansen88 gap)")
    p.add_argument("--sleep-fight", type=float, default=0.35)
    p.add_argument("--sleep-event", type=float, default=0.5)
    p.add_argument("--max-events", type=int, default=None, help="Cap events for testing")
    p.add_argument("--rebuild", action="store_true", help="Drop prior extension parquet")
    args = p.parse_args()
    since_d = date.fromisoformat(str(args.since))
    if args.rebuild and EXTENSION_PATH.exists():
        EXTENSION_PATH.unlink()
    n = run_sync(
        since=since_d,
        sleep_fight=args.sleep_fight,
        sleep_event=args.sleep_event,
        max_events=args.max_events,
        rebuild=args.rebuild,
    )
    raise SystemExit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
