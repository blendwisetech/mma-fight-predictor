"""
Write ``fight_outcomes.csv`` rows for prediction ``fight_id``s found in historical data.

Uses official ``outcome`` (``fighter1`` / ``fighter2``) plus the same **Fighter A/B** hash as training.

Run: ``python -m ml.backfill_outcomes_from_history_mma``
"""

from __future__ import annotations

import pandas as pd

from utils.data_io import ROOT, ensure_dirs, load_predictions, upsert_outcomes
from utils.feature_engineering_mma import fight_id_for_row, fighter_a_win_label_from_outcome_row
from utils.ufc_historical import load_complete_data


def main() -> None:
    ensure_dirs()
    pred = load_predictions()
    if pred.empty or "fight_id" not in pred.columns:
        print("No predictions to backfill.")
        return
    ids = set(pred["fight_id"].astype(str).unique())
    hist = load_complete_data()
    hist = hist.copy()
    hist["event_date"] = pd.to_datetime(hist["event_date"], errors="coerce")
    hist["_fid"] = hist.apply(fight_id_for_row, axis=1)
    hist = hist[hist["_fid"].isin(ids)]
    if hist.empty:
        print("No matching historical rows for logged fight_id values.")
        return
    rows = []
    for _, r in hist.iterrows():
        lab = fighter_a_win_label_from_outcome_row(r)
        if lab is None:
            continue
        rows.append(
            {
                "fight_id": r["_fid"],
                "event_date": r["event_date"].strftime("%Y-%m-%d") if pd.notna(r["event_date"]) else "",
                "fighter_a_win": int(lab),
            }
        )
    if not rows:
        print("No decisive outcomes for matched fights.")
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["fight_id"], keep="last")
    upsert_outcomes(df)
    print(f"Upserted {len(df)} outcome rows -> {ROOT / 'data' / 'outcomes' / 'fight_outcomes.csv'}")


if __name__ == "__main__":
    main()
