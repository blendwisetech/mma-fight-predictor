"""
Join prediction log with outcome log by ``fight_id``.

Produces ``data/processed/training_fights.parquet`` for trainers.

Run: ``python -m ml.merge_training_data_mma``
"""

from __future__ import annotations

from utils.data_io import PATH_MERGED, ROOT, ensure_dirs, load_outcomes, load_predictions


def main() -> None:
    ensure_dirs()
    pred = load_predictions()
    out = load_outcomes()
    if pred.empty or out.empty:
        print("Need non-empty predictions and outcomes CSVs before merge.")
        return
    if "fight_id" not in pred.columns or "fight_id" not in out.columns:
        print("fight_id column missing.")
        return
    out = out.drop_duplicates(subset=["fight_id"], keep="last")
    pred = (
        pred.sort_values("logged_at").drop_duplicates(subset=["fight_id"], keep="last")
        if "logged_at" in pred.columns
        else pred.drop_duplicates(subset=["fight_id"], keep="last")
    )
    merged = pred.merge(out, on="fight_id", how="inner", suffixes=("", "_out"))
    PATH_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PATH_MERGED, index=False)
    print(f"Wrote {len(merged)} rows to {PATH_MERGED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
