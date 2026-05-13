"""
Cold-start training table: official ``outcome`` labels + strict pre-fight features.

Run from ``mma_predictor``: ``python -m ml.seed_training_from_history_mma``
"""

from __future__ import annotations

import pandas as pd

from ml.feature_config_mma import GAME_FEATURE_NAMES
from utils.data_io import PATH_MERGED, ROOT, ensure_dirs
from utils.feature_engineering_mma import (
    fit_weight_class_map,
    heuristic_fighter_a_win_prob,
    save_weight_class_map,
)
from utils.prefight_history import build_prefight_training_table
from utils.ufc_historical import load_merged_bout_history


def main() -> None:
    ensure_dirs()
    df = load_merged_bout_history()
    if df.empty:
        print("Historical CSV empty.")
        return

    wc_map = fit_weight_class_map(df["weight_class"].astype(str))
    save_weight_class_map(wc_map)

    feat = build_prefight_training_table(df, wc_map)
    if feat.empty:
        print("No training rows produced.")
        return

    feat["pred_fighter_a_win_prob_heur"] = feat.apply(
        lambda r: heuristic_fighter_a_win_prob(pd.Series(r)), axis=1
    )

    miss = [c for c in GAME_FEATURE_NAMES if c not in feat.columns]
    if miss:
        print(f"Internal error: missing feature cols: {miss}")
        return

    PATH_MERGED.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(PATH_MERGED, index=False)
    print(
        f"Wrote {len(feat)} rows -> {PATH_MERGED.relative_to(ROOT)} | "
        f"fighter_a_win mean={feat['fighter_a_win'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
