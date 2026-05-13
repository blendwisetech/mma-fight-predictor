"""
Join NBA prediction log rows with NBA outcomes → ``training_games.parquet``.

WNBA rows live in ``training_games_wnba.parquet`` via ``merge_training_data_wnba``.

Run from ``nba_predictor``:
  python -m ml.merge_training_data_nba
"""

from __future__ import annotations

import pandas as pd

from ml.merge_training_logs import merge_league_logs


def merge_logs() -> pd.DataFrame:
    return merge_league_logs("nba")


def main() -> None:
    m = merge_logs()
    print(f"merged rows: {len(m)}")


if __name__ == "__main__":
    main()
