"""
Join WNBA prediction log rows with WNBA outcomes → ``training_games_wnba.parquet``.

Run from ``nba_predictor``:
  python -m ml.merge_training_data_wnba
"""

from __future__ import annotations

import pandas as pd

from ml.merge_training_logs import merge_league_logs


def merge_logs() -> pd.DataFrame:
    return merge_league_logs("wnba")


def main() -> None:
    m = merge_logs()
    print(f"merged WNBA rows: {len(m)}")


if __name__ == "__main__":
    main()
