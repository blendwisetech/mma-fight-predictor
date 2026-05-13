"""
Ordered tabular features for fight-level ML (**Fighter A** vs **B**).

Pre-fight record book + physicals + stance + **true** striking/grappling rates from merged
``events_raw`` bout totals, plus **Elo**, **streak / finish mix**, **recent form**, and **avg bout length**
from a strict chronological walk (see ``utils.prefight_history``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAME_FEATURE_NAMES: list[str] = [
    "f_is_womens",
    "f_wc_index",
    "f_a_height_cm",
    "f_b_height_cm",
    "f_a_reach_cm",
    "f_b_reach_cm",
    "f_a_age_y",
    "f_b_age_y",
    "f_a_wins_before",
    "f_b_wins_before",
    "f_a_losses_before",
    "f_b_losses_before",
    "f_a_fights_before",
    "f_b_fights_before",
    "f_a_win_rate_before",
    "f_b_win_rate_before",
    "f_a_days_since_last_fight",
    "f_b_days_since_last_fight",
    "f_a_slpm_before",
    "f_b_slpm_before",
    "f_a_sapm_before",
    "f_b_sapm_before",
    "f_a_td_per15_before",
    "f_b_td_per15_before",
    "f_a_kd_per_min_before",
    "f_b_kd_per_min_before",
    "f_a_sub_per_fight_before",
    "f_b_sub_per_fight_before",
    "f_a_elo_before",
    "f_b_elo_before",
    "f_a_win_streak",
    "f_b_win_streak",
    "f_a_ko_rate_before",
    "f_b_ko_rate_before",
    "f_a_sub_rate_before",
    "f_b_sub_rate_before",
    "f_a_dec_rate_before",
    "f_b_dec_rate_before",
    "f_a_avg_bout_min_before",
    "f_b_avg_bout_min_before",
    "f_a_l3_win_rate_before",
    "f_b_l3_win_rate_before",
    "f_a_stance_code",
    "f_b_stance_code",
    "f_height_diff_cm",
    "f_reach_diff_cm",
    "f_age_diff_y",
    "f_wins_before_diff",
    "f_losses_before_diff",
    "f_fights_before_diff",
    "f_win_rate_before_diff",
    "f_days_since_last_fight_diff",
    "f_slpm_diff",
    "f_sapm_diff",
    "f_td15_diff",
    "f_kdpm_diff",
    "f_sub_pf_diff",
    "f_elo_diff",
    "f_win_streak_diff",
    "f_ko_rate_diff",
    "f_sub_rate_diff",
    "f_dec_rate_diff",
    "f_avg_bout_min_diff",
    "f_l3_win_rate_diff",
]


def enriched_row_to_feature_vector(row: pd.Series) -> dict[str, float]:
    return {k: float(row.get(k, np.nan)) for k in GAME_FEATURE_NAMES}


def dataframe_X(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in GAME_FEATURE_NAMES})
    return out.astype(float)
