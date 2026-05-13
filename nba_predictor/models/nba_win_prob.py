"""
Home win probability: transparent logistic on NET / fatigue / home court,
with optional sklearn bundle from ``data/models``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def heuristic_home_win_prob(row: pd.Series) -> float:
    """
    P(home) from season NET differential, home court, neutral floor, rest, B2B, and injury load.

    Coefficients are a sane prior; replace with trained weights once you have enough labels.
    """
    hn = float(row.get("home_team_NET_RATING") or float("nan"))
    an = float(row.get("away_team_NET_RATING") or float("nan"))
    if not np.isfinite(hn) or not np.isfinite(an):
        return 0.5
    net_adv = hn - an
    hb = 1.0 if bool(row.get("home_b2b")) else 0.0
    ab = 1.0 if bool(row.get("away_b2b")) else 0.0
    fatigue = 0.08 * (ab - hb)

    rh = row.get("home_rest_days")
    ra = row.get("away_rest_days")
    try:
        rhf = float(rh) if rh is not None and not (isinstance(rh, float) and np.isnan(rh)) else float("nan")
        raf = float(ra) if ra is not None and not (isinstance(ra, float) and np.isnan(ra)) else float("nan")
    except (TypeError, ValueError):
        rhf = raf = float("nan")
    if np.isfinite(rhf) and np.isfinite(raf):
        rest_term = 0.034 * float(rhf - raf)
    else:
        rest_term = fatigue

    hinj = float(row.get("home_injury_load") or 0.0)
    ainj = float(row.get("away_injury_load") or 0.0)
    if not np.isfinite(hinj):
        hinj = 0.0
    if not np.isfinite(ainj):
        ainj = 0.0
    inj_term = 0.048 * float(hinj - ainj)

    home_court = 0.0 if bool(row.get("neutral_site")) else 0.45
    z = 0.12 * net_adv + home_court + rest_term - inj_term
    p = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(p, 0.03, 0.97))


def predict_home_win(
    row: pd.Series,
    clf_bundle: dict | None,
    *,
    blend_ml_weight: float = 0.0,
) -> tuple[float, str, float | None]:
    """
    Returns ``(p_home, version_label, raw_ml_or_none)``.

    ``blend_ml_weight`` in ``[0,1]`` mixes heuristic and sklearn **before** clipping.
    """
    base = heuristic_home_win_prob(row)
    raw_ml: float | None = None
    ver = "heuristic_net_v1"
    if clf_bundle and clf_bundle.get("pipeline") is not None:
        from ml.feature_config_nba import dataframe_X, enriched_row_to_feature_vector

        X = dataframe_X(pd.DataFrame([enriched_row_to_feature_vector(row)]))
        try:
            raw_ml = float(clf_bundle["pipeline"].predict_proba(X)[0, 1])
        except Exception:
            raw_ml = None
        if raw_ml is not None:
            w = float(np.clip(blend_ml_weight, 0.0, 1.0))
            if w > 0:
                ver = str(clf_bundle.get("version") or "sklearn_v1")
                base = (1.0 - w) * base + w * raw_ml
            else:
                ver = "heuristic_net_v1"
    return float(np.clip(base, 0.02, 0.98)), ver, raw_ml
