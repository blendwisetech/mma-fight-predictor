"""Odds / Kelly helpers for exploratory bet-sizing UI (not financial advice)."""

from __future__ import annotations

import math

import numpy as np


def american_to_implied_prob(american: float) -> float:
    """Implied win probability for one side from American odds (includes vig)."""
    if american is None or (isinstance(american, float) and math.isnan(american)):
        return float("nan")
    o = float(american)
    if o == 0.0:
        return float("nan")
    if o < 0:
        a = abs(o)
        return a / (a + 100.0)
    return 100.0 / (o + 100.0)


def fair_probs_two_way_moneyline(away_american: float, home_american: float) -> tuple[float, float]:
    """
    Remove two-way vig by normalizing implied probabilities so they sum to 1.

    Returns ``(p_away, p_home)`` or ``(nan, nan)`` if odds are unusable.
    """
    pa = american_to_implied_prob(away_american)
    ph = american_to_implied_prob(home_american)
    if not math.isfinite(pa) or not math.isfinite(ph):
        return float("nan"), float("nan")
    s = float(pa) + float(ph)
    if s <= 1e-9:
        return float("nan"), float("nan")
    return float(pa) / s, float(ph) / s


def blend_home_prob_with_market(
    p_model_home: float,
    away_ml: float | None,
    home_ml: float | None,
    market_weight: float,
) -> float:
    """
    Convex blend of model ``P(home win)`` with **de-vigged** two-way moneyline fair ``p_home``.

    ``market_weight`` in ``[0, 1]`` — 0 leaves the model unchanged.
    """
    w = float(np.clip(market_weight, 0.0, 1.0))
    if w <= 0.0 or away_ml is None or home_ml is None:
        return float(p_model_home)
    _, ph_fair = fair_probs_two_way_moneyline(float(away_ml), float(home_ml))
    if not math.isfinite(ph_fair):
        return float(p_model_home)
    out = (1.0 - w) * float(p_model_home) + w * float(ph_fair)
    return float(np.clip(out, 0.02, 0.98))


def american_to_decimal(american: float) -> float:
    o = float(american)
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return 1.0 + o / 100.0


def kelly_fraction(p_win: float, american: float) -> float:
    if p_win is None or math.isnan(p_win) or p_win <= 0.0:
        return 0.0
    p_win = float(p_win)
    if p_win >= 1.0:
        p_win = 0.9999
    D = american_to_decimal(american)
    if D <= 1.001:
        return 0.0
    b = D - 1.0
    q = 1.0 - p_win
    num = p_win * b - q
    if num <= 0.0:
        return 0.0
    return num / b


def suggest_stakes_quarter_kelly(
    bankroll: float,
    model_probs: list[float],
    american_odds: list[float],
    *,
    kelly_scale: float = 0.25,
    cap_per_bet: float = 0.15,
) -> list[float]:
    if bankroll <= 0.0:
        return [0.0] * len(model_probs)
    raw: list[float] = []
    for p, o in zip(model_probs, american_odds):
        k = kelly_fraction(p, o)
        stake = bankroll * kelly_scale * max(0.0, k)
        stake = min(stake, bankroll * cap_per_bet)
        raw.append(max(0.0, stake))
    total = sum(raw)
    if total > bankroll and total > 0.0:
        scale = bankroll / total
        raw = [s * scale for s in raw]
    return raw
