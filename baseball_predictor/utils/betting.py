"""Odds / Kelly helpers for exploratory bet-sizing UI (not financial advice)."""

from __future__ import annotations

import math


def american_to_implied_prob(american: float) -> float:
    """Vig-free implied win probability for one side from American odds."""
    if american is None or (isinstance(american, float) and math.isnan(american)):
        return float("nan")
    o = float(american)
    if o == 0.0:
        return float("nan")
    if o < 0:
        a = abs(o)
        return a / (a + 100.0)
    return 100.0 / (o + 100.0)


def american_to_decimal(american: float) -> float:
    """Decimal odds (stake + profit on a 1-unit win) from American."""
    o = float(american)
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return 1.0 + o / 100.0


def kelly_fraction(p_win: float, american: float) -> float:
    """
    Full Kelly fraction of bankroll for a single win bet at ``american`` odds.

    Uses net fractional odds b = D - 1 with decimal price D: f* = (p*b - (1-p)) / b.
    Returns 0 when the bet is not +EV at these odds.
    """
    if p_win is None or math.isnan(p_win) or p_win <= 0.0:
        return 0.0
    p_win = float(p_win)
    # Kelly is undefined at p==1. Rounded display probs can become exactly 1.0; cap to a sane ceiling.
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
    """
    Per-row dollar stakes: quarter-Kelly (by default), each row capped at ``cap_per_bet`` × bankroll,
    then scaled down uniformly if the sum exceeds ``bankroll``.
    """
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
