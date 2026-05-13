"""Helpers for betting-style displays: decimal / American conversion, de-vig, EV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.feature_engineering_mma import normalize_name


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def decimal_to_american(d: float) -> float:
    """Return American moneyline equivalent to decimal odds ``d`` (> 1). NaN if invalid."""
    if not _finite(d) or d <= 1.0:
        return float("nan")
    if d >= 2.0:
        return float(round((d - 1.0) * 100.0))
    return float(round(-100.0 / (d - 1.0)))


def prob_to_fair_decimal(p: float) -> float:
    """Fair decimal odds for win probability ``p`` in (0, 1)."""
    if not _finite(p) or p <= 0.0 or p >= 1.0:
        return float("nan")
    return 1.0 / p


def devig_implied_pair(dec_a: float, dec_b: float) -> tuple[float, float]:
    """Two-way de-vig (proportional): implied win probs for sides A and B."""
    if not _finite(dec_a) or not _finite(dec_b) or dec_a <= 1.0 or dec_b <= 1.0:
        return float("nan"), float("nan")
    ia, ib = 1.0 / dec_a, 1.0 / dec_b
    s = ia + ib
    if s <= 0.0:
        return float("nan"), float("nan")
    return ia / s, ib / s


def ev_per_unit_stake_win_bet(p_win: float, decimal_odds: float) -> float:
    """Expected profit per 1 unit staked on win-only bet: ``p * d - 1``."""
    if not _finite(p_win) or not _finite(decimal_odds) or decimal_odds <= 1.0:
        return float("nan")
    return float(p_win * decimal_odds - 1.0)


def book_decimals_for_fighters(
    raw: pd.Series,
    name_a: str,
    name_b: str,
) -> tuple[float, float]:
    """Map ``favourite`` / ``underdog`` decimal columns onto Fighter A / B. NaNs if missing."""
    fav = raw.get("favourite")
    und = raw.get("underdog")
    try:
        d_fav = float(raw.get("favourite_odds"))
        d_und = float(raw.get("underdog_odds"))
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not _finite(d_fav) or not _finite(d_und) or d_fav <= 1.0 or d_und <= 1.0:
        return float("nan"), float("nan")
    if not isinstance(fav, str) or not isinstance(und, str):
        return float("nan"), float("nan")
    if not fav.strip() or not und.strip():
        return float("nan"), float("nan")

    nf, nu = normalize_name(fav), normalize_name(und)
    na, nb = normalize_name(name_a), normalize_name(name_b)
    if nf == na and nu == nb:
        return float(d_fav), float(d_und)
    if nf == nb and nu == na:
        return float(d_und), float(d_fav)
    return float("nan"), float("nan")


def format_american(x: float) -> str:
    if not _finite(x):
        return "—"
    xi = int(round(x))
    return f"+{xi}" if xi > 0 else str(xi)


def format_ev(x: float) -> str:
    if not _finite(x):
        return "—"
    if abs(x) < 0.0005:
        return "0.00"
    return f"{x:+.3f}"
