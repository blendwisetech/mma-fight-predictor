"""
Identifiers, hashing-based **Fighter A / Fighter B** assignment, and heuristics.

The source CSV lists the winner in ``fighter1`` for most rows (``outcome == "fighter1"``).
We still read ``outcome`` as the official corner result (``fighter1`` / ``fighter2``) and
assign **A/B** via a stable hash so the classification target is not label leakage from column order.

Strict **pre-fight** numeric features are built in ``utils.prefight_history`` (walk over past bouts).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from utils.data_io import PATH_WEIGHT_CLASS_MAP, ensure_dirs


def normalize_name(s: object) -> str:
    return " ".join(str(s or "").strip().lower().split())


def side_ab_for_fight(f1: str, f2: str, event_date: str, weight_class: str) -> tuple[str, str, bool]:
    """
    Deterministic random side: returns ``(name_a, name_b, a_is_original_f1)``.
    Same inputs as training/inference when strings match the CSV.
    """
    key = f"{normalize_name(f1)}|{normalize_name(f2)}|{event_date}|{str(weight_class or '').strip()}"
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    if h % 2 == 0:
        return f1, f2, True
    return f2, f1, False


def fight_id_for_row(row: pd.Series) -> str:
    ed = row.get("event_date")
    try:
        if ed is None or (isinstance(ed, float) and np.isnan(ed)):
            ed_s = ""
        else:
            ts = pd.to_datetime(ed, errors="coerce")
            ed_s = "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    except Exception:
        ed_s = str(ed or "")
    key = "|".join(
        [
            ed_s,
            str(row.get("fighter1") or ""),
            str(row.get("fighter2") or ""),
            str(row.get("weight_class") or ""),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def load_weight_class_map() -> dict[str, int]:
    if not PATH_WEIGHT_CLASS_MAP.exists():
        return {}
    return json.loads(PATH_WEIGHT_CLASS_MAP.read_text(encoding="utf-8"))


def save_weight_class_map(mapping: dict[str, int]) -> None:
    ensure_dirs()
    PATH_WEIGHT_CLASS_MAP.parent.mkdir(parents=True, exist_ok=True)
    PATH_WEIGHT_CLASS_MAP.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def fit_weight_class_map(weight_classes: pd.Series) -> dict[str, int]:
    cats = sorted({str(x).strip() for x in weight_classes.dropna().unique() if str(x).strip()})
    return {c: i for i, c in enumerate(cats)}


def weight_class_index(wc: object, wc_map: dict[str, int]) -> float:
    if not wc_map:
        return float("nan")
    s = str(wc or "").strip()
    if not s:
        return float("nan")
    if s in wc_map:
        return float(wc_map[s])
    return float(len(wc_map))


def fighter_a_win_label_from_outcome_row(r: pd.Series) -> int | None:
    """Official label: 1 iff Fighter **A** (hash side) won."""
    o = str(r.get("outcome") or "").strip().lower()
    if o not in ("fighter1", "fighter2"):
        return None
    f1 = str(r.get("fighter1") or "").strip()
    f2 = str(r.get("fighter2") or "").strip()
    ed = pd.to_datetime(r.get("event_date"), errors="coerce")
    if pd.isna(ed) or not f1 or not f2:
        return None
    d_str = ed.strftime("%Y-%m-%d")
    wc = str(r.get("weight_class") or "").strip()
    name_a, _, _ = side_ab_for_fight(f1, f2, d_str, wc)
    wnorm = normalize_name(f1 if o == "fighter1" else f2)
    return int(wnorm == normalize_name(name_a))


def heuristic_fighter_a_win_prob(row: pd.Series) -> float:
    """Baseline aligned with pre-fight record + rate features."""
    ra = _f(row, "f_a_reach_cm")
    rb = _f(row, "f_b_reach_cm")
    aa = _f(row, "f_a_age_y")
    ab = _f(row, "f_b_age_y")
    wra = _f(row, "f_a_win_rate_before")
    wrb = _f(row, "f_b_win_rate_before")
    wa = _f(row, "f_a_wins_before")
    wb = _f(row, "f_b_wins_before")
    na = _f(row, "f_a_fights_before")
    nb = _f(row, "f_b_fights_before")
    ds = _f(row, "f_slpm_diff")
    ap = _f(row, "f_sapm_diff")
    td = _f(row, "f_td15_diff")
    kd = _f(row, "f_kdpm_diff")
    sb = _f(row, "f_sub_pf_diff")
    elo_d = _f(row, "f_elo_diff")
    wst_d = _f(row, "f_win_streak_diff")
    l3d = _f(row, "f_l3_win_rate_diff")
    krd = _f(row, "f_ko_rate_diff")
    avgd = _f(row, "f_avg_bout_min_diff")

    z = 0.12
    if np.isfinite(ra) and np.isfinite(rb):
        z += 0.028 * (ra - rb)
    if np.isfinite(aa) and np.isfinite(ab):
        z += 0.090 * (ab - aa)
    if np.isfinite(wra) and np.isfinite(wrb):
        z += 1.15 * (wra - wrb)
    if np.isfinite(wa) and np.isfinite(wb):
        z += 0.04 * (wa - wb)
    if np.isfinite(na) and np.isfinite(nb):
        z += 0.025 * (na - nb)
    if np.isfinite(ds):
        z += 0.55 * ds
    if np.isfinite(ap):
        z += -0.35 * ap
    if np.isfinite(td):
        z += 0.12 * td
    if np.isfinite(kd):
        z += 0.85 * kd
    if np.isfinite(sb):
        z += 0.18 * sb
    if np.isfinite(elo_d):
        z += 0.0018 * elo_d
    if np.isfinite(wst_d):
        z += 0.055 * wst_d
    if np.isfinite(l3d):
        z += 0.35 * l3d
    if np.isfinite(krd):
        z += 0.22 * krd
    if np.isfinite(avgd):
        z += -0.028 * avgd
    p = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _f(row: pd.Series, key: str, default: float = float("nan")) -> float:
    v = row.get(key)
    if v is None:
        return default
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
    except TypeError:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


