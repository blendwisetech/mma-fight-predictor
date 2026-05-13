"""
Strictly **pre-fight** features from a chronological walk over bouts.

Combines:

- **Record book** (wins/losses/days off/bout count) from outcomes.
- **True rate features** from ``events_raw`` per-bout totals (merged onto each row):
  cumulative sig strikes (landed/absorbed), TDs, KD, sub attempts, scaled by **prior**
  cage minutes (or bout counts for submission averages).

**Labels**: official ``outcome`` (``fighter1`` / ``fighter2``) vs hash-assigned **Fighter A/B**
(see ``feature_engineering_mma.side_ab_for_fight``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.feature_engineering_mma import (
    fight_id_for_row,
    normalize_name,
    side_ab_for_fight,
    weight_class_index,
)


@dataclass
class _Rec:
    wins: int = 0
    losses: int = 0
    fights: int = 0
    last_dt: pd.Timestamp | None = None
    cum_min: float = 0.0
    cum_sig_land: float = 0.0
    cum_sig_abs: float = 0.0
    cum_td_land: float = 0.0
    cum_td_abs: float = 0.0
    cum_kd_land: float = 0.0
    cum_kd_abs: float = 0.0
    cum_sub: float = 0.0


class PreFightBuilder:
    """Per-fighter state; call ``ingest_after_row`` once per chronological bout after featurizing."""

    def __init__(self) -> None:
        self._st: dict[str, _Rec] = {}

    def _rec(self, display_name: str) -> _Rec:
        k = normalize_name(display_name)
        if k not in self._st:
            self._st[k] = _Rec()
        return self._st[k]

    def _win_rate(self, r: _Rec) -> float:
        d = r.wins + r.losses
        if d <= 0:
            return float("nan")
        return float(r.wins / d)

    def _days_since(self, r: _Rec, event_dt: pd.Timestamp) -> float:
        if r.last_dt is None or pd.isna(event_dt):
            return float("nan")
        return float((event_dt - r.last_dt).days)

    @staticmethod
    def _slpm(rec: _Rec) -> float:
        if rec.cum_min <= 1e-9:
            return float("nan")
        return float(rec.cum_sig_land / rec.cum_min)

    @staticmethod
    def _sapm(rec: _Rec) -> float:
        if rec.cum_min <= 1e-9:
            return float("nan")
        return float(rec.cum_sig_abs / rec.cum_min)

    @staticmethod
    def _td_per15(rec: _Rec) -> float:
        if rec.cum_min <= 1e-9:
            return float("nan")
        return float((rec.cum_td_land / rec.cum_min) * 15.0)

    @staticmethod
    def _kd_per_min(rec: _Rec) -> float:
        if rec.cum_min <= 1e-9:
            return float("nan")
        return float(rec.cum_kd_land / rec.cum_min)

    @staticmethod
    def _sub_per_fight(rec: _Rec) -> float:
        if rec.fights <= 0:
            return float("nan")
        return float(rec.cum_sub / rec.fights)

    def featurize(
        self,
        r: pd.Series,
        *,
        wc_map: dict[str, int],
        include_label: bool,
    ) -> dict[str, object] | None:
        o = str(r.get("outcome") or "").strip().lower()
        if include_label and o not in ("fighter1", "fighter2"):
            return None

        f1 = str(r.get("fighter1") or "").strip()
        f2 = str(r.get("fighter2") or "").strip()
        wc = str(r.get("weight_class") or "").strip()
        ed = pd.to_datetime(r.get("event_date"), errors="coerce")
        if pd.isna(ed) or not f1 or not f2:
            return None

        d_str = ed.strftime("%Y-%m-%d")
        name_a, name_b, a_is_f1 = side_ab_for_fight(f1, f2, d_str, wc)

        ra = self._rec(name_a)
        rb = self._rec(name_b)

        h1 = _float_cell(r.get("fighter1_height"))
        h2 = _float_cell(r.get("fighter2_height"))
        r1c = _float_cell(r.get("fighter1_reach"))
        r2c = _float_cell(r.get("fighter2_reach"))
        a1 = _age_years(r.get("fighter1_dob"), ed)
        a2 = _age_years(r.get("fighter2_dob"), ed)
        if a_is_f1:
            h_a, h_b, reach_a, reach_b, age_a, age_b = h1, h2, r1c, r2c, a1, a2
        else:
            h_a, h_b, reach_a, reach_b, age_a, age_b = h2, h1, r2c, r1c, a2, a1

        w_a, w_b = ra.wins, rb.wins
        l_a, l_b = ra.losses, rb.losses
        n_a, n_b = ra.fights, rb.fights
        wr_a, wr_b = self._win_rate(ra), self._win_rate(rb)
        ds_a = self._days_since(ra, ed)
        ds_b = self._days_since(rb, ed)

        slpm_a, slpm_b = self._slpm(ra), self._slpm(rb)
        sapm_a, sapm_b = self._sapm(ra), self._sapm(rb)
        td15_a, td15_b = self._td_per15(ra), self._td_per15(rb)
        kdpm_a, kdpm_b = self._kd_per_min(ra), self._kd_per_min(rb)
        sub_a, sub_b = self._sub_per_fight(ra), self._sub_per_fight(rb)

        is_wm = 1.0 if "women" in wc.lower() else 0.0
        wc_ix = float(weight_class_index(wc, wc_map))

        row: dict[str, object] = {
            "fight_id": fight_id_for_row(r),
            "event_date": d_str,
            "event_name": r.get("event_name"),
            "weight_class": wc,
            "orig_fighter1": f1,
            "orig_fighter2": f2,
            "fighter_a": name_a,
            "fighter_b": name_b,
            "f_is_womens": is_wm,
            "f_wc_index": wc_ix,
            "f_a_height_cm": h_a,
            "f_b_height_cm": h_b,
            "f_a_reach_cm": reach_a,
            "f_b_reach_cm": reach_b,
            "f_a_age_y": age_a,
            "f_b_age_y": age_b,
            "f_a_wins_before": float(w_a),
            "f_b_wins_before": float(w_b),
            "f_a_losses_before": float(l_a),
            "f_b_losses_before": float(l_b),
            "f_a_fights_before": float(n_a),
            "f_b_fights_before": float(n_b),
            "f_a_win_rate_before": wr_a,
            "f_b_win_rate_before": wr_b,
            "f_a_days_since_last_fight": ds_a,
            "f_b_days_since_last_fight": ds_b,
            "f_a_slpm_before": slpm_a,
            "f_b_slpm_before": slpm_b,
            "f_a_sapm_before": sapm_a,
            "f_b_sapm_before": sapm_b,
            "f_a_td_per15_before": td15_a,
            "f_b_td_per15_before": td15_b,
            "f_a_kd_per_min_before": kdpm_a,
            "f_b_kd_per_min_before": kdpm_b,
            "f_a_sub_per_fight_before": sub_a,
            "f_b_sub_per_fight_before": sub_b,
            "f_height_diff_cm": _diff(h_a, h_b),
            "f_reach_diff_cm": _diff(reach_a, reach_b),
            "f_age_diff_y": _diff(age_a, age_b),
            "f_wins_before_diff": _diff(float(w_a), float(w_b)),
            "f_losses_before_diff": _diff(float(l_a), float(l_b)),
            "f_fights_before_diff": _diff(float(n_a), float(n_b)),
            "f_win_rate_before_diff": _diff(wr_a, wr_b),
            "f_days_since_last_fight_diff": _diff(ds_a, ds_b),
            "f_slpm_diff": _diff(slpm_a, slpm_b),
            "f_sapm_diff": _diff(sapm_a, sapm_b),
            "f_td15_diff": _diff(td15_a, td15_b),
            "f_kdpm_diff": _diff(kdpm_a, kdpm_b),
            "f_sub_pf_diff": _diff(sub_a, sub_b),
        }

        if include_label:
            wnorm = normalize_name(f1 if o == "fighter1" else f2)
            row["fighter_a_win"] = int(wnorm == normalize_name(name_a))
            row["outcome"] = o

        return row

    def ingest_after_row(self, r: pd.Series) -> None:
        f1 = str(r.get("fighter1") or "").strip()
        f2 = str(r.get("fighter2") or "").strip()
        ed = pd.to_datetime(r.get("event_date"), errors="coerce")
        if pd.isna(ed) or not f1 or not f2:
            return

        dur = _float_cell(r.get("bout_duration_min"))
        s1 = _float_cell(r.get("f1_sig_str_landed"))
        s2 = _float_cell(r.get("f2_sig_str_landed"))
        t1 = _float_cell(r.get("f1_td"))
        t2 = _float_cell(r.get("f2_td"))
        k1 = _float_cell(r.get("f1_kd"))
        k2 = _float_cell(r.get("f2_kd"))
        b1 = _float_cell(r.get("f1_sub_att"))
        b2 = _float_cell(r.get("f2_sub_att"))

        if np.isfinite(dur) and dur > 0:
            p1, p2 = self._rec(f1), self._rec(f2)
            if np.isfinite(s1) and np.isfinite(s2):
                p1.cum_sig_land += float(s1)
                p1.cum_sig_abs += float(s2)
                p2.cum_sig_land += float(s2)
                p2.cum_sig_abs += float(s1)
            if np.isfinite(t1) and np.isfinite(t2):
                p1.cum_td_land += float(t1)
                p1.cum_td_abs += float(t2)
                p2.cum_td_land += float(t2)
                p2.cum_td_abs += float(t1)
            if np.isfinite(k1) and np.isfinite(k2):
                p1.cum_kd_land += float(k1)
                p1.cum_kd_abs += float(k2)
                p2.cum_kd_land += float(k2)
                p2.cum_kd_abs += float(k1)
            if np.isfinite(b1):
                p1.cum_sub += float(b1)
            if np.isfinite(b2):
                p2.cum_sub += float(b2)
            p1.cum_min += float(dur)
            p2.cum_min += float(dur)

        o = str(r.get("outcome") or "").strip().lower()
        if o == "fighter1":
            self._rec(f1).wins += 1
            self._rec(f2).losses += 1
        elif o == "fighter2":
            self._rec(f2).wins += 1
            self._rec(f1).losses += 1

        for nm in (f1, f2):
            p = self._rec(nm)
            p.fights += 1
            p.last_dt = ed


def _float_cell(v: object) -> float:
    if v is None:
        return float("nan")
    try:
        if isinstance(v, float) and np.isnan(v):
            return float("nan")
    except TypeError:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _age_years(dob_raw: object, fight_dt: pd.Timestamp) -> float:
    dob = pd.to_datetime(dob_raw, errors="coerce")
    if pd.isna(dob) or pd.isna(fight_dt):
        return float("nan")
    return float((fight_dt - dob).days / 365.25)


def _diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def build_prefight_training_table(raw: pd.DataFrame, wc_map: dict[str, int]) -> pd.DataFrame:
    df = raw.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df["_idx"] = np.arange(len(df))
    df = df.sort_values(["event_date", "event_name", "_idx"], kind="mergesort")

    b = PreFightBuilder()
    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        feat = b.featurize(r, wc_map=wc_map, include_label=True)
        if feat is not None:
            rows.append(feat)
        b.ingest_after_row(r)
    return pd.DataFrame(rows)


def walk_history_until_date(sorted_raw: pd.DataFrame, cutoff: pd.Timestamp, builder: PreFightBuilder) -> None:
    for _, r in sorted_raw.iterrows():
        ed = pd.to_datetime(r.get("event_date"), errors="coerce")
        if pd.isna(ed):
            continue
        if ed.normalize() < cutoff.normalize():
            builder.ingest_after_row(r)


def featurize_slate_with_builder(
    builder: PreFightBuilder,
    slate_rows: pd.DataFrame,
    wc_map: dict[str, int],
    *,
    ingest_after_each: bool = True,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for _, r in slate_rows.iterrows():
        feat = builder.featurize(r, wc_map=wc_map, include_label=False)
        if feat is None:
            continue
        out.append(feat)
        if ingest_after_each:
            builder.ingest_after_row(r)
    return out
