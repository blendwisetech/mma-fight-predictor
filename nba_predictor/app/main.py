"""
NBA Prediction App — Streamlit entrypoint.

Run from the ``nba_predictor`` directory:
  pip install -r requirements.txt
  streamlit run app/main.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from models.ml_predict_nba import load_win_bundle
from models.nba_win_prob import predict_home_win
from utils.betting import american_to_implied_prob, blend_home_prob_with_market, suggest_stakes_quarter_kelly
from utils.data_io import append_predictions_df, load_eval_report, load_registry, save_registry, upsert_outcomes, utc_now_iso
from utils.feature_engineering_nba import enrich_games_nba
from utils.nevada_odds_nba import fetch_nba_bovada_slate, fetch_wnba_bovada_slate, lookup_bovada_row, moneyline_for_pick
from utils.nba_core import (
    NBA_LEAGUE_ID,
    WNBA_LEAGUE_ID,
    fetch_scoreboard_games,
    season_str_for_league,
)
from utils.nba_outcomes import scoreboard_to_outcomes
from utils.pipeline_runner import start_auto_pipeline_background

_PACIFIC = ZoneInfo("America/Los_Angeles")


def _slate_league_config(choice: str) -> tuple[str, str, str, str]:
    """
    From UI league label → ``(stats.nba league_id, injury ESPN slug, bovada slug, predictions csv league)``.
    """
    if choice.strip().upper() == "WNBA":
        return WNBA_LEAGUE_ID, "wnba", "wnba", "wnba"
    return NBA_LEAGUE_ID, "nba", "nba", "nba"


def _format_start_utc(iso_utc: str | None) -> str:
    if not iso_utc:
        return "—"
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        pt = dt.astimezone(_PACIFIC)
        h24 = pt.hour
        h12 = h24 % 12 or 12
        am_pm = "AM" if h24 < 12 else "PM"
        return f"{h12}:{pt.minute:02d} {am_pm} PT"
    except (ValueError, OSError, OverflowError):
        return "—"


@st.cache_data(ttl=180, show_spinner="Loading Bovada lines…")
def _cached_bovada(slate_date: date, refresh_nonce: int, bovada_key: str) -> list[dict]:
    if bovada_key == "wnba":
        return fetch_wnba_bovada_slate(slate_date)
    return fetch_nba_bovada_slate(slate_date)


def _winner_pick_team(row: pd.Series, ph: float) -> str:
    """Side the model favors (``ph`` = P(home wins))."""
    home = str(row.get("home_display") or "")
    away = str(row.get("away_display") or "")
    if ph >= 0.5:
        return home
    return away


def _final_home_prob(
    row: pd.Series,
    bundle: dict | None,
    blend_ml: float,
    br: dict | None,
    market_weight: float,
) -> tuple[float, str, float | None]:
    """Model layer then optional de-vig moneyline blend toward fair ``P(home)``."""
    ph, ver, raw_ml = predict_home_win(row, bundle, blend_ml_weight=blend_ml)
    aw = br.get("away_ml") if br else None
    hw = br.get("home_ml") if br else None
    ph = blend_home_prob_with_market(ph, aw, hw, market_weight)
    return ph, ver, raw_ml


def _show_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("(no rows)")
        return
    st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True)


def _fmt_am(x: object) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and pd.isna(x):
            return "—"
        v = int(round(float(x)))
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"


def _model_prob_for_pick(row: pd.Series, pick: str) -> float:
    home = str(row.get("home_display") or "")
    away = str(row.get("away_display") or "")
    ph = float(row.get("pred_home_win_prob") or 0.5)
    if not pick or pick == "—":
        return float("nan")
    if pick == home:
        return float(ph)
    if pick == away:
        return float(1.0 - ph)
    return float("nan")


def _bet_sizing_section(
    pr: pd.DataFrame,
    slate_date: date,
    pick_col: str = "pick",
    *,
    bovada_key: str = "nba",
) -> None:
    if pr.empty or "game_id" not in pr.columns:
        return

    label = "WNBA" if bovada_key == "wnba" else "NBA"
    st.subheader("Bet sizing — bankroll vs Bovada")
    st.caption(
        f"Moneylines load from **Bovada.lv**’s public {label} coupon JSON (same feed as the Slate table). "
        "The feed keys off **America/New_York** calendar dates — use the same calendar date you see on Bovada for best matching. "
        "Educational only — not betting advice."
    )
    c1, c2 = st.columns([1.4, 1.0])
    with c2:
        if st.button("Refresh lines", key="bb_odds_refresh"):
            st.session_state["bb_odds_nonce"] = int(st.session_state.get("bb_odds_nonce", 0)) + 1
            st.rerun()

    events: list[dict] = []
    odds_err: str | None = None
    try:
        nonce = int(st.session_state.get("bb_odds_nonce", 0))
        events = _cached_bovada(slate_date, nonce, bovada_key)
    except Exception as e:
        odds_err = str(e)
        events = []
    if odds_err:
        st.warning(f"Could not load Bovada lines: {odds_err}")

    bankroll = st.number_input(
        "Bankroll for this slate ($)",
        min_value=0.0,
        value=100.0,
        step=25.0,
        key="bb_bankroll",
    )

    rows_edit: list[dict] = []
    for _, r in pr.iterrows():
        home = str(r.get("home_display") or "")
        away = str(r.get("away_display") or "")
        pick = str(r.get(pick_col) or "—")
        ao = moneyline_for_pick(events, away, home, pick) if events else None
        mwp = _model_prob_for_pick(r, pick)
        rows_edit.append(
            {
                "game_id": str(r["game_id"]),
                "Matchup": f"{away} @ {home}",
                "Pick": pick,
                "Model win %": round(100 * mwp, 1) if not math.isnan(mwp) else float("nan"),
                "American odds (pick)": int(round(ao)) if ao is not None else -110,
            }
        )
    edit_src = pd.DataFrame(rows_edit)
    edited_df = st.data_editor(
        edit_src,
        key="bb_betsize_editor",
        hide_index=True,
        column_config={
            "game_id": st.column_config.TextColumn("game_id", disabled=True),
            "Matchup": st.column_config.TextColumn("Matchup", disabled=True),
            "Pick": st.column_config.TextColumn("Pick", disabled=True),
            "Model win %": st.column_config.NumberColumn("Model win %", disabled=True, format="%.1f"),
            "American odds (pick)": st.column_config.NumberColumn("American odds (pick)", step=1, format="%d"),
        },
        use_container_width=True,
    )
    rows_edit = edited_df.to_dict("records")

    model_probs: list[float] = []
    american_list: list[float] = []
    out_rows: list[dict] = []
    for row in rows_edit:
        o_raw = float(row.get("American odds (pick)") or -110.0)
        o_eff = o_raw if o_raw != 0.0 else -110.0
        pick = str(row.get("Pick") or "—")
        gid = str(row.get("game_id") or "")
        src = pr.loc[pr["game_id"].astype(str) == gid]
        mwp = float("nan")
        if not src.empty:
            mwp = _model_prob_for_pick(src.iloc[0], pick)
        impl = american_to_implied_prob(o_eff)
        edge = (mwp - impl) * 100.0 if not math.isnan(mwp) and not math.isnan(impl) else float("nan")
        model_probs.append(mwp if not math.isnan(mwp) else 0.0)
        american_list.append(o_eff)
        out_rows.append(
            {
                "Matchup": row.get("Matchup"),
                "Pick": pick,
                "American odds": int(round(o_eff)),
                "Implied win %": round(100.0 * impl, 1) if not math.isnan(impl) else float("nan"),
                "Model win %": round(100.0 * mwp, 1) if not math.isnan(mwp) else float("nan"),
                "Edge %": round(edge, 2) if not math.isnan(edge) else float("nan"),
            }
        )

    stakes = suggest_stakes_quarter_kelly(bankroll, model_probs, american_list)
    for i, stake in enumerate(stakes):
        if i < len(out_rows):
            out_rows[i]["Suggested stake ($)"] = round(stake, 2)
    st.markdown("**Suggested allocation (¼ Kelly, capped)**")
    _show_table(pd.DataFrame(out_rows))


def _run_cli(module: str) -> tuple[int, str, str]:
    r = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return r.returncode, r.stdout or "", r.stderr or ""


def main() -> None:
    if os.environ.get("SPORTS_HUB") != "1":
        st.set_page_config(page_title="NBA / WNBA Predictor", layout="wide")
    st.title("Pro basketball game predictions")
    st.caption(
        "Pick **NBA** or **WNBA**, **Log predictions** after loading a slate, then **Fetch final scores** on dates "
        "that finished. Ratings use optional **E_* blend + GP shrink**; lines can **anchor to fair ML** when Bovada "
        "matches. Separate Parquets (**training_games.parquet** / **training_games_wnba.parquet**) feed sklearn."
    )

    reg = load_registry()
    prod = reg.setdefault("production", {})
    default_blend_nba = float(prod.get("ml_blend_weight", 0.0) or 0.0)
    default_blend_wnba = float(prod.get("ml_blend_weight_wnba", 0.0) or 0.0)

    with st.sidebar:
        st.header("Model")
        blend_nba = st.slider(
            "ML ↔ heuristic blend (NBA)",
            min_value=0.0,
            max_value=1.0,
            value=float(np.clip(default_blend_nba, 0.0, 1.0)),
            help="0 = heuristic only; 1 = sklearn only after train_win_model_nba. Saved to registry.",
        )
        if blend_nba != default_blend_nba:
            prod["ml_blend_weight"] = float(blend_nba)
            save_registry(reg)
            st.caption("Saved NBA blend to registry.")

        blend_wnba = st.slider(
            "ML ↔ heuristic blend (WNBA)",
            min_value=0.0,
            max_value=1.0,
            value=float(np.clip(default_blend_wnba, 0.0, 1.0)),
            help="0 = heuristic until you run train_win_model_wnba; then mix with the WNBA logreg bundle.",
        )
        if blend_wnba != default_blend_wnba:
            prod["ml_blend_weight_wnba"] = float(blend_wnba)
            save_registry(reg)
            st.caption("Saved WNBA blend to registry.")

        season_type = st.selectbox(
            "Team stats sample",
            options=["Regular Season", "Playoffs"],
            index=0,
            help="Which stats.nba.com slice to use for season advanced rates (both leagues).",
        )
        timeout = st.slider("HTTP timeout (s)", 30, 180, 120)
        rest_scan = st.slider(
            "Rest-day scan (calendar days back)",
            min_value=3,
            max_value=14,
            value=7,
            help="How far back to search scoreboards for each team’s last game before this slate.",
        )

        st.divider()
        st.subheader("Rating hygiene")
        expected_rating_blend_ui = st.slider(
            "Box ↔ tracking (E_NET / E_OFF / E_DEF) blend",
            min_value=0.0,
            max_value=0.6,
            value=0.2,
            step=0.05,
            help="Mix raw efficiency ratings with stats.nba.com expected ratings before GP shrink.",
        )
        rating_shrink_k = st.slider(
            "GP shrink prior (pseudo games)",
            min_value=0.0,
            max_value=25.0,
            value=8.0,
            step=0.5,
            help="Pull ratings toward league average — stronger early season. 0 disables.",
        )

        st.divider()
        st.subheader("Market anchoring")
        market_blend_home = st.slider(
            "Model P(home) ↔ fair moneyline blend",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help="When both moneylines match on Bovada, blend toward de-vigged two-way fair probability.",
        )

        st.divider()
        st.subheader("ML maintenance")
        st.checkbox(
            "Auto merge / train / eval after logging predictions or outcomes",
            value=True,
            key="auto_pipeline_bg",
            help="Runs NBA pipeline then WNBA: merge → train (if ≥ min rows) → eval; logged to data/processed/auto_pipeline.log.",
        )
        train_cfg = load_registry().get("training") or {}
        _min_nba = int(train_cfg.get("min_rows_win", 40))
        _min_wnba = int(train_cfg.get("min_rows_win_wnba", _min_nba))
        st.caption(
            f"NBA train threshold ≥ **`{_min_nba}`** · WNBA ≥ **`{_min_wnba}`** "
            f"(`registry.json` → `training.min_rows_*`)."
        )

        st.subheader("Recent evaluation")
        ev_nba = load_eval_report(league="nba")
        ev_wnba = load_eval_report(league="wnba")
        if ev_nba:
            with st.expander("NBA — eval_report.json (truncated)", expanded=False):
                txt = json.dumps(ev_nba, indent=2, default=str)
                st.code(txt[:12000] + ("\n… (truncated)" if len(txt) > 12000 else ""), language="json")
        else:
            st.caption("NBA eval: run **Evaluate models** or the auto pipeline.")
        if ev_wnba:
            with st.expander("WNBA — eval_report_wnba.json (truncated)", expanded=False):
                txt = json.dumps(ev_wnba, indent=2, default=str)
                st.code(txt[:12000] + ("\n… (truncated)" if len(txt) > 12000 else ""), language="json")
        else:
            st.caption("WNBA eval: appears after merged WNBA rows + **Evaluate models (WNBA)**.")

    tab_slate, tab_odds, tab_data = st.tabs(["Slate & projections", "Odds & sizing", "Data & training"])

    with tab_slate:
        c1, c2, c3, _ = st.columns([1, 1, 1, 1])
        with c1:
            slate_date = st.date_input("Slate date", value=date.today())
        with c2:
            slate_league = st.selectbox("League", ["NBA", "WNBA"], index=0)
        with c3:
            show_final = st.checkbox("Include finished games", value=False)

        stats_lid, inj_lg, bov_key, pred_lg = _slate_league_config(slate_league)
        season = season_str_for_league(stats_lid, slate_date)
        league_label = "WNBA" if pred_lg == "wnba" else "NBA"
        st.caption(f"{league_label} season label (advanced tables): **{season}** · stats: **{season_type}**")

        if st.button("Load slate + stats", type="primary"):
            with st.spinner(f"Calling stats.nba.com ({league_label}, can take 15–60s)…"):
                raw = fetch_scoreboard_games(slate_date, timeout=int(timeout), league_id=stats_lid)
            st.session_state["bb_raw_slate"] = raw
            st.session_state["bb_slate_date"] = slate_date
            st.session_state["bb_season_type"] = season_type
            st.session_state["bb_timeout"] = int(timeout)
            st.session_state["bb_stats_lid"] = stats_lid
            st.session_state["bb_inj_lg"] = inj_lg
            st.session_state["bb_bov_key"] = bov_key
            st.session_state["bb_pred_lg"] = pred_lg
            st.session_state["bb_season_str"] = season

        raw = st.session_state.get("bb_raw_slate")
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            st.info('Pick league + date and press **Load slate + stats**.')
        else:
            slate_date = st.session_state.get("bb_slate_date", slate_date)
            season_type = st.session_state.get("bb_season_type", season_type)
            timeout = int(st.session_state.get("bb_timeout", timeout))
            stats_lid = str(st.session_state.get("bb_stats_lid", NBA_LEAGUE_ID))
            inj_lg = str(st.session_state.get("bb_inj_lg", "nba"))
            bov_key = str(st.session_state.get("bb_bov_key", "nba"))
            pred_lg = str(st.session_state.get("bb_pred_lg", "nba"))
            season = str(st.session_state.get("bb_season_str", season_str_for_league(stats_lid, slate_date)))
            is_wnba = pred_lg == "wnba"

            games = raw.copy()
            if not show_final:
                games = games.loc[games["game_status_text"].astype(str).str.lower() != "final"].copy()

            if games.empty:
                st.warning("No games match filters (try including finished games, or another date).")
            else:
                with st.spinner("Enriching with team rates, rest days, ESPN injury load…"):
                    enriched = enrich_games_nba(
                        games,
                        season,
                        slate_date,
                        season_type=season_type,
                        timeout=timeout,
                        rest_scan_days=int(rest_scan),
                        league_id=stats_lid,
                        injury_league=inj_lg,
                        expected_rating_blend=float(expected_rating_blend_ui),
                        rating_shrink_gp_prior=float(rating_shrink_k),
                    )

                reg_sl = load_registry()
                win_bundle = load_win_bundle(reg_sl, "wnba" if is_wnba else "nba")
                pc = reg_sl.get("production") or {}
                blend_effective = float(
                    pc.get(
                        "ml_blend_weight_wnba" if is_wnba else "ml_blend_weight",
                        blend_wnba if is_wnba else blend_nba,
                    )
                )

                try:
                    bov_rows = _cached_bovada(
                        slate_date, int(st.session_state.get("bb_odds_nonce", 0)), bov_key
                    )
                except Exception as ex:
                    bov_rows = []
                    st.warning(f"Bovada lines unavailable: {ex}")

                slate_rows: list[dict] = []
                pick_list: list[str] = []
                ph_list: list[float] = []
                ver_list: list[str] = []

                for _, row in enriched.iterrows():
                    away = str(row["away_display"])
                    home = str(row["home_display"])
                    br = lookup_bovada_row(bov_rows, away, home) if bov_rows else None
                    ph, ver, _ = _final_home_prob(
                        row, win_bundle, blend_effective, br, market_blend_home
                    )
                    pick_team = _winner_pick_team(row, ph)
                    pick_list.append(pick_team)
                    ph_list.append(ph)
                    ver_list.append(ver)

                    ml_pick = (
                        moneyline_for_pick(bov_rows, away, home, pick_team) if bov_rows else None
                    )
                    sp_txt = "—"
                    tot_txt = "—"
                    if br:
                        hspl = br.get("home_spread_line")
                        aspl = br.get("away_spread_line")
                        if hspl is not None and aspl is not None:
                            sp_txt = (
                                f"H {hspl:g} ({_fmt_am(br.get('home_spread_am'))}) · "
                                f"A {aspl:g} ({_fmt_am(br.get('away_spread_am'))})"
                            )
                        tot = br.get("total_line")
                        if tot is not None:
                            tot_txt = f"{tot:g} · O {_fmt_am(br.get('over_am'))} / U {_fmt_am(br.get('under_am'))}"

                    slate_rows.append(
                        {
                            "Start (PT)": _format_start_utc(str(row.get("gameTimeUTC") or "")),
                            "Matchup": f"{away} @ {home}",
                            "Status": str(row.get("game_status_text") or ""),
                            "P (home) %": round(100 * ph, 1),
                            "P (away) %": round(100 * (1.0 - ph), 1),
                            "Model pick": pick_team,
                            "ML (pick)": _fmt_am(ml_pick),
                            "Away ML": _fmt_am(br.get("away_ml")) if br else "—",
                            "Home ML": _fmt_am(br.get("home_ml")) if br else "—",
                            "Spread": sp_txt,
                            "Total": tot_txt,
                            "Model": ver,
                        }
                    )

                st.subheader("Slate — model pick & Bovada")
                st.caption(
                    "**Model pick** uses blended win probability (ML layer + optional fair-ML anchor). "
                    "**ML (pick)** is Bovada’s moneyline on that side when the game matches the feed."
                )
                _show_table(pd.DataFrame(slate_rows))

                st.session_state["bb_enriched"] = enriched

                pr = enriched.copy()
                pr["pick"] = pick_list
                pr["pred_home_win_prob"] = ph_list
                pr["pred_version"] = ver_list

                if st.button("Log predictions for this slate"):
                    logs: list[dict] = []
                    for (_, r), pick_team, ph, ver in zip(pr.iterrows(), pick_list, ph_list, ver_list):
                        d = r.to_dict()
                        d["logged_at"] = utc_now_iso()
                        d["season_str"] = season
                        d["season_type"] = season_type
                        d["slate_date"] = str(slate_date)
                        d["league"] = pred_lg
                        _, _, raw_ml = predict_home_win(r, win_bundle, blend_ml_weight=blend_effective)
                        d["pred_home_win_prob"] = ph
                        d["pred_away_win_prob"] = float(1.0 - ph)
                        d["pred_version"] = ver
                        d["model_pick"] = pick_team
                        if raw_ml is not None:
                            d["pred_raw_ml_home"] = raw_ml
                        logs.append(d)
                    append_predictions_df(pd.DataFrame(logs))
                    st.success(f"Appended **{len(logs)}** rows to predictions CSV.")
                    if st.session_state.get("auto_pipeline_bg", True):
                        start_auto_pipeline_background()
                        st.caption(
                            "Started background **merge → train → eval** "
                            "(see sidebar / `data/processed/auto_pipeline.log`)."
                        )

    with tab_odds:
        enriched = st.session_state.get("bb_enriched")
        slate_date = st.session_state.get("bb_slate_date", date.today())
        bov_key = str(st.session_state.get("bb_bov_key", "nba"))
        pred_lg = str(st.session_state.get("bb_pred_lg", "nba"))
        is_wnba = pred_lg == "wnba"
        if enriched is None or enriched.empty:
            st.info("Load a slate on the first tab first.")
        else:
            pr = enriched.copy()
            reg_od = load_registry()
            win_bundle = load_win_bundle(reg_od, "wnba" if is_wnba else "nba")
            pc = reg_od.get("production") or {}
            blend_effective = float(
                pc.get(
                    "ml_blend_weight_wnba" if is_wnba else "ml_blend_weight",
                    blend_wnba if is_wnba else blend_nba,
                )
            )
            try:
                events = _cached_bovada(
                    slate_date, int(st.session_state.get("bb_odds_nonce", 0)), bov_key
                )
            except Exception:
                events = []
            ph_list: list[float] = []
            pick_list: list[str] = []
            for _, r in pr.iterrows():
                away = str(r.get("away_display") or "")
                home = str(r.get("home_display") or "")
                br = lookup_bovada_row(events, away, home) if events else None
                ph, _, _ = _final_home_prob(r, win_bundle, blend_effective, br, market_blend_home)
                ph_list.append(ph)
                pick_list.append(_winner_pick_team(r, ph))
            pr["pred_home_win_prob"] = ph_list
            pr["pick"] = pick_list
            _bet_sizing_section(pr, slate_date, bovada_key=bov_key)

    with tab_data:
        st.subheader("Outcomes & training")
        st.caption(
            "Fetches **final** games for the selected league using the same stats.nba.com scoreboard as the slate tab "
            "(set league by loading a slate, or pick below)."
        )
        od = st.date_input("Outcome fetch date", value=date.today(), key="bb_out_date")
        out_league = st.selectbox(
            "Outcome league",
            ["NBA", "WNBA"],
            index=1 if str(st.session_state.get("bb_pred_lg", "nba")) == "wnba" else 0,
            key="bb_outcome_league",
        )
        stats_lid_o, _, _, slug_o = _slate_league_config(out_league)
        if st.button("Fetch final scores for that date"):
            with st.spinner("stats.nba.com scoreboard…"):
                out_df = scoreboard_to_outcomes(
                    od,
                    timeout=120,
                    league_id=stats_lid_o,
                    league_slug=slug_o,
                )
            if out_df.empty:
                st.warning("No final games for that date (yet).")
            else:
                n = upsert_outcomes(out_df)
                st.success(f"Upserted **{n}** outcome rows.")
                if st.session_state.get("auto_pipeline_bg", True) and n > 0:
                    start_auto_pipeline_background()
                    st.caption("Started background **merge → train → eval**.")

        st.markdown(
            "**Logging workflow:** load slate → **Log predictions** → when games finish, **Fetch final scores** "
            "(matching league). Auto pipeline merges **NBA** and **WNBA** separately, trains each model only "
            "when row counts hit `training.min_rows_win` / `min_rows_win_wnba`, then writes eval JSON per league."
        )
        st.markdown(
            "**Retune heuristic vs separate sklearn:** use **Evaluate** metrics on merged rows — if the heuristic "
            "is competitive, tweak coefficients in `models/nba_win_prob.py` with fewer rows; if sklearn wins clearly, "
            "raise **ML blend** for that league after training."
        )
        _maint = {
            "Merge predictions + outcomes (NBA)": "ml.merge_training_data_nba",
            "Train win model (NBA)": "ml.train_win_model_nba",
            "Evaluate models (NBA)": "ml.evaluate_models_nba",
            "Merge predictions + outcomes (WNBA)": "ml.merge_training_data_wnba",
            "Train win model (WNBA)": "ml.train_win_model_wnba",
            "Evaluate models (WNBA)": "ml.evaluate_models_wnba",
            "Full ML bootstrap (NBA + WNBA steps)": "ml.bootstrap_models",
            "Auto pipeline only (logged)": "ml.auto_pipeline",
        }
        with st.form("nba_maint_form"):
            choice = st.selectbox("Run one step", list(_maint.keys()))
            run_one = st.form_submit_button("Run selected")
        if run_one:
            mod = _maint[choice]
            code, so, se = _run_cli(mod)
            st.code((so + "\n" + se).strip() or f"(no output, exit {code})", language="text")
            if code != 0:
                st.error(f"Exit code {code}")

        st.markdown("**CLI (same as maintenance form)**")
        st.code(
            "python -m ml.auto_pipeline\n# or\npython -m ml.bootstrap_models",
            language="bash",
        )


if __name__ == "__main__":
    main()
