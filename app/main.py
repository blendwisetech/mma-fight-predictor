"""
MMA fight predictor — Streamlit entrypoint.

Run from the ``mma_predictor`` directory::

  pip install -r requirements.txt
  streamlit run app/main.py

Cold start::

  python -m utils.ufc_historical download
  python -m ml.seed_training_from_history_mma
  python -m ml.train_win_model_mma

Continuous learning (after logging predictions on past cards)::

  python -m ml.backfill_outcomes_from_history_mma
  python -m ml.auto_pipeline
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from ml.feature_config_mma import GAME_FEATURE_NAMES
from ml.win_prob_utils import blended_prob
from models.ml_predict_mma import load_production_pipelines, predict_fighter_a_win_ml
from utils.data_io import append_predictions_df, load_eval_report, utc_now_iso
from utils.feature_engineering_mma import (
    fight_id_for_row,
    fit_weight_class_map,
    heuristic_fighter_a_win_prob,
    load_weight_class_map,
    save_weight_class_map,
)
from utils.odds_display import (
    book_decimals_for_fighters,
    decimal_to_american,
    devig_implied_pair,
    ev_per_unit_stake_win_bet,
    format_american,
    format_ev,
    prob_to_fair_decimal,
)
from utils.mma_schedule_calendar import schedule_calendar_html
from utils.pipeline_runner import start_auto_pipeline_background, start_ufcstats_sync_background
from utils.prefight_history import PreFightBuilder, featurize_slate_with_builder, walk_history_until_date
from utils.ufc_historical import load_merged_bout_history
from utils.ufc_upcoming_odds_api import (
    fetch_mma_slate_for_event_date,
    fetch_upcoming_mma_schedule,
    get_odds_api_key,
)

_FUTURE_SLATE_SESSION_KEY = "_mma_future_slate"
_EXTENSION_PARQUET = _ROOT / "data" / "raw" / "ufcstats_extension.parquet"


def _extension_freshness_blurb() -> str | None:
    if not _EXTENSION_PARQUET.exists() or _EXTENSION_PARQUET.stat().st_size < 32:
        return None
    ts = datetime.fromtimestamp(_EXTENSION_PARQUET.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _maybe_autostart_ufcstats_sync() -> None:
    """Opt-in: set env MMA_UFCSTATS_AUTOSYNC_STALE_HOURS=168 (etc.) to sync in background when parquet is missing or older than this many hours (once per browser session)."""
    raw = (os.environ.get("MMA_UFCSTATS_AUTOSYNC_STALE_HOURS") or "").strip()
    if not raw:
        return
    try:
        stale_h = float(raw)
    except ValueError:
        return
    if stale_h <= 0:
        return
    if st.session_state.get("_mma_ufcstats_autosync_started"):
        return
    need = False
    if not _EXTENSION_PARQUET.exists() or _EXTENSION_PARQUET.stat().st_size < 32:
        need = True
    else:
        age_h = (time.time() - _EXTENSION_PARQUET.stat().st_mtime) / 3600.0
        need = age_h >= stale_h
    if not need:
        return
    st.session_state["_mma_ufcstats_autosync_started"] = True
    start_ufcstats_sync_background()
    st.info(
        f"Extension data is older than **{stale_h:g}** hours (or missing). Started **ufcstats.com** sync in background. "
        "When the log shows completion, click **Reload history cache**."
    )


def _unpack_future_bundle(bundle: object) -> tuple[date | None, pd.DataFrame | None, str | None]:
    if bundle is None:
        return None, None, None
    try:
        if len(bundle) == 2:
            b0, b1 = bundle  # type: ignore[misc]
            return b0, b1, "paste"
        b0, b1, b2 = bundle  # type: ignore[misc]
        return b0, b1, str(b2)
    except (TypeError, ValueError):
        return None, None, None


@st.cache_resource
def _cached_pipelines(_registry_mtime: float):
    return load_production_pipelines()


def _registry_mtime() -> float:
    p = _ROOT / "data" / "models" / "registry.json"
    try:
        return float(p.stat().st_mtime)
    except OSError:
        return 0.0


def get_pipelines():
    return _cached_pipelines(_registry_mtime())


def _bet_lean_label(pick: str, ev_u: float, dec_pick: float) -> str:
    if pick == "Toss-up (50/50)":
        return "—"
    if not np.isfinite(dec_pick) or dec_pick <= 1.0:
        return "Need book"
    if not np.isfinite(ev_u):
        return "—"
    if ev_u > 0.002:
        return f"+EV ({format_ev(ev_u)}/$1)"
    if ev_u < -0.002:
        return "No +EV"
    return "~Fair"


_MANUAL_SLATE_CSV_EXAMPLE = """fighter1,fighter2,weight_class,event_name,favourite,underdog,favourite_odds,underdog_odds
Fighter A,Fighter B,Lightweight,My UFC Card,Fighter A,Fighter B,1.72,2.15"""


def _parse_manual_upcoming_slate(
    text: str,
    pick: date,
    default_event_name: str,
) -> pd.DataFrame | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return None
    need = ("fighter1", "fighter2", "weight_class")
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"Missing columns {missing}. Required: fighter1, fighter2, weight_class. Optional: event_date, event_name, favourite, underdog, favourite_odds, underdog_odds.")
        return None
    df = df.copy()
    df["fighter1"] = df["fighter1"].astype(str).str.strip()
    df["fighter2"] = df["fighter2"].astype(str).str.strip()
    df["weight_class"] = df["weight_class"].astype(str).str.strip()
    df = df[(df["fighter1"] != "") & (df["fighter2"] != "")]
    if df.empty:
        st.error("No valid fighter rows after stripping blanks.")
        return None
    if "event_date" in df.columns:
        ed = pd.to_datetime(df["event_date"], errors="coerce")
        df["event_date"] = ed.dt.strftime("%Y-%m-%d")
        df.loc[ed.isna(), "event_date"] = pick.strftime("%Y-%m-%d")
    else:
        df["event_date"] = pick.strftime("%Y-%m-%d")
    if "event_name" not in df.columns:
        df["event_name"] = default_event_name
    df["event_name"] = df["event_name"].fillna(default_event_name).astype(str)
    for odds_col in ("favourite_odds", "underdog_odds"):
        if odds_col not in df.columns:
            df[odds_col] = np.nan
    for name_col in ("favourite", "underdog"):
        if name_col not in df.columns:
            df[name_col] = np.nan
    df["outcome"] = np.nan
    df["_idx"] = np.arange(len(df))
    return df


def _pick_and_confidence(fighter_a: str, fighter_b: str, p_a: float) -> tuple[str, float]:
    """Return (predicted_winner_label, probability that side wins)."""
    if p_a > 0.5 + 1e-9:
        return str(fighter_a), float(p_a)
    if p_a < 0.5 - 1e-9:
        return str(fighter_b), float(1.0 - p_a)
    return "Toss-up (50/50)", 0.5


def _rows_to_display_df(rows_out: list[dict]) -> pd.DataFrame:
    """Readable scan columns; ``fight_id`` kept for filtering (not shown in slim view)."""
    recs: list[dict] = []
    for r in rows_out:
        pa = float(r["P(A)"])
        pb = float(r["P(B)"])
        pick, conf = _pick_and_confidence(str(r["Fighter A"]), str(r["Fighter B"]), pa)
        evt = str(r.get("event_name") or "")
        if evt.startswith("UFC Fight Night: "):
            evt = evt.replace("UFC Fight Night: ", "", 1)
        if len(evt) > 40:
            evt = evt[:37] + "…"

        bda = float(r.get("book_dec_a", float("nan")))
        bdb = float(r.get("book_dec_b", float("nan")))
        p_undev_a, p_undev_b = devig_implied_pair(bda, bdb)
        fa, fb = str(r["Fighter A"]), str(r["Fighter B"])
        if pick == fa:
            dec_pick, fair_mkt = bda, p_undev_a
        elif pick == fb:
            dec_pick, fair_mkt = bdb, p_undev_b
        else:
            dec_pick, fair_mkt = float("nan"), float("nan")

        fair_dec_model = prob_to_fair_decimal(float(conf))
        ev_u = ev_per_unit_stake_win_bet(float(conf), dec_pick) if pick not in ("Toss-up (50/50)",) else float("nan")

        recs.append(
            {
                "Class": r.get("weight_class"),
                "Fighter A": r.get("Fighter A"),
                "Fighter B": r.get("Fighter B"),
                "Model pick": pick,
                "P model": float(conf),
                "Fair US (model)": format_american(decimal_to_american(fair_dec_model)),
                "Book dec (pick)": dec_pick if np.isfinite(dec_pick) else float("nan"),
                "Book US (pick)": format_american(decimal_to_american(dec_pick)),
                "Mkt fair % (pick)": float(fair_mkt) if np.isfinite(fair_mkt) else float("nan"),
                "EV fmt": format_ev(ev_u),
                "Bet lean": _bet_lean_label(pick, ev_u, dec_pick),
                "P(A)": pa,
                "P(B)": pb,
                "Src": r.get("source"),
                "Card": evt,
                "fight_id": r.get("fight_id"),
                "_ev_sort": ev_u if np.isfinite(ev_u) else -999.0,
            }
        )
    return pd.DataFrame(recs)


def _multiselect_label(r: dict) -> str:
    sid = str(r.get("fight_id"))[:8]
    wc = str(r.get("weight_class") or "")
    return f"[{sid}] {wc}: {r['Fighter A']} vs {r['Fighter B']}"


_SLATE_DISPLAY_COLS = [
    "Class",
    "Card",
    "Fighter A",
    "Fighter B",
    "Model pick",
    "P model",
    "Fair US (model)",
    "Book dec (pick)",
    "Book US (pick)",
    "Mkt fair % (pick)",
    "EV fmt",
    "Bet lean",
    "P(A)",
    "P(B)",
    "Src",
]


def _fight_table_column_config() -> dict[str, st.column_config.Column]:
    return {
        "Class": st.column_config.TextColumn("Wt class", width="small"),
        "Card": st.column_config.TextColumn("Event (short)", width="medium"),
        "Fighter A": st.column_config.TextColumn(width="medium"),
        "Fighter B": st.column_config.TextColumn(width="medium"),
        "Model pick": st.column_config.TextColumn(
            width="medium", help="Predicted winner = whichever of Fighter A or B has the higher win probability."
        ),
        "P model": st.column_config.ProgressColumn(
            "Model %",
            help="Model win probability for *Model pick*.",
            format="%.1f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "Fair US (model)": st.column_config.TextColumn(
            "Fair US",
            width="small",
            help="American odds equivalent to *Model %* (no vig; fair line implied by the model).",
        ),
        "Book dec (pick)": st.column_config.NumberColumn(
            "Bk dec",
            help="Offered decimal odds on *Model pick* from jansen88 favourite/underdog columns.",
            format="%.2f",
        ),
        "Book US (pick)": st.column_config.TextColumn(
            "Bk US",
            width="small",
            help="American equivalent of *Bk dec* on *Model pick*.",
        ),
        "Mkt fair % (pick)": st.column_config.ProgressColumn(
            "Mkt %",
            help="De-vig two-way implied win probability on *Model pick* from the book decimals.",
            format="%.1f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "EV fmt": st.column_config.TextColumn(
            "EV $1",
            width="small",
            help="Expected profit per \\$1 risked on *Model pick* at offered *Bk dec*: p_model×d−1.",
        ),
        "Bet lean": st.column_config.TextColumn(
            "Bet?",
            width="small",
            help="+EV if model edge vs offered line (rough; shop lines).",
        ),
        "P(A)": st.column_config.NumberColumn("P(A)", format="%.3f"),
        "P(B)": st.column_config.NumberColumn("P(B)", format="%.3f"),
        "Src": st.column_config.TextColumn(width="small"),
    }


def _ensure_wc_map(hist: pd.DataFrame) -> dict[str, int]:
    m = load_weight_class_map()
    if m:
        return m
    m = fit_weight_class_map(hist["weight_class"].astype(str))
    save_weight_class_map(m)
    return m


@st.cache_data(ttl=3600, show_spinner="Loading historical fights…")
def _cached_hist():
    return load_merged_bout_history()


@st.cache_data(ttl=1800, show_spinner="Loading upcoming MMA (Odds API)…")
def _cached_upcoming_mma_schedule(api_key: str) -> pd.DataFrame:
    return fetch_upcoming_mma_schedule(api_key)


def _mma_upcoming_calendar_section() -> None:
    """Odds API: multi-month grids + table of all upcoming fights (US/Eastern card dates)."""
    st.subheader("Upcoming fight calendar")
    st.caption(
        "These bouts are **not yet fought** — there is no official **winner**. When a book posts **H2H** prices, "
        "the table shows **favourite / underdog** (by implied line) and **decimal + American** odds."
    )
    api = get_odds_api_key()
    if not api:
        st.caption(
            "Set **THE_ODDS_API_KEY** (environment or Streamlit Secrets) to load all upcoming MMA fights "
            "from [The Odds API](https://the-odds-api.com) into the calendar and table below."
        )
        with st.expander("Troubleshooting: key saved in Streamlit but still not detected?", expanded=False):
            st.markdown(
                """
1. **Community Cloud:** open your app → **⋮** → **Settings** → **Secrets**. Save this **exact** TOML (one line, your real key):

```toml
THE_ODDS_API_KEY = "your-key-here"
```

2. Click **Save**, then **Reboot app** (or **Manage app → Reboot**). Secrets load only when the Python process starts — a browser refresh is not enough.

3. Key name must match **`THE_ODDS_API_KEY`** (or `ODDS_API_KEY`) at the **top level**, **or** under a **`[api]`** section in Secrets, for example:

```toml
[api]
THE_ODDS_API_KEY = "your-key-here"
```

4. Alternatively, paste the key in **The Odds API key** under *Upcoming fights* (never commit it to git).
"""
            )
            try:
                names = sorted(getattr(st.secrets, "keys", lambda: [])())
                st.caption("Top-level secret **names** visible to this app (values never shown):")
                st.code("\n".join(names) if names else "(none — Secrets file empty or unreadable)")
            except Exception as ex:
                st.caption(f"Could not list secret names: `{ex!s}`")
        return
    try:
        dfu = _cached_upcoming_mma_schedule(api)
    except Exception as e:
        st.warning(f"Could not load The Odds API schedule: {e}")
        return
    if dfu.empty:
        st.info("The Odds API returned no upcoming MMA fights (or every listed bout has already started).")
        return
    html = schedule_calendar_html(dfu, max_months=4)
    components.html(html, height=460, scrolling=True)

    def _jump_to_card_date() -> None:
        sel = st.session_state.get("_mma_cal_jump_date")
        if sel and sel != "—":
            st.session_state.mma_event_date = date.fromisoformat(sel)

    card_dates = sorted({pd.Timestamp(x).date() for x in dfu["card_date"].tolist()})
    opts = ["—"] + [d.isoformat() for d in card_dates]
    cj, cr = st.columns([2, 1])
    with cj:
        st.selectbox(
            "Jump **Event date** to a card",
            opts,
            key="_mma_cal_jump_date",
            on_change=_jump_to_card_date,
            help="Updates the Event date control below so you can model that card.",
        )
    with cr:
        if st.button("Refresh Odds API", help="Clear the 30-minute schedule cache and refetch."):
            _cached_upcoming_mma_schedule.clear()
            st.rerun()

    show = dfu.drop(columns=["commence_utc", "_idx"], errors="ignore").copy()
    _order = [
        "card_date",
        "commence_et_str",
        "fighter1",
        "fighter2",
        "bookmaker",
        "favourite",
        "underdog",
        "favourite_odds_dec",
        "underdog_odds_dec",
        "favourite_american",
        "underdog_american",
        "has_h2h_odds",
        "event_title",
    ]
    show = show[[c for c in _order if c in show.columns]]
    show = show.rename(
        columns={
            "commence_et_str": "Start (ET)",
            "bookmaker": "Book",
            "favourite_odds_dec": "Fav dec",
            "underdog_odds_dec": "Dog dec",
            "favourite_american": "Fav US",
            "underdog_american": "Dog US",
            "has_h2h_odds": "Has H2H",
        }
    )
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Fav dec": st.column_config.NumberColumn("Fav dec", format="%.2f", help="Favourite decimal odds (H2H)."),
            "Dog dec": st.column_config.NumberColumn("Dog dec", format="%.2f", help="Underdog decimal odds (H2H)."),
            "Fav US": st.column_config.NumberColumn("Fav US", format="%.0f", help="Favourite American moneyline."),
            "Dog US": st.column_config.NumberColumn("Dog US", format="%.0f", help="Underdog American moneyline."),
            "Has H2H": st.column_config.CheckboxColumn("Has H2H", help="Book offered a parseable two-way H2H market."),
        },
    )


def main() -> None:
    if os.environ.get("SPORTS_HUB") != "1":
        st.set_page_config(page_title="MMA fight model", layout="wide")
    _maybe_autostart_ufcstats_sync()
    st.title("MMA — Fighter A win probability")
    st.caption(
        "**Fighter A / B** are assigned per bout with the same deterministic hash as training (not winner order). "
        "**Labels** use official ``outcome``. **Rates** (sig strikes, TD/15, KD/min, subs/fight) use **prior** "
        "cage time from merged ``events_raw`` bout totals—not the wide CSV career columns."
    )

    today = date.today()
    if "mma_event_date" not in st.session_state:
        st.session_state.mma_event_date = today
    _mma_upcoming_calendar_section()

    hist = _cached_hist()
    hist = hist.copy()
    hist["event_date"] = pd.to_datetime(hist["event_date"], errors="coerce")
    hist = hist.dropna(subset=["event_date"])
    hist["_idx"] = np.arange(len(hist))
    hist_sorted = hist.sort_values(["event_date", "event_name", "_idx"], kind="mergesort")
    wc_map = _ensure_wc_map(hist)

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        if st.button("Refresh UFC CSV cache", help="Re-download complete_ufc_data + events_raw"):
            st.cache_data.clear()
            load_merged_bout_history(refresh=True, refresh_events=True)
            st.rerun()
    with c1:
        if st.button("Run merge → train → eval (background)"):
            start_auto_pipeline_background()
            st.info("Started auto pipeline in background. Check data/processed/auto_pipeline.log.")
    with c2:
        if st.button(
            "Sync ufcstats.com (background)",
            help="Free scrape: bouts after ~Sep 2023 into data/raw/ufcstats_extension.parquet. Can take 30–90+ min first run. Log: data/processed/ufcstats_sync.log. Then click “Reload history cache”.",
        ):
            start_ufcstats_sync_background()
            st.info("Started UFCStats sync in background. When finished, use **Reload history cache**.")
    with c3:
        if st.button("Reload history cache", help="Clear Streamlit cache so ufcstats extension is picked up."):
            st.cache_data.clear()
            st.rerun()
    ext_m = _extension_freshness_blurb()
    if ext_m:
        st.caption(
            f"Post-jansen88 extension last written: **{ext_m}**. "
            "Merged history max date is below. Auto-sync on open: set env `MMA_UFCSTATS_AUTOSYNC_STALE_HOURS` "
            "(hours idle before a background sync triggers once per session)."
        )
    else:
        st.caption(
            "No ufcstats extension file yet — click **Sync ufcstats.com** (or set `MMA_UFCSTATS_AUTOSYNC_STALE_HOURS`)."
        )
    with st.container():
        rep = load_eval_report()
        if rep:
            st.caption(f"Last eval rows: {rep.get('n_rows', '—')}")

    d_min = hist["event_date"].min().date()
    d_max = hist["event_date"].max().date()
    d_hi = max(d_max, today + timedelta(days=730))
    st.session_state.mma_event_date = min(max(st.session_state.mma_event_date, d_min), d_hi)
    st.date_input(
        "Event date",
        min_value=d_min,
        max_value=d_hi,
        key="mma_event_date",
        help="Pick a card date. Future UFC uses **Fetch MMA matchups** or the calendar above (Odds API). Dates use **US/Eastern** for API matchups.",
    )
    pick = st.session_state.mma_event_date

    if today > d_max:
        st.info(
            f"Historical training CSV ends **{d_max}**. Real upcoming cards: use **Fetch MMA matchups** with a free key from "
            f"[the-odds-api.com](https://the-odds-api.com) (env `THE_ODDS_API_KEY`), or paste CSV. "
            f"API dates match **US/Eastern**."
        )

    builder = PreFightBuilder()
    walk_history_until_date(hist_sorted, pd.Timestamp(pick), builder)
    slate_hist = hist_sorted[hist_sorted["event_date"].dt.date == pick].sort_values(["event_name", "_idx"], kind="mergesort")

    slate = slate_hist
    future_source: str | None = None

    if slate_hist.empty:
        st.warning("No fights for this date in the cached historical CSV.")
        b_pick, b_df, b_src = _unpack_future_bundle(st.session_state.get(_FUTURE_SLATE_SESSION_KEY))
        if b_pick == pick and b_df is not None and not b_df.empty and b_src is not None:
            slate = b_df.sort_values(["event_name", "_idx"], kind="mergesort")
            future_source = b_src
            st.success(f"Using **{len(slate)}** fight(s) loaded via **{future_source}** for **{pick}**.")

        api_env = get_odds_api_key()
        st.markdown("### Upcoming fights")
        ck = st.columns([2, 1])
        with ck[0]:
            api_key_in = st.text_input(
                "The Odds API key",
                type="password",
                placeholder="Optional if THE_ODDS_API_KEY is set in Secrets / environment",
                help="Sign up at the-odds-api.com (free quota). Never commit keys. "
                "If you use Streamlit Secrets, reboot the app after saving — do not rely on this field.",
                key="odds_api_key_field",
            )
        api_key_use = (api_env or (api_key_in or "").strip()).strip()
        with ck[1]:
            st.write("")
            st.write("")
            fetch_clicked = st.button(
                "Fetch MMA matchups",
                type="primary",
                disabled=not api_key_use,
                help="UFC/MMA moneylines for this Event date (US/Eastern card day). Uses API quota.",
            )
        if fetch_clicked and api_key_use:
            try:
                df_api = fetch_mma_slate_for_event_date(api_key_use, pick)
                if df_api.empty:
                    st.warning(
                        "No MMA events from The Odds API on this **US/Eastern** date. "
                        "Try an adjacent day or confirm a card is scheduled."
                    )
                else:
                    st.session_state[_FUTURE_SLATE_SESSION_KEY] = (pick, df_api, "odds_api")
                    st.rerun()
            except requests.HTTPError as e:
                st.error(f"Odds API HTTP error: {e}")
            except Exception as e:
                st.error(f"Odds API error: {e!s}")

        with st.expander("Or — paste custom card (CSV)", expanded=future_source is None):
            st.markdown(
                "One row per fight. **Required:** `fighter1`, `fighter2`, `weight_class`. "
                "**Optional:** `event_date`, `event_name`, `favourite`, `underdog`, `favourite_odds`, `underdog_odds` (decimal). "
                "Match **favourite/underdog** names to fighters for prices and **Bet?** / EV."
            )
            st.code(_MANUAL_SLATE_CSV_EXAMPLE, language="csv")
            evt_title = st.text_input("Default event title (if row has no event_name)", value="Upcoming / manual card")
            pasted = st.text_area("CSV text", height=180, key="manual_slate_csv")
            if st.button("Load pasted card", type="primary"):
                manual_df = _parse_manual_upcoming_slate(pasted, pick, evt_title)
                if manual_df is not None:
                    st.session_state[_FUTURE_SLATE_SESSION_KEY] = (pick, manual_df, "paste")
                    st.rerun()
            bundle_prev = st.session_state.get(_FUTURE_SLATE_SESSION_KEY)
            b_pick_prev, _, _ = _unpack_future_bundle(bundle_prev)
            if b_pick_prev is not None and b_pick_prev != pick:
                st.caption(f"Saved slate is for **{b_pick_prev}** — fetch or paste again for **{pick}**.")
            if future_source and st.button("Clear loaded slate"):
                st.session_state.pop(_FUTURE_SLATE_SESSION_KEY, None)
                st.rerun()

        if slate.empty:
            return
    else:
        st.session_state.pop(_FUTURE_SLATE_SESSION_KEY, None)

    reg, win_b = get_pipelines()
    blend_w = float((reg.get("production") or {}).get("win_blend_weight", 0.0) or 0.0)

    ingest_after_each = future_source is None
    feat_dicts = featurize_slate_with_builder(builder, slate, wc_map, ingest_after_each=ingest_after_each)
    by_id = {str(d["fight_id"]): d for d in feat_dicts}

    slate_by_fid = {fight_id_for_row(r): r for _, r in slate.iterrows()}

    rows_out: list[dict] = []
    for d in feat_dicts:
        raw = slate_by_fid[str(d["fight_id"])]
        s = pd.Series(d)
        ph = heuristic_fighter_a_win_prob(s)
        p_ml, ver = predict_fighter_a_win_ml(s, win_b, reg)
        if p_ml is None:
            p_final = ph
            src = "heuristic"
        else:
            p_final = blended_prob(float(p_ml), ph, blend_w)
            src = "ml+heuristic" if blend_w > 0 else "ml"
        ba, bb = book_decimals_for_fighters(raw, str(d["fighter_a"]), str(d["fighter_b"]))
        rows_out.append(
            {
                "fight_id": str(d["fight_id"]),
                "event_date": d.get("event_date"),
                "event_name": d.get("event_name"),
                "weight_class": d.get("weight_class"),
                "Fighter A": d.get("fighter_a"),
                "Fighter B": d.get("fighter_b"),
                "P(A)": round(float(p_final), 4),
                "P(B)": round(float(1.0 - p_final), 4),
                "source": src,
                "model": ver if p_ml is not None else "heuristic_only",
                "book_dec_a": ba,
                "book_dec_b": bb,
            }
        )

    disp_view = _rows_to_display_df(rows_out)
    if st.checkbox("Sort slate by EV ($1), highest first", value=False):
        disp_view = disp_view.sort_values("_ev_sort", ascending=False, kind="mergesort")
    disp_view = disp_view.drop(columns=["_ev_sort"], errors="ignore")

    models_line = " · ".join(sorted({str(r.get("model") or "?") for r in rows_out}))
    provenance_lbl = (
        ""
        if future_source is None
        else (" · Odds API" if future_source == "odds_api" else " · pasted CSV")
    )
    st.subheader(f"Slate — {pick} ({len(disp_view)} fights){provenance_lbl}")
    st.caption(
        "**Who wins:** *Model pick* = higher of P(A)/P(B). **Bet?** flags rough **+EV** on *Model pick* at **Bk dec** (not financial advice). "
        "**Fair US** = model fair line; **Bk** = books (CSV or Odds API). **EV $1** = p×decimal−1. Models: "
        + models_line
    )
    st.dataframe(
        disp_view[_SLATE_DISPLAY_COLS],
        use_container_width=True,
        hide_index=True,
        column_config=_fight_table_column_config(),
    )

    options = [_multiselect_label(r) for r in rows_out]
    choice_map = {options[i]: rows_out[i] for i in range(len(options))}
    picked = st.multiselect("Pick fights to focus / log", options=options, default=[])
    if picked:
        sub = disp_view[disp_view["fight_id"].isin({choice_map[p]["fight_id"] for p in picked})]
        st.subheader("Selected")
        st.dataframe(
            sub[_SLATE_DISPLAY_COLS],
            use_container_width=True,
            hide_index=True,
            column_config=_fight_table_column_config(),
        )

        if st.button("Append predictions for selected fights (CSV log)", type="primary"):
            log_rows = []
            ts = utc_now_iso()
            for p in picked:
                base = choice_map[p]
                d = by_id[base["fight_id"]]
                s = pd.Series(d)
                ph = heuristic_fighter_a_win_prob(s)
                p_ml, ver = predict_fighter_a_win_ml(s, win_b, reg)
                p_final = float(ph if p_ml is None else blended_prob(float(p_ml), ph, blend_w))
                rec = {
                    "fight_id": base["fight_id"],
                    "event_date": str(base["event_date"]),
                    "event_name": d.get("event_name"),
                    "weight_class": d.get("weight_class"),
                    "fighter_a": d.get("fighter_a"),
                    "fighter_b": d.get("fighter_b"),
                    "orig_fighter1": d.get("orig_fighter1"),
                    "orig_fighter2": d.get("orig_fighter2"),
                    "logged_at": ts,
                    "pred_fighter_a_win_prob": p_final,
                    "pred_fighter_a_win_prob_heur": ph,
                    "pred_fighter_b_win_prob": float(1.0 - p_final),
                    "ml_model_version": ver,
                    "ml_raw_available": int(p_ml is not None),
                }
                for c in GAME_FEATURE_NAMES:
                    rec[c] = float(s.get(c, np.nan)) if c in s.index else float("nan")
                log_rows.append(rec)
            append_predictions_df(pd.DataFrame(log_rows))
            st.success(f"Logged {len(log_rows)} prediction rows.")

    st.divider()
    st.subheader("Outcomes + retrain")
    st.write(
        "For **past** cards, run ``python -m ml.backfill_outcomes_from_history_mma`` then ``python -m ml.auto_pipeline``."
    )
    st.subheader("Cold-start training (historical)")
    st.code(
        "python -m ml.seed_training_from_history_mma\npython -m ml.train_win_model_mma",
        language="bash",
    )


if __name__ == "__main__":
    main()
