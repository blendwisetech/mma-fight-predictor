"""HTML month grids for upcoming MMA cards (Odds API schedule); optional pickable links."""

from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, datetime
from html import escape

import pandas as pd

from utils.odds_display import format_american

# URL query key: clicking a calendar cell navigates with this param; we consume it once
# and sync session_state (Streamlit buttons in nested columns were unusably narrow).
MMA_CAL_QUERY_KEY = "mma_cal_pick"


def _preserved_query_excluding_mma() -> str:
    """Rebuild query string without ``mma_cal_pick`` so calendar links keep other params."""
    import streamlit as st
    from urllib.parse import quote

    parts: list[str] = []
    qp = st.query_params
    for k in qp.keys():
        if k == MMA_CAL_QUERY_KEY:
            continue
        v = qp.get(k)
        if isinstance(v, (list, tuple)):
            for item in v:
                parts.append(f"{quote(str(k), safe='')}={quote(str(item), safe='')}")
        else:
            parts.append(f"{quote(str(k), safe='')}={quote(str(v), safe='')}")
    return "&".join(parts)


def aggregate_counts_and_tooltips(df: pd.DataFrame) -> tuple[dict[date, int], dict[date, str]]:
    """Per **US/Eastern card date**: fight count and a short hover summary."""
    if df.empty or "card_date" not in df.columns:
        return {}, {}
    counts: dict[date, int] = defaultdict(int)
    lines: dict[date, list[str]] = defaultdict(list)
    for _, r in df.iterrows():
        cd = r["card_date"]
        if hasattr(cd, "date"):
            cd = cd.date()
        if not isinstance(cd, date):
            continue
        counts[cd] += 1
        if len(lines[cd]) < 5:
            f1 = str(r.get("fighter1") or "")
            f2 = str(r.get("fighter2") or "")
            fav = str(r.get("favourite") or "")
            und = str(r.get("underdog") or "")
            if bool(r.get("has_h2h_odds")) and fav and und and pd.notna(r.get("favourite_american")):
                fa = float(r["favourite_american"])
                ua = float(r["underdog_american"])
                fav_s = format_american(fa)
                und_s = format_american(ua)
                lines[cd].append(f"{f1} vs {f2} ({fav} {fav_s} / {und} {und_s})")
            else:
                lines[cd].append(f"{f1} vs {f2}")
    tips = {d: "; ".join(lines[d])[:400] for d in lines}
    return dict(counts), tips


def _months_to_display(counts: dict[date, int], *, max_months: int = 4) -> list[tuple[int, int]]:
    if not counts:
        td = date.today()
        out: list[tuple[int, int]] = []
        y, m = td.year, td.month
        for _ in range(max_months):
            out.append((y, m))
            m += 1
            if m > 12:
                m, y = 1, y + 1
        return out
    lo, hi = min(counts.keys()), max(counts.keys())
    months: list[tuple[int, int]] = []
    y, m = lo.year, lo.month
    while (y, m) <= (hi.year, hi.month) and len(months) < max_months:
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def display_months_for_counts(counts: dict[date, int], *, max_months: int = 4) -> list[tuple[int, int]]:
    """Public alias: which year/months to show for a fight-count map."""
    return _months_to_display(counts, max_months=max_months)


def _month_grid_html(
    y: int,
    m: int,
    counts: dict[date, int],
    tips: dict[date, str],
    *,
    pickable: bool = False,
    selected: date | None = None,
    query_key: str = MMA_CAL_QUERY_KEY,
    query_prefix: str = "",
) -> str:
    cal = calendar.Calendar(firstweekday=calendar.SUNDAY)
    weeks = cal.monthdatescalendar(y, m)
    month_name = date(y, m, 1).strftime("%B %Y")
    th = "".join(f"<th>{escape(w)}</th>" for w in ("Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"))
    body_rows: list[str] = []
    for week in weeks:
        tds: list[str] = []
        for d in week:
            if d.month != m:
                tds.append('<td class="pad"></td>')
                continue
            n = int(counts.get(d, 0))
            tip = tips.get(d, "")
            tip_attr = f' title="{escape(tip)}"' if tip else ""
            if n:
                inner = f'<span class="cnt">{n}</span><span class="dn">{d.day}</span>'
                if pickable:
                    qk_e = escape(query_key)
                    iso_e = escape(d.isoformat())
                    p = query_prefix.strip()
                    if p:
                        href = f"?{p}&{qk_e}={iso_e}"
                    else:
                        href = f"?{qk_e}={iso_e}"
                    inner = f'<a class="mma-cal-pick" href="{href}">{inner}</a>'
                cls = "day has"
                if pickable and selected is not None and d == selected:
                    cls += " sel"
            else:
                inner = f'<span class="dn">{d.day}</span>'
                cls = "day"
            tds.append(f'<td class="{cls}"{tip_attr}>{inner}</td>')
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    return (
        f'<div class="mma-cal-mon"><div class="mma-cal-hdr">{escape(month_name)}</div>'
        f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def schedule_calendar_html(
    df: pd.DataFrame,
    *,
    max_months: int = 4,
    pickable: bool = False,
    selected: date | None = None,
    query_prefix: str = "",
) -> str:
    """Full HTML fragment: CSS + up to ``max_months`` month grids (Sunday-first, US layout).

    When ``pickable`` is True, days with bouts are ``<a href="?mma_cal_pick=YYYY-MM-DD">`` links;
    use :func:`consume_mma_calendar_url_pick` on each run to apply the choice and strip the param.
    """
    counts, tips = aggregate_counts_and_tooltips(df)
    months = display_months_for_counts(counts, max_months=max_months)
    cols = "".join(
        f'<div class="mma-cal-col">{_month_grid_html(y, m, counts, tips, pickable=pickable, selected=selected, query_prefix=query_prefix)}</div>'
        for y, m in months
    )
    css_pick = """
    .mma-cal-mon td.day.has { cursor: pointer; }
    .mma-cal-mon td.day.has a.mma-cal-pick { display: flex; flex-direction: column; align-items: center; justify-content: center;
      text-decoration: none; color: inherit; width: 100%; height: 100%; min-height: 38px; box-sizing: border-box; padding: 2px 0; }
    .mma-cal-mon td.day.has.sel { background: #dc2626 !important; color: #fff !important; border: 1px solid #991b1b !important; }
    .mma-cal-mon td.day.has.sel .cnt, .mma-cal-mon td.day.has.sel .dn { color: #fff !important; }
    """
    css = """
    .mma-cal-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 13px; color: #1a1a2e; }
    .mma-cal-row { display: flex; flex-wrap: nowrap; gap: 12px; justify-content: flex-start; overflow-x: auto;
      padding-bottom: 8px; -webkit-overflow-scrolling: touch; }
    .mma-cal-col { flex: 0 0 auto; width: 232px; min-width: 232px; background: #f6f8fc; border-radius: 8px; padding: 8px 10px 12px; border: 1px solid #d8dee9; }
    .mma-cal-hdr { font-weight: 600; margin-bottom: 6px; color: #111827; }
    .mma-cal-mon table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    .mma-cal-mon th { font-size: 11px; font-weight: 500; color: #6b7280; padding: 4px 2px; }
    .mma-cal-mon td { height: 40px; text-align: center; vertical-align: middle; border-radius: 4px; }
    .mma-cal-mon td.pad { background: transparent; }
    .mma-cal-mon td.day { color: #4b5563; }
    .mma-cal-mon td.day.has { background: #dbeafe; color: #1e3a8a; font-weight: 600; border: 1px solid #93c5fd; }
    .mma-cal-mon td.day .cnt { display: block; font-size: 10px; line-height: 1.1; color: #2563eb; }
    .mma-cal-mon td.day .dn { font-size: 13px; }
    .mma-cal-leg { margin-top: 10px; font-size: 12px; color: #6b7280; }
    """
    if pickable:
        css += css_pick
    leg = (
        "Cell <strong>number</strong> = fights on that <strong>US/Eastern</strong> card date. "
        "Hover a shaded day for matchups and <strong>book favourite / underdog with American odds</strong> when available. "
        "There is no official <strong>winner</strong> until after the bout. Data from The Odds API."
    )
    if pickable:
        leg += " <strong>Click</strong> a shaded day to set <strong>Event date</strong> and load that card."
    return (
        f"<style>{css}</style>"
        f'<div class="mma-cal-wrap"><div class="mma-cal-row">{cols}</div>'
        f'<p class="mma-cal-leg">{leg}</p></div>'
    )


def consume_mma_calendar_url_pick(schedule_df: pd.DataFrame, *, future_session_key: str) -> None:
    """Apply ``?mma_cal_pick=YYYY-MM-DD`` once: sync session state, drop the query key, rerun."""
    import streamlit as st

    from utils.ufc_upcoming_odds_api import schedule_df_to_slate_for_day

    key = MMA_CAL_QUERY_KEY
    if key not in st.query_params:
        return
    raw = st.query_params[key]
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if not raw:
        try:
            del st.query_params[key]
        except Exception:
            pass
        st.rerun()
        return
    try:
        d = date.fromisoformat(str(raw))
    except ValueError:
        try:
            del st.query_params[key]
        except Exception:
            pass
        st.rerun()
        return
    valid: set[date] = set()
    for x in schedule_df["card_date"].tolist():
        if hasattr(x, "date"):
            x = x.date()
        if isinstance(x, date):
            valid.add(x)
    if d not in valid:
        try:
            del st.query_params[key]
        except Exception:
            pass
        st.rerun()
        return
    st.session_state.mma_event_date = d
    slate = schedule_df_to_slate_for_day(schedule_df, d)
    if slate is not None and not slate.empty:
        st.session_state[future_session_key] = (d, slate, "odds_api")
    else:
        st.session_state.pop(future_session_key, None)
    try:
        del st.query_params[key]
    except Exception:
        pass
    st.rerun()


def render_pickable_mma_calendar(
    schedule_df: pd.DataFrame,
    counts: dict[date, int],
    months: list[tuple[int, int]],
    *,
    future_session_key: str,
    max_months: int = 4,
) -> None:
    """
    Month grids in **HTML table** layout (original style): shaded cells, bout count above day number.
    Fight days link with ``?mma_cal_pick=…``; :func:`consume_mma_calendar_url_pick` applies the pick.
    """
    import streamlit as st

    _: object = (counts, months)  # API compat; grid window follows ``schedule_df`` like ``schedule_calendar_html``.

    consume_mma_calendar_url_pick(schedule_df, future_session_key=future_session_key)

    preserved = _preserved_query_excluding_mma()

    raw_sel = st.session_state.get("mma_event_date")
    if isinstance(raw_sel, datetime):
        sel_d: date | None = raw_sel.date()
    elif type(raw_sel) is date:
        sel_d = raw_sel
    else:
        sel_d = None

    html = schedule_calendar_html(
        schedule_df,
        max_months=max_months,
        pickable=True,
        selected=sel_d,
        query_prefix=preserved,
    )
    st.markdown(html, unsafe_allow_html=True)
