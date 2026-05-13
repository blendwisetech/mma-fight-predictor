# Sports predictors (Streamlit hub)

One Streamlit app with **MMA (UFC)**, **NBA / WNBA**, and **MLB** predictors. Use the sidebar **League** control to switch; each league keeps its own `data/` tree under this repo.

## Run locally (hub)

```bash
cd mma_predictor   # repo root (historical folder name)
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m utils.ufc_historical download   # MMA raw CSVs if missing
streamlit run Home.py
```

MMA-only (no hub sidebar):

```bash
streamlit run app/main.py
```

NBA/WNBA or MLB alone: `cd nba_predictor` or `cd baseball_predictor` then `streamlit run app/main.py`.

## Deploy on Streamlit Community Cloud

1. Repo: **blendwisetech/mma-fight-predictor** (or your fork).
2. **Main file path:** `Home.py` (not `app/main.py`) so all three sports load.
3. **Secrets** (optional): `THE_ODDS_API_KEY` for MMA **Fetch MMA matchups** — same as before (`utils/ufc_upcoming_odds_api.py`).

Cold start / data: each subproject can download or refresh from its UI; MMA bundles `data/raw` + models when committed.

## Notes

- **Switching leagues** reloads Python modules from that folder only (top-level `utils` / `ml` / `models` names would clash without this).
- **Ufcstats** background sync on Cloud is still best-effort on ephemeral disk; see MMA docs in `app/main.py` docstring.
- Not financial advice; respect data providers’ terms.
