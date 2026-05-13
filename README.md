# MMA fight predictor (Streamlit)

UFC / MMA **fighter win probabilities** from merged historical data (jansen88 CSV + optional ufcstats.com extension) and a trained scikit-learn model.

## Run locally

```bash
cd mma_predictor
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m utils.ufc_historical download
streamlit run app/main.py
```

Optional: set `THE_ODDS_API_KEY` for **Fetch MMA matchups** (The Odds API). For local dev you can copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill the key.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (this folder is the repo root).
2. In [Streamlit Cloud](https://streamlit.io/cloud), **New app** → pick the repo.
3. **Main file path:** `app/main.py`
4. **Python version:** 3.11+ (3.13 if available is fine).
5. **Secrets:** App settings → Secrets, add:

   ```toml
   THE_ODDS_API_KEY = "your-key"
   ```

   The app reads this via `st.secrets` (see `utils/ufc_upcoming_odds_api.py`).

6. **Cold start:** On first load the app uses bundled `data/raw` and `data/models` when present; otherwise run **Refresh UFC CSV cache** in the UI or `python -m utils.ufc_historical download` in a one-off job.

**Note:** Background **ufcstats.com** sync can run many minutes and uses ephemeral disk on Cloud; for production-scale refresh, prefer running sync on your machine or in GitHub Actions, then committing updated `ufcstats_extension.parquet`, or using external object storage.

## Training (optional)

```bash
python -m ml.seed_training_from_history_mma
python -m ml.train_win_model_mma
```

## License

Use at your own risk; not financial advice. Respect [ufcstats.com](http://www.ufcstats.com) and data providers’ terms.
