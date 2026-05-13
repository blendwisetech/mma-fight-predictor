"""
Multi-sport hub: MMA (UFC), NBA / WNBA, and MLB predictors in **one** Streamlit app.

Each subproject uses top-level package names ``utils`` / ``ml`` / ``models``, so we
purge modules loaded from this repo and prepend only the active project on ``sys.path``
before loading that app's ``main``.

Deploy on Streamlit Cloud with **Main file path:** ``Home.py``
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent
NBA_ROOT = REPO / "nba_predictor"
MLB_ROOT = REPO / "baseball_predictor"
MMA_ROOT = REPO

SPORT_ORDER: tuple[tuple[str, str, Path, Path], ...] = (
    ("MMA (UFC)", "mma", MMA_ROOT, REPO / "app" / "main.py"),
    ("NBA / WNBA", "nba", NBA_ROOT, NBA_ROOT / "app" / "main.py"),
    ("MLB", "mlb", MLB_ROOT, MLB_ROOT / "app" / "main.py"),
)


def _purge_predictor_modules(repo: Path) -> None:
    """Drop cached imports from any sport folder so the next sport gets a clean ``utils``."""
    root_s = str(repo.resolve())
    to_del: list[str] = []
    for name, mod in list(sys.modules.items()):
        if name == "__main__" or mod is None:
            continue
        fp = getattr(mod, "__file__", None)
        if not fp:
            continue
        try:
            rfp = str(Path(fp).resolve())
        except Exception:
            continue
        if "site-packages" in rfp.replace("\\", "/").lower():
            continue
        if rfp.startswith(root_s):
            to_del.append(name)
    for name in to_del:
        del sys.modules[name]


def _prepend_sys_path(root: Path) -> None:
    s = str(root.resolve())
    try:
        sys.path.remove(s)
    except ValueError:
        pass
    sys.path.insert(0, s)


def _load_and_run(module_id: str, main_py: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_id, main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {main_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_id] = mod
    spec.loader.exec_module(mod)
    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        raise RuntimeError(f"{main_py} has no main()")
    main_fn()


st.set_page_config(
    page_title="Sports predictors",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("League")
labels = [t[0] for t in SPORT_ORDER]
choice = st.sidebar.radio("Open", labels, index=0, label_visibility="collapsed")

sel = next(t for t in SPORT_ORDER if t[0] == choice)
_, key, root, main_py = sel

if not main_py.is_file():
    st.error(f"Missing entrypoint: **{main_py}**. Clone the full repo (includes `nba_predictor/` and `baseball_predictor/`).")
    st.stop()

os.environ["SPORTS_HUB"] = "1"
_purge_predictor_modules(REPO)
_prepend_sys_path(root)
_load_and_run(f"_hub_sport_{key}", main_py)
