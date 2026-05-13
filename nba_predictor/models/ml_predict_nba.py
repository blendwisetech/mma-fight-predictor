"""Load optional sklearn win model for NBA features."""

from __future__ import annotations

from typing import Any

import joblib

from utils.data_io import ROOT, load_registry


def _bundle(rel_path: str | None) -> dict[str, Any] | None:
    if not rel_path:
        return None
    p = ROOT / rel_path
    if not p.exists():
        return None
    b = joblib.load(p)
    return b if isinstance(b, dict) else {"pipeline": b, "version": "joblib_v1"}


def load_production_win_bundle() -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Registry plus **NBA** production sklearn bundle (backward compatible)."""
    reg = load_registry()
    prod = reg.get("production", {})
    return reg, _bundle(prod.get("win_model_path"))


def load_win_bundle(reg: dict[str, Any], league: str) -> dict[str, Any] | None:
    """
    Sklearn bundle for ``league`` (``\"nba\"`` or ``\"wnba\"``).

    Paths: ``production.win_model_path`` / ``production.win_model_path_wnba``.
    """
    prod = reg.get("production") or {}
    lg = league.strip().lower()
    key = "win_model_path_wnba" if lg == "wnba" else "win_model_path"
    return _bundle(prod.get(key))
