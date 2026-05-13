"""Fire-and-forget wrappers for long-running jobs from Streamlit."""

from __future__ import annotations

import threading
import traceback


def start_auto_pipeline_background() -> None:
    def _work() -> None:
        try:
            from ml.auto_pipeline import append_pipeline_log, run_pipeline

            append_pipeline_log(run_pipeline())
        except Exception:
            from pathlib import Path

            p = Path(__file__).resolve().parents[1] / "data" / "processed" / "auto_pipeline.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write("\n--- THREAD EXCEPTION ---\n")
                f.write(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()


def start_ufcstats_sync_background(*, since: str = "2023-09-17") -> None:
    """Incremental scrape from ufcstats.com (can run a long time)."""

    def _work() -> None:
        from datetime import date as date_cls
        from pathlib import Path

        log_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "ufcstats_sync.log"

        def _log(msg: str) -> None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        try:
            from utils.ufcstats_sync import run_sync

            _log(f"--- ufcstats sync start since={since} ---")
            n = run_sync(since=date_cls.fromisoformat(since), sleep_fight=0.35, sleep_event=0.55)
            _log(f"done, new rows reported: {n}")
        except Exception:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n--- UFCSTATS SYNC THREAD EXCEPTION ---\n")
                f.write(traceback.format_exc())

    threading.Thread(target=_work, daemon=True).start()
