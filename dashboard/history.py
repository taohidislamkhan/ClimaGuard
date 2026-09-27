"""Daily risk snapshots in SQLite (``risk_history``).

One row per (location, date). Live runs upsert today's row. When a location
has no history yet, it is backfilled with model outputs for the last seven
dataset weeks (``source = 'backfill'``) so the trend and "vs yesterday"
have something to compare against; the UI shows a Demo Data chip while any
backfilled row is in view.

Environment columns hold the week-level model inputs (PM2.5, AQI, rainfall,
temperature in real °C) so snapshot-to-snapshot changes compare like with like.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .inference import ROOT

DB_PATH = ROOT / "data" / "cache" / "risk_history.db"
SCORE_COLS = ["overall", "respiratory", "vector", "heat", "waterborne", "cardio"]
ENV_COLS = ["pm25", "aqi", "temperature", "rainfall"]

_lock = threading.Lock()


class History:
    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cols = ", ".join(f"{c} REAL" for c in SCORE_COLS + ENV_COLS)
        with _lock:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS risk_history (
                    location TEXT NOT NULL,
                    date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    overall_level TEXT,
                    {cols},
                    PRIMARY KEY (location, date))""")
            self.conn.commit()

    def has_rows(self, location: str) -> bool:
        with _lock:
            cur = self.conn.execute(
                "SELECT 1 FROM risk_history WHERE location = ? LIMIT 1", (location,))
            return cur.fetchone() is not None

    def upsert(self, location: str, date: str, source: str, level: str,
               scores: dict[str, float], env: dict[str, float | None]) -> None:
        cols = ["location", "date", "source", "overall_level"] + SCORE_COLS + ENV_COLS
        vals = [location, date, source, level] + [scores[c] for c in SCORE_COLS] \
            + [env.get(c) for c in ENV_COLS]
        with _lock:
            self.conn.execute(
                f"INSERT OR REPLACE INTO risk_history ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})", vals)
            self.conn.commit()

    def recent(self, location: str, n: int = 7) -> list[dict]:
        """Last ``n`` snapshots, oldest first."""
        with _lock:
            rows = self.conn.execute(
                "SELECT * FROM risk_history WHERE location = ? "
                "ORDER BY date DESC LIMIT ?", (location, n)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def previous(self, location: str, before_date: str) -> dict | None:
        with _lock:
            r = self.conn.execute(
                "SELECT * FROM risk_history WHERE location = ? AND date < ? "
                "ORDER BY date DESC LIMIT 1", (location, before_date)).fetchone()
        return dict(r) if r else None
