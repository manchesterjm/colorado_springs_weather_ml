"""Aggregate NWS hourly digital_forecast rows to per-target-date Hi/Lo/PoP.

The source data lives in ``D:\\Scripts\\weather_data\\weather.db`` table
``digital_forecast``. Each row is a single hourly forecast point keyed by
``(fetch_time, forecast_date, forecast_hour)``. ``forecast_date`` and
``forecast_hour`` are in **local Mountain time** (verified empirically:
overnight minima land at hours 1-6, daily peak lands at hours 13-15).

This module collapses one fetch into a list of ``DailyEntry`` records
identical in shape to ``mex_mos.DailyEntry``, so both training (MEX archive)
and inference (live NWS) can write to ``nws_daily_forecast`` with the same
downstream schema.

Definitions used here (climatic-day convention to match actuals):
    * Hi(D) = max(temperature) over forecast_date = D, all hours
    * Lo(D) = min(temperature) over forecast_date = D, all hours
    * PoP_day(D) = max(precip_probability) for forecast_hour in [6, 17]
    * PoP_night(D) = max(precip_probability) for forecast_hour in [0, 5] U [18, 23]

These differ from MEX's 12-hour MOS windows by roughly +/- 1F on Hi/Lo;
the difference is well below model noise and below MOS rounding.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from forecast_ml_pkg import constants
from forecast_ml_pkg.dates import utc_iso_to_denver_date
from forecast_ml_pkg.externals import db_utils
from forecast_ml_pkg.mex_mos import DailyEntry  # reuse the same dataclass

logger = logging.getLogger(__name__)

# Local-hour ranges used for daytime/overnight PoP aggregation.
DAY_HOURS = range(6, 18)        # 06:00 .. 17:59 inclusive
NIGHT_HOURS = list(range(0, 6)) + list(range(18, 24))


def open_weather_db_ro(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a read-only connection to weather.db.

    Defaults to the canonical ``db_utils.DB_PATH``. Read-only mode keeps this
    aggregator from ever contending with the live weather loggers.
    """
    path = db_path or db_utils.DB_PATH
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA query_only=ON")
    return conn


def latest_fetch_time(conn: sqlite3.Connection) -> Optional[str]:
    """Return the most recent fetch_time, or None if the table is empty."""
    row = conn.execute("SELECT MAX(fetch_time) FROM digital_forecast").fetchone()
    return row[0] if row else None


def fetch_time_before(conn: sqlite3.Connection, cutoff_utc_iso: str) -> Optional[str]:
    """Return the most recent fetch_time at or before ``cutoff_utc_iso``.

    Used at training simulation time when we want to know "what NWS would have
    been visible to production at moment T." For live inference, prefer
    ``latest_fetch_time``.
    """
    row = conn.execute(
        "SELECT MAX(fetch_time) FROM digital_forecast WHERE fetch_time <= ?",
        (cutoff_utc_iso,),
    ).fetchone()
    return row[0] if row else None


@dataclass(frozen=True)
class _HourPoint:
    forecast_date: str
    forecast_hour: int
    temperature: Optional[int]
    precip_probability: Optional[int]


def _fetch_hourly_points(conn: sqlite3.Connection, fetch_time: str) -> list[_HourPoint]:
    rows = conn.execute(
        """
        SELECT forecast_date, forecast_hour, temperature, precip_probability
        FROM digital_forecast
        WHERE fetch_time = ?
        ORDER BY forecast_date, forecast_hour
        """,
        (fetch_time,),
    ).fetchall()
    return [_HourPoint(*r) for r in rows]


def aggregate_fetch(
    conn: sqlite3.Connection,
    fetch_time: str,
    min_hours_per_day: int = 12,
) -> list[DailyEntry]:
    """Return per-target-date entries derived from one ``fetch_time``.

    ``min_hours_per_day`` filters out fetch-day partial coverage that has too
    few hours to produce a meaningful Hi/Lo (e.g. a late-afternoon fetch only
    sees ~6 hours of "today"). Days with fewer hours are skipped entirely.
    """
    points = _fetch_hourly_points(conn, fetch_time)
    if not points:
        return []
    issue_local = utc_iso_to_denver_date(fetch_time)
    issue_obj = datetime.fromisoformat(issue_local).date()

    # Group by forecast_date.
    buckets: dict[str, list[_HourPoint]] = {}
    for p in points:
        buckets.setdefault(p.forecast_date, []).append(p)

    out: list[DailyEntry] = []
    for date_str in sorted(buckets):
        target_obj = datetime.fromisoformat(date_str).date()
        horizon = (target_obj - issue_obj).days
        if horizon < 1:
            # Skip today and earlier; we only forecast forward.
            continue
        rows = buckets[date_str]
        if len(rows) < min_hours_per_day:
            continue

        temps = [r.temperature for r in rows if r.temperature is not None]
        hi = max(temps) if temps else None
        lo = min(temps) if temps else None

        day_pops = [
            r.precip_probability for r in rows
            if r.forecast_hour in DAY_HOURS and r.precip_probability is not None
        ]
        night_pops = [
            r.precip_probability for r in rows
            if r.forecast_hour in NIGHT_HOURS and r.precip_probability is not None
        ]
        pop_day = max(day_pops) if day_pops else None
        pop_night = max(night_pops) if night_pops else None

        out.append(DailyEntry(
            issue_date_local=issue_local,
            target_date_local=date_str,
            horizon_days=horizon,
            nws_hi_f=float(hi) if hi is not None else None,
            nws_lo_f=float(lo) if lo is not None else None,
            nws_pop_day=float(pop_day) if pop_day is not None else None,
            nws_pop_night=float(pop_night) if pop_night is not None else None,
        ))
    return out


def upsert_entries(
    fc_conn: sqlite3.Connection,
    created_at: str,
    entries: list[DailyEntry],
    station: str = constants.STATION_KCOS,
) -> int:
    """Insert DailyEntry rows into ``nws_daily_forecast`` with source='NWS_DIGITAL'.

    ``fc_conn`` is the forecast_ml.db connection. Idempotent: the UNIQUE
    constraint (station, issue_date_local, target_date_local, source) silently
    dedupes re-runs.
    """
    n = 0
    for e in entries:
        if e.horizon_days < 1:
            continue
        cur = fc_conn.execute(
            """
            INSERT OR IGNORE INTO nws_daily_forecast (
                station_id, issue_date_local, target_date_local,
                horizon_days, nws_hi_f, nws_lo_f,
                nws_pop_day, nws_pop_night, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                station, e.issue_date_local, e.target_date_local,
                e.horizon_days, e.nws_hi_f, e.nws_lo_f,
                e.nws_pop_day, e.nws_pop_night, "NWS_DIGITAL", created_at,
            ),
        )
        n += cur.rowcount or 0
    return n


def aggregate_all_fetches(
    conn: sqlite3.Connection,
    one_per_day: bool = True,
) -> list[DailyEntry]:
    """Aggregate every fetch_time present in ``digital_forecast``.

    With ``one_per_day=True`` (default), if multiple fetches exist on the same
    UTC date, only the **earliest** fetch of that day is used. This matches
    production behavior: the forecast issued at 06:30 MDT (~12:30Z) uses the
    morning's NWS feed, not an afternoon one. Earliest-of-day approximates
    "what was visible at issue time" for historical NWS_DIGITAL backfill.
    """
    if one_per_day:
        rows = conn.execute(
            """
            SELECT MIN(fetch_time) AS earliest, substr(fetch_time,1,10) AS d
            FROM digital_forecast
            GROUP BY d
            ORDER BY d
            """
        ).fetchall()
        fetch_times = [r[0] for r in rows]
    else:
        fetch_times = [r[0] for r in conn.execute(
            "SELECT DISTINCT fetch_time FROM digital_forecast ORDER BY fetch_time"
        ).fetchall()]

    out: list[DailyEntry] = []
    for ft in fetch_times:
        out.extend(aggregate_fetch(conn, ft))
    return out
