"""Derive ``daily_summary`` rows from ``metar_hourly`` observations.

Per-station-per-day aggregates of Tmax / Tmin / Tavg, precipitation,
mean dewpoint, max gust, hours-with-precip, hours-observed, mean pressure
and mean visibility. ``date_local`` follows the climatic-day convention
(local America/Denver date, midnight-to-midnight).

Two correctness considerations:

1. **SPECI duplication.** During convective storms, SPECI reports inside
   the same UTC hour each carry an AA1 1-hour precipitation value that
   refers to the same physical hour, so summing all rows over-counts.
   The aggregator collapses each (station, UTC-hour) bucket to a single
   "best" row (the highest precipitation reading wins, since SPECI
   typically arrives as conditions intensify).

2. **Local date crossing.** The 06Z observation belongs to the previous
   Mountain-time day; the 23Z observation belongs to the same day. We
   convert UTC -> America/Denver via ``zoneinfo`` so DST is handled.

Run modes::

    # Aggregate everything that doesn't already have a daily_summary row
    python daily_aggregator.py

    # Force re-aggregate one station
    python daily_aggregator.py --station KCOS --force

    # Aggregate a specific date range
    python daily_aggregator.py --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Optional

from forecast_ml_pkg import db, stations
from forecast_ml_pkg.dates import DENVER
from forecast_ml_pkg.externals import tz_utils

LOG_PATH = db.PROJECT_ROOT / "data" / "daily_aggregator.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("daily_aggregator")

PRECIP_HOUR_THRESHOLD_IN = 0.005  # Trace cutoff for "hour with precip"


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------

def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _safe_max(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return max(cleaned) if cleaned else None


def _safe_min(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return min(cleaned) if cleaned else None


def parse_obs_time(text: str) -> datetime:
    """Parse a UTC ISO timestamp (with or without trailing Z) to aware datetime."""
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def hour_bucket(dt_utc: datetime) -> str:
    """Return ``YYYY-MM-DDTHH`` UTC hour key for de-duplication."""
    return dt_utc.strftime("%Y-%m-%dT%H")


def local_date(dt_utc: datetime) -> str:
    """Return the America/Denver local date string ``YYYY-MM-DD`` for a UTC dt."""
    return dt_utc.astimezone(DENVER).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Per-day aggregation
# ---------------------------------------------------------------------------

def aggregate_day_rows(rows: list[dict]) -> dict:
    """Aggregate the deduped hour-rows for one (station, date_local) into a summary.

    ``rows`` is the list of best-observation-per-hour dicts in any order.
    """
    temps = [r["temperature_c"] for r in rows]
    dews = [r["dewpoint_c"] for r in rows]
    gusts = [r["wind_gust_kt"] for r in rows]
    pressures = [r["pressure_inhg"] for r in rows]
    vises = [r["visibility_sm"] for r in rows]
    precips = [r["precip_in"] for r in rows if r["precip_in"] is not None]

    tmax = _safe_max(temps)
    tmin = _safe_min(temps)
    tavg = (tmax + tmin) / 2.0 if tmax is not None and tmin is not None else None

    precip_total = sum(precips) if precips else 0.0
    hours_with_precip = sum(1 for p in precips if p >= PRECIP_HOUR_THRESHOLD_IN)
    hours_observed = len({r["hour_bucket"] for r in rows})

    return {
        "tmax_c": tmax,
        "tmin_c": tmin,
        "tavg_c": tavg,
        "precip_in": precip_total,
        "mean_dewpoint_c": _safe_mean(dews),
        "max_gust_kt": _safe_max(gusts),
        "hours_with_precip": hours_with_precip,
        "hours_observed": hours_observed,
        "mean_pressure_inhg": _safe_mean(pressures),
        "mean_visibility_sm": _safe_mean(vises),
    }


def collect_hour_buckets(
    raw_rows: Iterable[tuple],
) -> dict[str, dict[str, list[dict]]]:
    """Reduce raw observation rows to {date_local: {hour_bucket: [rows...]}}.

    Each input tuple is ``(observation_time, temperature_c, dewpoint_c,
    wind_gust_kt, visibility_sm, precip_in, pressure_inhg)``.
    """
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for (obs_time, temp_c, dew_c, gust_kt, vis_smi, precip_in, pres_inhg) in raw_rows:
        try:
            dt_utc = parse_obs_time(obs_time)
        except (ValueError, TypeError):
            continue
        date_loc = local_date(dt_utc)
        hour_key = hour_bucket(dt_utc)
        buckets[date_loc][hour_key].append({
            "hour_bucket": hour_key,
            "temperature_c": temp_c,
            "dewpoint_c": dew_c,
            "wind_gust_kt": gust_kt,
            "visibility_sm": vis_smi,
            "precip_in": precip_in,
            "pressure_inhg": pres_inhg,
        })
    return buckets


def collapse_hour_bucket(rows: list[dict]) -> dict:
    """Return one representative row for a UTC-hour bucket.

    Strategy: take the row with the highest precipitation reading (resolves
    SPECI duplication during storms); ties broken by latest temperature
    presence. For non-precip fields, use that row's values directly.
    """
    if len(rows) == 1:
        return rows[0]
    best = rows[0]
    best_precip = best["precip_in"] if best["precip_in"] is not None else -1.0
    for row in rows[1:]:
        candidate = row["precip_in"] if row["precip_in"] is not None else -1.0
        if candidate > best_precip:
            best, best_precip = row, candidate
    return best


# ---------------------------------------------------------------------------
# Per-station aggregation
# ---------------------------------------------------------------------------

def existing_dates(conn: sqlite3.Connection, station_id: str) -> set[str]:
    cur = conn.execute(
        "SELECT date_local FROM daily_summary WHERE station_id = ?", (station_id,)
    )
    return {row[0] for row in cur.fetchall()}


def fetch_station_observations(
    conn: sqlite3.Connection,
    station_id: str,
    start: Optional[str],
    end: Optional[str],
) -> list[tuple]:
    sql = (
        "SELECT observation_time, temperature_c, dewpoint_c, wind_gust_kt, "
        "visibility_sm, precip_in, pressure_inhg FROM metar_hourly "
        "WHERE station_id = ?"
    )
    params: list = [station_id]
    if start:
        sql += " AND observation_time >= ?"
        params.append(start)
    if end:
        sql += " AND observation_time < ?"
        params.append(end)
    return conn.execute(sql, params).fetchall()


def upsert_daily_summary(
    conn: sqlite3.Connection,
    station_id: str,
    date_local: str,
    summary: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO daily_summary (
            station_id, date_local,
            tmax_c, tmin_c, tavg_c,
            precip_in, mean_dewpoint_c, max_gust_kt,
            hours_with_precip, hours_observed,
            mean_pressure_inhg, mean_visibility_sm,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_id, date_local) DO UPDATE SET
            tmax_c = excluded.tmax_c,
            tmin_c = excluded.tmin_c,
            tavg_c = excluded.tavg_c,
            precip_in = excluded.precip_in,
            mean_dewpoint_c = excluded.mean_dewpoint_c,
            max_gust_kt = excluded.max_gust_kt,
            hours_with_precip = excluded.hours_with_precip,
            hours_observed = excluded.hours_observed,
            mean_pressure_inhg = excluded.mean_pressure_inhg,
            mean_visibility_sm = excluded.mean_visibility_sm,
            updated_at = excluded.updated_at
        """,
        (
            station_id, date_local,
            summary["tmax_c"], summary["tmin_c"], summary["tavg_c"],
            summary["precip_in"], summary["mean_dewpoint_c"], summary["max_gust_kt"],
            summary["hours_with_precip"], summary["hours_observed"],
            summary["mean_pressure_inhg"], summary["mean_visibility_sm"],
            tz_utils.now_utc(),
        ),
    )


def aggregate_station(
    conn: sqlite3.Connection,
    station_id: str,
    start: Optional[str],
    end: Optional[str],
    force: bool,
) -> tuple[int, int]:
    """Aggregate one station's hourly observations into daily_summary rows.

    Returns ``(dates_written, dates_skipped)``.
    """
    raw_rows = fetch_station_observations(conn, station_id, start, end)
    if not raw_rows:
        logger.info("  %s: no metar_hourly rows in range; skipping", station_id)
        return 0, 0

    buckets = collect_hour_buckets(raw_rows)
    skip_dates = set() if force else existing_dates(conn, station_id)

    written = 0
    skipped = 0
    for date_loc in sorted(buckets):
        if date_loc in skip_dates:
            skipped += 1
            continue
        per_hour = [collapse_hour_bucket(rows) for rows in buckets[date_loc].values()]
        summary = aggregate_day_rows(per_hour)
        upsert_daily_summary(conn, station_id, date_loc, summary)
        written += 1

    conn.commit()
    return written, skipped


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate metar_hourly into daily_summary")
    parser.add_argument("--station", help="Single ICAO code (default: all)")
    parser.add_argument("--start", help="UTC start (YYYY-MM-DD or full ISO)")
    parser.add_argument("--end", help="UTC end exclusive (YYYY-MM-DD or full ISO)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing daily_summary rows in range",
    )
    args = parser.parse_args()

    db.init_db()

    if args.station:
        try:
            stations.get_station(args.station)
        except KeyError:
            logger.error("Unknown ICAO: %s", args.station)
            return 2
        codes = [args.station.upper()]
    else:
        codes = [s.icao for s in stations.STATIONS]

    start = args.start
    end = args.end
    if start and len(start) == 10:
        start = f"{start}T00:00:00Z"
    if end and len(end) == 10:
        end = f"{end}T00:00:00Z"

    conn = db.get_connection()
    grand_written = 0
    grand_skipped = 0
    try:
        for code in codes:
            written, skipped = aggregate_station(conn, code, start, end, args.force)
            grand_written += written
            grand_skipped += skipped
            logger.info("  %s: wrote %d days, skipped %d existing", code, written, skipped)
    finally:
        conn.close()

    logger.info(
        "Aggregator done. dates_written=%d, dates_skipped=%d", grand_written, grand_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
