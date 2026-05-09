"""One-time NCEI ISD hourly METAR backfill for the 23-station network.

Per-station-per-year batched download + insert with progress tracking in
the ``backfill_progress`` table so the script is fully resumable. Default
behavior runs every (station, year) pair from 1973 through the previous
calendar year that does not already have status='completed'.

Examples::

    # Smoke test: KCOS only, 2024 only
    python metar_ncei_backfill.py --station KCOS --year 2024

    # Full run, resumable
    python metar_ncei_backfill.py

    # Run only the local stations
    python metar_ncei_backfill.py --role local

    # Override the year window
    python metar_ncei_backfill.py --start-year 2000 --end-year 2010
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Make the package importable when running as a plain script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg import db, ncei_isd, stations  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.io import tz_utils  # noqa: E402  pylint: disable=wrong-import-position

LOG_PATH = _ROOT / "data" / "metar_ncei_backfill.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("metar_ncei_backfill")

BATCH_SIZE = 5000
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 10


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def get_progress_status(conn: sqlite3.Connection, station_id: str, year: int) -> Optional[str]:
    cur = conn.execute(
        "SELECT status FROM backfill_progress WHERE station_id = ? AND year = ?",
        (station_id, year),
    )
    row = cur.fetchone()
    return row[0] if row else None


def upsert_progress(
    conn: sqlite3.Connection,
    station_id: str,
    year: int,
    status: str,
    rows_inserted: int = 0,
    error: Optional[str] = None,
) -> None:
    now = tz_utils.now_utc()
    completed_at = now if status in ("completed", "no_data") else None
    conn.execute(
        """
        INSERT INTO backfill_progress
            (station_id, year, status, rows_inserted, attempted_at, completed_at, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_id, year) DO UPDATE SET
            status = excluded.status,
            rows_inserted = excluded.rows_inserted,
            attempted_at = excluded.attempted_at,
            completed_at = excluded.completed_at,
            error_message = excluded.error_message
        """,
        (station_id, year, status, rows_inserted, now, completed_at, error),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fetch with retry
# ---------------------------------------------------------------------------

def fetch_with_retry(usaf: str, wban: str, year: int) -> tuple[Optional[str], str]:
    """Fetch a year of NCEI ISD CSV with backoff. Returns (csv_text|None, status).

    Status: 'ok' (got CSV), 'no_data' (404), 'failed' (transient errors exhausted).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            csv_text = ncei_isd.fetch_year_csv(usaf, wban, year)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            wait = RETRY_BACKOFF_SEC * attempt
            logger.warning(
                "Fetch error %s/%s/%d (attempt %d/%d): %s; sleeping %ds",
                usaf, wban, year, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
            continue
        if csv_text is None:
            return None, "no_data"
        return csv_text, "ok"
    return None, "failed"


# ---------------------------------------------------------------------------
# Per-(station, year) backfill
# ---------------------------------------------------------------------------

def chunked(iterable: Iterable, size: int) -> Iterator[list]:
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def insert_rows(
    conn: sqlite3.Connection,
    fetch_time: str,
    station_icao: str,
    rows: Iterable[dict[str, object]],
) -> int:
    """Insert a batch of parsed rows. Returns count actually inserted."""
    total = 0
    for batch in chunked(rows, BATCH_SIZE):
        params = [
            (
                fetch_time,
                "NCEI_ISD",
                station_icao,
                r["observation_time"],
                r["temperature_c"],
                r["dewpoint_c"],
                r["wind_direction_deg"],
                r["wind_speed_kt"],
                r["wind_gust_kt"],
                r["visibility_sm"],
                r["precip_in"],
                r["pressure_inhg"],
                r["sky_condition"],
                r["weather_phenomena"],
                r["raw_data"],
            )
            for r in batch
        ]
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO metar_hourly (
                fetch_time, source, station_id, observation_time,
                temperature_c, dewpoint_c,
                wind_direction_deg, wind_speed_kt, wind_gust_kt,
                visibility_sm, precip_in, pressure_inhg,
                sky_condition, weather_phenomena, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
    return total


def backfill_station_year(
    conn: sqlite3.Connection,
    station: stations.Station,
    year: int,
    isd_map: dict[str, tuple[str, str]],
) -> tuple[str, int]:
    """Backfill one (station, year) pair. Returns (status, rows_inserted)."""
    if station.icao not in isd_map:
        upsert_progress(conn, station.icao, year, "no_isd_id", 0, "ICAO not in NCEI history")
        return "no_isd_id", 0

    usaf, wban = isd_map[station.icao]
    upsert_progress(conn, station.icao, year, "in_progress")

    csv_text, fetch_status = fetch_with_retry(usaf, wban, year)
    if fetch_status == "no_data":
        upsert_progress(conn, station.icao, year, "no_data", 0, "404 from NCEI")
        return "no_data", 0
    if fetch_status == "failed" or csv_text is None:
        upsert_progress(conn, station.icao, year, "failed", 0, "fetch retries exhausted")
        return "failed", 0

    fetch_time = tz_utils.now_utc()
    try:
        inserted = insert_rows(
            conn,
            fetch_time,
            station.icao,
            ncei_isd.parse_year_csv(csv_text),
        )
    except (sqlite3.Error, ValueError) as exc:
        logger.exception("Insert failed for %s %d", station.icao, year)
        upsert_progress(conn, station.icao, year, "failed", 0, str(exc)[:500])
        return "failed", 0

    upsert_progress(conn, station.icao, year, "completed", inserted)
    return "completed", inserted


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def select_stations(role: Optional[str], icao: Optional[str]) -> list[stations.Station]:
    if icao:
        try:
            return [stations.get_station(icao)]
        except KeyError:
            logger.error("Unknown ICAO: %s", icao)
            sys.exit(2)
    if role:
        result = list(stations.stations_by_role(role))
        if not result:
            logger.error("No stations with role %s", role)
            sys.exit(2)
        return result
    return list(stations.STATIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="NCEI ISD hourly METAR backfill")
    parser.add_argument("--station", help="Single ICAO code (smoke-test mode)")
    parser.add_argument("--role", choices=["local", "adjacent", "mid_w", "sw", "far_w", "s"])
    parser.add_argument("--year", type=int, help="Single year (overrides start/end)")
    parser.add_argument("--start-year", type=int, default=1973)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year - 1)
    parser.add_argument(
        "--include-current-year",
        action="store_true",
        help="Also pull the current calendar year (NCEI updates it daily)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt rows currently marked 'failed'",
    )
    args = parser.parse_args()

    # Initialize schema (idempotent) so a fresh checkout works first try.
    db.init_db()

    selected = select_stations(args.role, args.station)
    if args.year:
        years = [args.year]
    else:
        end = args.end_year
        if args.include_current_year:
            end = datetime.now(timezone.utc).year
        years = list(range(args.start_year, end + 1))

    logger.info(
        "Backfill plan: %d station(s) x %d year(s) = %d (station,year) pairs",
        len(selected), len(years), len(selected) * len(years),
    )
    logger.info("Stations: %s", ", ".join(s.icao for s in selected))
    logger.info("Year range: %d-%d", years[0], years[-1])

    isd_map = stations.load_isd_history(db.ISD_HISTORY_CACHE)
    conn = db.get_connection()

    skip_statuses = {"completed", "no_data", "no_isd_id"}
    if not args.retry_failed:
        skip_statuses.add("failed")

    grand_total = 0
    pairs_attempted = 0
    pairs_skipped = 0
    pairs_failed = 0
    start = time.time()

    try:
        for station in selected:
            for year in years:
                status = get_progress_status(conn, station.icao, year)
                if status in skip_statuses:
                    pairs_skipped += 1
                    continue
                pairs_attempted += 1
                result, inserted = backfill_station_year(conn, station, year, isd_map)
                grand_total += inserted
                if result not in ("completed", "no_data", "no_isd_id"):
                    pairs_failed += 1
                logger.info(
                    "  %s %d -> %s (%d rows). Running total: %d rows in %d pairs.",
                    station.icao, year, result, inserted, grand_total, pairs_attempted,
                )
    finally:
        conn.close()

    elapsed = time.time() - start
    logger.info(
        "Backfill done. attempted=%d, skipped=%d, failed=%d, rows=%d, elapsed=%.1fs",
        pairs_attempted, pairs_skipped, pairs_failed, grand_total, elapsed,
    )
    return 1 if pairs_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
