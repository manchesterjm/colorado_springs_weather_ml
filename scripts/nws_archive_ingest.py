"""Ingest historical MEX MOS bulletins from the Iowa State Mesonet archive.

Fetches one bulletin per day (12Z run) for KCOS from the IEM archive,
inserts the raw rows into ``mex_mos_raw`` and derived per-target-date Hi/Lo/
PoP rows into ``nws_daily_forecast`` (source='MEX_MOS'). Resumable via
``nws_archive_progress``: skips runtimes already marked 'completed' or
'no_data'.

Defaults to a 2021-01-01 -> today window. Override with --start / --end.

Usage::

    python scripts/nws_archive_ingest.py
    python scripts/nws_archive_ingest.py --start 2021-01-01 --end 2025-12-31
    python scripts/nws_archive_ingest.py --retry-failed
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
import urllib.error
from datetime import date, datetime, timedelta, timezone

from forecast_ml_pkg import db, mex_mos, netfetch
from forecast_ml_pkg.externals import tz_utils

STATION = "KCOS"
MODEL = "MEX"
RUNTIME_HOUR = 12  # MEX is issued at 00Z and 12Z; we use 12Z for production parity.
DEFAULT_START = date(2021, 1, 1)
SLEEP_BETWEEN_CALLS_SEC = 0.25  # ~4 req/sec
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 10

LOG_PATH = db.PROJECT_ROOT / "data" / "nws_archive_ingest.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("nws_archive_ingest")


def daterange(start: date, end_inclusive: date):
    """Yield each date from start through end inclusive."""
    cur = start
    while cur <= end_inclusive:
        yield cur
        cur = cur + timedelta(days=1)


def runtime_iso(d: date, hour: int = RUNTIME_HOUR) -> str:
    """Return the IEM-style runtime ISO string for one (date, hour) pair."""
    return f"{d.isoformat()}T{hour:02d}:00:00Z"


def fetch_with_retry(station: str, runtime: str) -> list[dict] | None:
    """Fetch a bulletin with bounded backoff. A 404 means 'no bulletin' -> None."""
    def _fetch() -> list[dict] | None:
        try:
            return mex_mos.fetch_bulletin(station, runtime)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    try:
        return netfetch.fetch_with_retry(
            _fetch, retries=MAX_RETRIES, backoff_sec=RETRY_BACKOFF_SEC,
            label=f"fetch {station} @ {runtime}", logger=logger,
        )
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(
            f"fetch failed after {MAX_RETRIES} attempts: {station} @ {runtime}"
        ) from exc


def insert_raw(conn: sqlite3.Connection, fetched_at: str, rows: list[mex_mos.MexRow]) -> int:
    """Insert MexRow records; returns count actually inserted."""
    n = 0
    for r in rows:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO mex_mos_raw (
                station_id, runtime_utc, ftime_utc,
                n_x, tmp, dpt, wsp, p12, q12, t12, snw,
                raw_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.station_id, r.runtime_utc, r.ftime_utc,
                r.n_x, r.tmp, r.dpt, r.wsp, r.p12, r.q12, r.t12, r.snw,
                r.raw_json, fetched_at,
            ),
        )
        n += cur.rowcount or 0
    return n


def insert_daily(
    conn: sqlite3.Connection,
    station: str,
    created_at: str,
    entries: list[mex_mos.DailyEntry],
) -> int:
    """Insert derived per-target-date entries; returns count actually inserted."""
    n = 0
    for e in entries:
        if e.horizon_days < 1:
            continue
        cur = conn.execute(
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
                e.nws_pop_day, e.nws_pop_night, "MEX_MOS", created_at,
            ),
        )
        n += cur.rowcount or 0
    return n


def upsert_progress(
    conn: sqlite3.Connection,
    runtime: str,
    status: str,
    rows_inserted: int = 0,
    error_message: str | None = None,
) -> None:
    """Insert or update the progress row for one (model, station, runtime)."""
    now = tz_utils.now_utc()
    conn.execute(
        """
        INSERT INTO nws_archive_progress (
            model, station_id, runtime_utc, status, rows_inserted,
            attempted_at, completed_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model, station_id, runtime_utc) DO UPDATE SET
            status = excluded.status,
            rows_inserted = excluded.rows_inserted,
            attempted_at = excluded.attempted_at,
            completed_at = excluded.completed_at,
            error_message = excluded.error_message
        """,
        (
            MODEL, STATION, runtime, status, rows_inserted,
            now, (now if status == "completed" else None), error_message,
        ),
    )


def completed_runtimes(conn: sqlite3.Connection, retry_failed: bool) -> set[str]:
    """Return runtimes we should skip on this pass."""
    statuses = ("completed", "no_data")
    if not retry_failed:
        statuses = statuses + ("failed",)
    placeholders = ",".join("?" * len(statuses))
    rows = conn.execute(
        f"""
        SELECT runtime_utc FROM nws_archive_progress
        WHERE model = ? AND station_id = ? AND status IN ({placeholders})
        """,
        (MODEL, STATION, *statuses),
    ).fetchall()
    return {r[0] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START,
                        help="Inclusive start date (default 2021-01-01)")
    parser.add_argument("--end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date(),
                        help="Inclusive end date (default today UTC)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt runtimes previously marked 'failed'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many fetches this run (testing)")
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be >= --start")

    db.init_db()
    conn = db.get_connection()
    try:
        skip = completed_runtimes(conn, retry_failed=args.retry_failed)
        all_runtimes = [runtime_iso(d) for d in daterange(args.start, args.end)]
        todo = [r for r in all_runtimes if r not in skip]
        logger.info(
            "MEX archive ingest: %s -> %s (%d total, %d already done, %d todo)",
            args.start, args.end, len(all_runtimes), len(skip), len(todo),
        )

        fetched = 0
        succeeded = 0
        no_data = 0
        failed = 0
        raw_rows_inserted = 0
        daily_rows_inserted = 0

        for runtime in todo:
            if args.limit is not None and fetched >= args.limit:
                logger.info("--limit reached; stopping early")
                break
            fetched += 1
            try:
                raw = fetch_with_retry(STATION, runtime)
            except RuntimeError as exc:
                upsert_progress(conn, runtime, "failed", error_message=str(exc))
                conn.commit()
                failed += 1
                logger.error("permanent failure for %s: %s", runtime, exc)
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                continue

            if raw is None:
                upsert_progress(conn, runtime, "no_data")
                conn.commit()
                no_data += 1
                logger.info("no_data for %s", runtime)
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                continue

            fetched_at = tz_utils.now_utc()
            norm = mex_mos.normalize_rows(STATION, raw)
            daily = mex_mos.derive_daily_entries(norm)
            n_raw = insert_raw(conn, fetched_at, norm)
            n_daily = insert_daily(conn, STATION, fetched_at, daily)
            upsert_progress(conn, runtime, "completed", rows_inserted=n_raw)
            conn.commit()
            succeeded += 1
            raw_rows_inserted += n_raw
            daily_rows_inserted += n_daily
            if fetched % 50 == 0:
                logger.info(
                    "progress: fetched=%d (ok=%d no_data=%d fail=%d) raw=%d daily=%d",
                    fetched, succeeded, no_data, failed,
                    raw_rows_inserted, daily_rows_inserted,
                )
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        logger.info(
            "done: fetched=%d ok=%d no_data=%d failed=%d raw_inserted=%d daily_inserted=%d",
            fetched, succeeded, no_data, failed, raw_rows_inserted, daily_rows_inserted,
        )
        return 0 if failed == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
