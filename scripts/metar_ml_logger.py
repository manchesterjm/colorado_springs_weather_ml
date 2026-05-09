"""Hourly METAR logger for the 23-station forecasting network.

Pulls the latest observation from the NWS API for each station and inserts
into ``metar_hourly`` (source='NWS_API'). Uses the existing
``script_metrics`` infrastructure for run/item tracking. Designed to be
called once per hour at :05 by Windows Task Scheduler (see
``Create-MLLoggerTasks.ps1``).

The UNIQUE(station_id, observation_time) constraint silently dedupes
observations that haven't ticked over yet, so the script is safe to run
more often than once per hour.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
import urllib.error
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg import db, nws_obs, stations  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.io import script_metrics, tz_utils  # noqa: E402  pylint: disable=wrong-import-position

LOG_PATH = _ROOT / "data" / "metar_ml_logger.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("metar_ml_logger")

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5


def fetch_with_retry(icao: str) -> Optional[dict]:
    """Fetch an observation JSON with bounded retry on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return nws_obs.fetch_latest(icao)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            wait = RETRY_BACKOFF_SEC * attempt
            logger.warning(
                "fetch %s attempt %d/%d failed: %s; sleeping %ds",
                icao, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    return None


def insert_observation(
    conn: sqlite3.Connection, fetch_time: str, station_icao: str, parsed: dict
) -> int:
    """Insert an observation. Returns rows actually inserted (0 on duplicate)."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO metar_hourly (
            fetch_time, source, station_id, observation_time,
            temperature_c, dewpoint_c,
            wind_direction_deg, wind_speed_kt, wind_gust_kt,
            visibility_sm, precip_in, pressure_inhg,
            sky_condition, weather_phenomena, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fetch_time, "NWS_API", station_icao, parsed["observation_time"],
            parsed["temperature_c"], parsed["dewpoint_c"],
            parsed["wind_direction_deg"], parsed["wind_speed_kt"], parsed["wind_gust_kt"],
            parsed["visibility_sm"], parsed["precip_in"], parsed["pressure_inhg"],
            parsed["sky_condition"], parsed["weather_phenomena"], parsed["raw_data"],
        ),
    )
    return cur.rowcount or 0


def run_one_station(
    conn: sqlite3.Connection, fetch_time: str, icao: str,
    metrics: script_metrics.ScriptMetrics,
) -> None:
    with metrics.track_item(icao, "station") as item:
        obs = fetch_with_retry(icao)
        if obs is None:
            item.status = "failed"
            item.error_message = "fetch returned None (404 or retries exhausted)"
            return
        parsed = nws_obs.parse_observation(obs)
        if parsed is None:
            item.status = "failed"
            item.error_message = "no usable observation_time in response"
            return
        # Override station_icao with the requested one (the API echoes it,
        # but force consistency in case of weird responses).
        parsed["station_icao"] = icao
        rows = insert_observation(conn, fetch_time, icao, parsed)
        conn.commit()
        item.records_inserted = rows


def main() -> int:
    db.init_db()
    fetch_time = tz_utils.now_utc()
    icaos = [s.icao for s in stations.STATIONS]

    conn = db.get_connection()
    try:
        with script_metrics.ScriptMetrics(
            "metar_ml_logger", expected_items=len(icaos)
        ) as metrics:
            for icao in icaos:
                try:
                    run_one_station(conn, fetch_time, icao, metrics)
                except sqlite3.Error as exc:
                    logger.exception("DB error for %s", icao)
                    metrics.item_failed(icao, str(exc)[:300], item_type="station")
            inserted = sum(
                i.records_inserted for i in metrics.items.values() if i.status == "success"
            )
            logger.info(
                "metar_ml_logger done: %d/%d stations succeeded, %d new rows inserted",
                sum(1 for i in metrics.items.values() if i.status == "success"),
                len(icaos), inserted,
            )
        return 0 if metrics.status != "failed" else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
