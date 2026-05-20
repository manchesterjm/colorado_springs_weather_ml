"""Schema definitions for ``forecast_ml.db``.

The database owns all project-specific data: extended-station hourly METAR,
derived daily summaries, production forecasts, verification, the model
registry, the training cache, and a backfill progress tracker.

It does *not* duplicate data already in ``D:\\Scripts\\weather_data\\weather.db``
(PWS, NWS forecast snapshots, GFS/NBM, actual_daily_climate). The forecast
runner reads those sources at use-time.

All timestamps are UTC with a ``Z`` suffix per the project-wide policy
(see ``tz_utils.py``). Dates in the ``_local`` columns are
``YYYY-MM-DD`` in ``America/Denver``.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

from forecast_ml_pkg.externals import db_utils

# Path-aware DB location so this works the same in Windows and WSL.
if sys.platform == "win32":
    PROJECT_ROOT = Path(r"D:\Projects\CO_Springs_Weather_ML")
else:
    PROJECT_ROOT = Path("/mnt/d/Projects/CO_Springs_Weather_ML")

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "forecast_ml.db"
ISD_HISTORY_CACHE = DATA_DIR / "isd_history.csv"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA: tuple[str, ...] = (
    # Hourly METAR for the 22-station network. Source can be NCEI_ISD
    # (historical backfill) or NWS_API (live ingestion). The same
    # (station_id, observation_time) row is unique across both sources.
    """
    CREATE TABLE IF NOT EXISTS metar_hourly (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fetch_time TEXT NOT NULL,
        source TEXT NOT NULL,
        station_id TEXT NOT NULL,
        observation_time TEXT NOT NULL,
        temperature_c REAL,
        dewpoint_c REAL,
        wind_direction_deg INTEGER,
        wind_speed_kt REAL,
        wind_gust_kt REAL,
        visibility_sm REAL,
        precip_in REAL,
        pressure_inhg REAL,
        sky_condition TEXT,
        weather_phenomena TEXT,
        raw_data TEXT,
        UNIQUE(station_id, observation_time)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_metar_hourly_obs ON metar_hourly(observation_time)",
    "CREATE INDEX IF NOT EXISTS idx_metar_hourly_station ON metar_hourly(station_id)",
    """
    CREATE INDEX IF NOT EXISTS idx_metar_hourly_station_obs
        ON metar_hourly(station_id, observation_time)
    """,

    # Per-station daily aggregates. ``date_local`` is the Mountain-time date
    # (climatic-day convention). ``hours_observed`` measures completeness so
    # downstream feature builders can flag short days.
    """
    CREATE TABLE IF NOT EXISTS daily_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        date_local TEXT NOT NULL,
        tmax_c REAL,
        tmin_c REAL,
        tavg_c REAL,
        precip_in REAL,
        mean_dewpoint_c REAL,
        max_gust_kt REAL,
        hours_with_precip INTEGER,
        hours_observed INTEGER,
        mean_pressure_inhg REAL,
        mean_visibility_sm REAL,
        updated_at TEXT NOT NULL,
        UNIQUE(station_id, date_local)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date_local)",
    "CREATE INDEX IF NOT EXISTS idx_daily_summary_station ON daily_summary(station_id)",

    # One row per (issue_date, target_date, model_version) combo; horizon
    # is derived but stored for fast filtering. ``regime_probs_json`` holds
    # the full 9-state probability vector. ``nws_compare_json`` snapshots the
    # NWS forecast for the same target at issue time so verification later
    # can reproduce the comparison without re-fetching anything.
    """
    CREATE TABLE IF NOT EXISTS production_forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_time TEXT NOT NULL,
        issue_date_local TEXT NOT NULL,
        target_date_local TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        regime_probs_json TEXT NOT NULL,
        regime_top1 TEXT NOT NULL,
        hi_f REAL,
        lo_f REAL,
        pop REAL,
        pop_threshold_in REAL DEFAULT 0.10,
        nws_compare_json TEXT,
        ml_blend_json TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(issue_date_local, target_date_local, model_version)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prodfc_target
        ON production_forecast(target_date_local)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prodfc_issue
        ON production_forecast(issue_date_local)
    """,

    # Verification: actual-vs-forecast metrics for each row in
    # production_forecast. Populated after target_date_local <= today.
    """
    CREATE TABLE IF NOT EXISTS forecast_verification (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_id INTEGER NOT NULL UNIQUE,
        target_date_local TEXT NOT NULL,
        actual_regime TEXT,
        actual_hi_f REAL,
        actual_lo_f REAL,
        actual_precip_in REAL,
        actual_pop_hit INTEGER,
        regime_top1_hit INTEGER,
        regime_p_actual REAL,
        hi_error_f REAL,
        lo_error_f REAL,
        pop_brier REAL,
        nws_regime_hit INTEGER,
        nws_hi_error_f REAL,
        nws_lo_error_f REAL,
        nws_pop_brier REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (forecast_id) REFERENCES production_forecast(id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_verif_target
        ON forecast_verification(target_date_local)
    """,

    # Model registry. ``is_active`` is enforced by the retrain pipeline:
    # exactly one row should have is_active=1.
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id TEXT PRIMARY KEY,
        model_version TEXT NOT NULL,
        training_start_date TEXT NOT NULL,
        training_end_date TEXT NOT NULL,
        val_start_date TEXT,
        val_end_date TEXT,
        hyperparams_json TEXT,
        val_metrics_json TEXT,
        feature_names_json TEXT,
        artifact_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 0,
        variant TEXT NOT NULL DEFAULT 'baseline'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_model_runs_created ON model_runs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_model_runs_active ON model_runs(is_active)",
    "CREATE INDEX IF NOT EXISTS idx_model_runs_variant ON model_runs(variant)",

    # Training cache: materialized (anchor_date, horizon) feature rows.
    # ``feature_json`` stores the full feature dict so retrains do not need
    # to recompute from raw observations every night.
    """
    CREATE TABLE IF NOT EXISTS training_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        anchor_date_local TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        feature_json TEXT NOT NULL,
        target_state TEXT,
        target_hi_f REAL,
        target_lo_f REAL,
        target_precip_in REAL,
        created_at TEXT NOT NULL,
        UNIQUE(anchor_date_local, horizon_days)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_training_cache_anchor
        ON training_cache(anchor_date_local)
    """,

    # NWS forecast snapshot at issue time, indexed by (issue, target).
    # This is a denormalized cache that survives even if the source DB
    # rolls forecast snapshots off later.
    """
    CREATE TABLE IF NOT EXISTS nws_compare_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_date_local TEXT NOT NULL,
        target_date_local TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        nws_hi_f REAL,
        nws_lo_f REAL,
        nws_pop REAL,
        nws_regime TEXT,
        nws_text TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(issue_date_local, target_date_local)
    )
    """,

    # Backfill progress tracker so the NCEI ingestion is resumable.
    # Each (station_id, year) pair gets one row; the backfill script
    # skips rows whose status is 'completed'.
    """
    CREATE TABLE IF NOT EXISTS backfill_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        year INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        rows_inserted INTEGER NOT NULL DEFAULT 0,
        attempted_at TEXT,
        completed_at TEXT,
        error_message TEXT,
        UNIQUE(station_id, year)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_backfill_status
        ON backfill_progress(status)
    """,

    # Raw MEX MOS bulletins from the Iowa State Mesonet archive. One row per
    # (station, runtime, ftime). Stores the parsed JSON record verbatim so any
    # future feature work can pull additional MOS fields (dewpoint, wind,
    # thunderstorm prob, etc.) without re-fetching from the archive.
    """
    CREATE TABLE IF NOT EXISTS mex_mos_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        runtime_utc TEXT NOT NULL,
        ftime_utc TEXT NOT NULL,
        n_x INTEGER,
        tmp INTEGER,
        dpt INTEGER,
        wsp INTEGER,
        p12 INTEGER,
        q12 REAL,
        t12 TEXT,
        snw REAL,
        raw_json TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        UNIQUE(station_id, runtime_utc, ftime_utc)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mex_raw_runtime
        ON mex_mos_raw(runtime_utc)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mex_raw_ftime
        ON mex_mos_raw(ftime_utc)
    """,

    # Derived per-target-date NWS forecast Hi/Lo/PoP used as ML features.
    # source identifies origin: 'MEX_MOS' for historical bulletins, or
    # 'NWS_DIGITAL' for live aggregates from weather.db.digital_forecast.
    # horizon_days = (target - issue) in days; negative is meaningless for
    # this table (rows where target_date_local <= issue_date_local are
    # excluded at insert time).
    """
    CREATE TABLE IF NOT EXISTS nws_daily_forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        issue_date_local TEXT NOT NULL,
        target_date_local TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        nws_hi_f REAL,
        nws_lo_f REAL,
        nws_pop_day REAL,
        nws_pop_night REAL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(station_id, issue_date_local, target_date_local, source)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_nws_daily_issue
        ON nws_daily_forecast(issue_date_local)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_nws_daily_target
        ON nws_daily_forecast(target_date_local)
    """,

    # Resumable progress tracker for the MOS archive ingestion script.
    # One row per (model, station, runtime_utc). Status values:
    # 'pending', 'completed', 'no_data', 'failed'.
    """
    CREATE TABLE IF NOT EXISTS nws_archive_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT NOT NULL,
        station_id TEXT NOT NULL,
        runtime_utc TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        rows_inserted INTEGER NOT NULL DEFAULT 0,
        attempted_at TEXT,
        completed_at TEXT,
        error_message TEXT,
        UNIQUE(model, station_id, runtime_utc)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_nws_archive_status
        ON nws_archive_progress(status)
    """,
)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a sqlite connection to forecast_ml.db.

    Delegates to the shared ``db_utils.get_connection`` for the WAL + busy
    pragmas, then adds ``foreign_keys=ON`` which this project's schema relies
    on (``forecast_verification`` references ``production_forecast``).
    """
    conn = db_utils.get_connection(db_path or DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Idempotent in-place migrations for older DBs predating new columns.

    SQLite ``ALTER TABLE ADD COLUMN`` is itself idempotent only via the
    ``duplicate column`` error swallow below; we cannot use
    ``IF NOT EXISTS`` because that clause does not exist for ADD COLUMN.
    """
    migrations = [
        # Phase 7: A/B variant column on model_runs (default 'baseline').
        "ALTER TABLE model_runs ADD COLUMN variant TEXT NOT NULL DEFAULT 'baseline'",
    ]
    for stmt in migrations:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            # Fresh DB: target table doesn't exist yet — SCHEMA loop creates
            # it with the column already present, so the migration is moot.
            # Old DB: column already exists from a prior migration run.
            if (
                "duplicate column" in msg
                or "already exists" in msg
                or "no such table" in msg
            ):
                continue
            raise


def init_db(db_path: Path | None = None) -> Path:
    """Create the schema if it does not already exist.

    Idempotent — safe to call from any script. Applies in-place migrations
    for older DBs. Returns the resolved DB path.
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(path)
    try:
        # Migrations first: ensure new columns exist on legacy tables before
        # the SCHEMA loop tries to create indexes that reference them. The
        # migration helper is a no-op for fresh DBs (the CREATE TABLE in
        # SCHEMA already includes the new columns).
        _apply_migrations(conn)
        for stmt in SCHEMA:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    logger.info("forecast_ml schema initialized at %s", path)
    return path


def list_tables(db_path: Path | None = None) -> list[str]:
    """Return sorted table names present in the database."""
    conn = get_connection(db_path or DB_PATH)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    resolved = init_db()
    print(f"Initialized: {resolved}")
    print("Tables:")
    for tname in list_tables():
        print(f"  {tname}")
