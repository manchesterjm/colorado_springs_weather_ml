"""Score every issued forecast against the actual outcome once it lands.

For each ``production_forecast`` row whose ``target_date_local <= today``
that does not yet have a ``forecast_verification`` row, the script:

    1. Pulls actuals from the main weather DB
       (``actual_daily_climate`` for KCOS, station_id 'COS').
    2. Classifies the actual day into a 9-state regime using the active
       model's DOY terciles + project-wide precip tiers.
    3. Computes per-metric errors for ML and NWS heads (regime hit /
       p(actual), Hi/Lo signed error, PoP Brier).
    4. Inserts a row into ``forecast_verification`` (UNIQUE on forecast_id
       so reruns are idempotent).

Designed to run nightly at 23:55 MST -- by then KCOS has at minimum a
preliminary daily summary in the main DB.
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg import db as fdb  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.io import tz_utils  # noqa: E402  pylint: disable=wrong-import-position

from weather_regime_pkg import STATE_LABELS, doy  # noqa: E402  pylint: disable=wrong-import-position
from weather_regime_pkg.state_classifier import (  # noqa: E402  pylint: disable=wrong-import-position
    PRECIP_DRY_MAX, PRECIP_LIGHT_MAX, precip_tier, temp_tier,
)

WEATHER_DB = (
    Path(r"D:\Scripts\weather_data\weather.db") if sys.platform == "win32"
    else Path("/mnt/d/Scripts/weather_data/weather.db")
)
LOG_PATH = _ROOT / "data" / "verification_logger.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("verification_logger")

# Map our project's ICAO codes to actual_daily_climate's 3-letter ids.
ICAO_TO_DAILY = {
    "KCOS": "COS", "KFLY": "FLY", "KAFF": "AFF",
    "KFCS": "FCS", "KAPA": "APA", "KPUB": "PUB",
}


def _label_to_state(label: str) -> Optional[int]:
    try:
        return STATE_LABELS.index(label)
    except ValueError:
        return None


def fetch_actual_kcos(target_date: str) -> Optional[dict]:
    """Return ``{maxt, mint, pcpn}`` for the date or None if not yet recorded."""
    conn = sqlite3.connect(WEATHER_DB)
    try:
        row = conn.execute(
            "SELECT max_temp_f, min_temp_f, precip_in, is_precip_trace "
            "FROM actual_daily_climate "
            "WHERE station_id = 'COS' AND observation_date = ?",
            (target_date,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    maxt, mint, precip, trace = row
    if maxt is None or mint is None:
        # Same-day partial -- skip, we'll catch it on a later run.
        return None
    if precip is None:
        precip = 0.0 if trace == 0 else PRECIP_DRY_MAX  # treat trace as min wet
    return {"maxt": float(maxt), "mint": float(mint), "pcpn": float(precip)}


def classify_actual(actual: dict, terciles: dict, target_date: str) -> Optional[int]:
    """Classify the actual day using the artifact's DOY terciles."""
    avg = 0.5 * (actual["maxt"] + actual["mint"])
    tt = temp_tier(avg, terciles[doy(target_date)])
    pt = precip_tier(actual["pcpn"])
    if tt is None or pt is None:
        return None
    return tt * 3 + pt


def load_active_artifact() -> dict:
    conn = fdb.get_connection()
    try:
        row = conn.execute(
            "SELECT artifact_path FROM model_runs "
            "WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise RuntimeError("No active model in model_runs")
    with open(row[0], "rb") as fh:
        return pickle.load(fh)


def pending_forecasts(conn: sqlite3.Connection, today: str) -> list[dict]:
    """Forecasts due for verification (target landed, no verification yet)."""
    cur = conn.execute(
        """
        SELECT pf.id, pf.target_date_local, pf.horizon_days,
               pf.regime_top1, pf.regime_probs_json,
               pf.hi_f, pf.lo_f, pf.pop, pf.pop_threshold_in,
               pf.nws_compare_json
        FROM production_forecast pf
        LEFT JOIN forecast_verification fv ON fv.forecast_id = pf.id
        WHERE pf.target_date_local <= ? AND fv.id IS NULL
        ORDER BY pf.target_date_local, pf.horizon_days
        """,
        (today,),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def write_verification(
    conn: sqlite3.Connection, forecast_id: int, target_date: str,
    actual: dict, actual_state: Optional[int],
    forecast_row: dict, terciles: dict,
) -> None:
    probs = json.loads(forecast_row["regime_probs_json"])
    forecast_top1 = _label_to_state(forecast_row["regime_top1"])
    pop_thresh = forecast_row["pop_threshold_in"] or 0.10
    actual_pop_hit = int(actual["pcpn"] > pop_thresh)

    regime_top1_hit = (
        int(forecast_top1 == actual_state)
        if forecast_top1 is not None and actual_state is not None else None
    )
    regime_p_actual = (
        float(probs[actual_state]) if actual_state is not None else None
    )
    hi_error = (
        float(forecast_row["hi_f"]) - actual["maxt"]
        if forecast_row["hi_f"] is not None else None
    )
    lo_error = (
        float(forecast_row["lo_f"]) - actual["mint"]
        if forecast_row["lo_f"] is not None else None
    )
    pop_brier = None
    if forecast_row["pop"] is not None:
        pop_brier = float((forecast_row["pop"] - actual_pop_hit) ** 2)

    # NWS metrics from the cached comparison snapshot.
    nws_compare = json.loads(forecast_row["nws_compare_json"] or "{}")
    nws_state_label = nws_compare.get("regime")
    nws_state = _label_to_state(nws_state_label) if isinstance(nws_state_label, str) else None
    nws_regime_hit = (
        int(nws_state == actual_state)
        if nws_state is not None and actual_state is not None else None
    )
    nws_hi = nws_compare.get("hi_f")
    nws_lo = nws_compare.get("lo_f")
    nws_hi_err = float(nws_hi) - actual["maxt"] if nws_hi is not None else None
    nws_lo_err = float(nws_lo) - actual["mint"] if nws_lo is not None else None
    nws_pop_pct = nws_compare.get("pop")
    nws_pop_brier = (
        float((nws_pop_pct / 100.0 - actual_pop_hit) ** 2)
        if nws_pop_pct is not None else None
    )

    conn.execute(
        """
        INSERT INTO forecast_verification (
            forecast_id, target_date_local,
            actual_regime, actual_hi_f, actual_lo_f, actual_precip_in,
            actual_pop_hit,
            regime_top1_hit, regime_p_actual,
            hi_error_f, lo_error_f, pop_brier,
            nws_regime_hit, nws_hi_error_f, nws_lo_error_f, nws_pop_brier,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            forecast_id, target_date,
            STATE_LABELS[actual_state] if actual_state is not None else None,
            actual["maxt"], actual["mint"], actual["pcpn"], actual_pop_hit,
            regime_top1_hit, regime_p_actual,
            hi_error, lo_error, pop_brier,
            nws_regime_hit, nws_hi_err, nws_lo_err, nws_pop_brier,
            tz_utils.now_utc(),
        ),
    )


def main() -> int:
    fdb.init_db()
    artifact = load_active_artifact()
    terciles = artifact["terciles"]
    today = date.today().isoformat()
    logger.info("Verification run as of %s", today)

    conn = fdb.get_connection()
    written = 0
    skipped_no_actual = 0
    try:
        with conn:
            pending = pending_forecasts(conn, today)
            logger.info("Pending forecasts to verify: %d", len(pending))

            for pf in pending:
                actual = fetch_actual_kcos(pf["target_date_local"])
                if actual is None:
                    skipped_no_actual += 1
                    logger.info(
                        "  skip %s h=%d: actual not yet available",
                        pf["target_date_local"], pf["horizon_days"],
                    )
                    continue
                actual_state = classify_actual(actual, terciles, pf["target_date_local"])
                write_verification(
                    conn, pf["id"], pf["target_date_local"], actual, actual_state,
                    pf, terciles,
                )
                written += 1
                actual_label = STATE_LABELS[actual_state] if actual_state is not None else "?"
                regime_hit = "yes" if (
                    actual_state is not None and pf["regime_top1"] == actual_label
                ) else "no"
                hi_err = (pf["hi_f"] - actual["maxt"]) if pf["hi_f"] is not None else 0.0
                lo_err = (pf["lo_f"] - actual["mint"]) if pf["lo_f"] is not None else 0.0
                logger.info(
                    "  verified %s h=%d: actual Hi=%.0f Lo=%.0f precip=%.2f, "
                    "regime=%s, regime_hit=%s, hi_err=%+.1f, lo_err=%+.1f",
                    pf["target_date_local"], pf["horizon_days"],
                    actual["maxt"], actual["mint"], actual["pcpn"],
                    actual_label, regime_hit, hi_err, lo_err,
                )
    finally:
        conn.close()

    logger.info(
        "verification_logger done: %d written, %d skipped (no actual yet)",
        written, skipped_no_actual,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
