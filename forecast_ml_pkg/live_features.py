"""Build the anchor-day feature vector for live forecasting.

Wraps ``weather_forecast_now.py``'s anchor logic into a reusable function
the production runner can call. Loads ACIS daily for KCOS and neighbors,
classifies states using the *trained* terciles, and emits a single
feature row matching the artifact's ``feature_names``.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import date, datetime
from typing import Optional

import numpy as np

from weather_regime_pkg import (avg_temp, classify, doy, fetch_acis)

from forecast_ml_pkg import db as fdb, mex_mos, nws_features as nwsf
from forecast_ml_pkg.externals import tz_utils

logger = logging.getLogger(__name__)


def _advance_anchor_with_metar(
    rows: list[dict],
    station_id: str,
    conn: sqlite3.Connection,
) -> list[dict]:
    """Splice fresh METAR-derived daily rows in so a stale ACIS anchor advances.

    ACIS daily climate publishes ~1-2 days late, which would otherwise pin the
    forecast anchor two days back -- so "h=1" lands on a date already in the past.
    The ``daily_summary`` table (METAR-derived, refreshed nightly) usually already
    has those missing days; we splice each *complete* past day ACIS still lacks,
    converted to the ACIS row shape and units (degF).

    Only the anchor day itself ends up METAR-sourced -- its lag days stay ACIS --
    so the known ~1 degF METAR-vs-ACIS sampling skew touches a single recent
    feature, a small price for advancing the anchor a full day.
    """
    last_complete = max(
        (r["date"] for r in rows
         if r["maxt"] is not None and r["mint"] is not None),
        default=None,
    )
    if last_complete is None:
        return rows
    today = date.today().isoformat()
    cur = conn.execute(
        "SELECT date_local, tmax_c, tmin_c, precip_in FROM daily_summary "
        "WHERE station_id=? AND date_local>? AND date_local<? "
        "AND tmax_c IS NOT NULL AND tmin_c IS NOT NULL AND hours_observed>=20 "
        "ORDER BY date_local",
        (station_id, last_complete, today),
    )
    bridge = [
        {"date": d,
         "maxt": round(tmax_c * 9 / 5 + 32, 1),
         "mint": round(tmin_c * 9 / 5 + 32, 1),
         "pcpn": pcpn if pcpn is not None else 0.0}
        for d, tmax_c, tmin_c, pcpn in cur.fetchall()
    ]
    if not bridge:
        return rows
    bridged_dates = {b["date"] for b in bridge}
    merged = [r for r in rows if r["date"] not in bridged_dates]
    merged.extend(bridge)
    merged.sort(key=lambda r: r["date"])
    logger.info(
        "Anchor bridge %s: ACIS through %s + METAR daily %s",
        station_id, last_complete, sorted(bridged_dates),
    )
    return merged


def load_classified_data(
    artifact: dict,
) -> tuple[list[dict], dict[str, dict]]:
    """Load KCOS + neighbors, classify in place using artifact's terciles.

    Each station's ACIS history is first bridged with fresh METAR-derived daily
    rows (see ``_advance_anchor_with_metar``) so the anchor reflects the most
    recent complete day rather than ACIS's ~2-day-late publication.

    Returns ``(kcos_rows, neighbor_lookups)`` where ``neighbor_lookups`` is
    ``{station_id: {date: row}}``.
    """
    conn = fdb.get_connection()
    try:
        rows = _advance_anchor_with_metar(fetch_acis("KCOS"), "KCOS", conn)
        classify(rows, artifact["terciles"])

        neighbor_lookups: dict[str, dict] = {}
        for sid in artifact.get("neighbor_stations", []):
            nrows = _advance_anchor_with_metar(fetch_acis(sid), sid, conn)
            n_terc = artifact["neighbor_terciles"].get(sid)
            if n_terc:
                classify(nrows, n_terc)
            neighbor_lookups[sid] = {r["date"]: r for r in nrows}
    finally:
        conn.close()

    return rows, neighbor_lookups


def find_anchor_index(rows: list[dict]) -> Optional[int]:
    """Last row with a complete contiguous 7-day classified history."""
    for i in range(len(rows) - 1, 5, -1):
        win = rows[i - 6:i + 1]
        if any(r["state"] is None for r in win):
            continue
        contiguous = all(
            (datetime.strptime(win[k]["date"], "%Y-%m-%d")
             - datetime.strptime(win[k - 1]["date"], "%Y-%m-%d")).days == 1
            for k in range(1, len(win))
        )
        if contiguous:
            return i
    return None


def build_anchor_features(
    rows: list[dict],
    anchor_idx: int,
    doy_clim_temp: np.ndarray,
    neighbor_lookups: dict[str, dict],
) -> np.ndarray:
    """Emit a (1, n_features) feature row for the given anchor index.

    Layout matches ``weather_regime_pkg.features.build_features``: KCOS
    base block, then 3 features per neighbor in the same order the
    artifact saved.
    """
    r_t = rows[anchor_idx]
    r_tm1 = rows[anchor_idx - 1]
    r_tm2 = rows[anchor_idx - 2]
    d = doy(r_t["date"])
    avg_t = avg_temp(r_t)
    anom = (
        avg_t - doy_clim_temp[d]
        if not math.isnan(doy_clim_temp[d]) else 0.0
    )
    feat: list[float] = [
        float(r_t["state"]),
        float(r_tm1["state"]),
        float(r_tm2["state"]),
        r_t["maxt"], r_t["mint"], avg_t,
        r_tm1["maxt"], r_tm1["mint"],
        r_t["pcpn"], r_tm1["pcpn"], r_tm2["pcpn"],
        sum(rows[j]["pcpn"] for j in range(anchor_idx - 6, anchor_idx + 1)),
        anom,
        math.sin(2 * math.pi * d / 365.25),
        math.cos(2 * math.pi * d / 365.25),
        float(int(r_t["date"][5:7])),
    ]
    for lookup in neighbor_lookups.values():
        nrow = lookup.get(r_t["date"])
        if nrow is None:
            feat.extend([np.nan, np.nan, np.nan])
            continue
        s = nrow.get("state")
        feat.extend([
            float(s) if s is not None else np.nan,
            nrow["maxt"] if nrow["maxt"] is not None else np.nan,
            nrow["pcpn"] if nrow["pcpn"] is not None else np.nan,
        ])
    return np.array([feat], dtype=float)


def fetch_nws_features_for_anchor(
    anchor_date_str: str,
    artifact: dict,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the (1, 28) NWS feature row for live inference, plus diagnostics.

    Strategy:
      1. Try to read from the local archive (``nws_daily_forecast``).
      2. If horizons are missing, attempt a live MEX fetch from IEM and upsert
         the result into the archive before retrying the lookup.
      3. Anything still missing falls back to DOY-mean climatology stored in
         the artifact and flips the corresponding ``nws_available_h*`` indicator
         to 0.

    Diagnostics returned: ``{"source": "archive"|"live"|"impute"|"mixed",
    "n_available": int, "n_imputed": int}`` so the runner can log + warn.
    """
    hi_clim = artifact.get("nws_hi_clim")
    lo_clim = artifact.get("nws_lo_clim")
    if hi_clim is None or lo_clim is None:
        raise ValueError(
            "Artifact is missing nws_hi_clim/nws_lo_clim; not a nwsfeat model."
        )

    lookup = nwsf.load_mex_lookup(
        start_anchor=anchor_date_str, end_anchor=anchor_date_str,
    )
    available = sum(
        1 for h in nwsf.HORIZONS if (anchor_date_str, h) in lookup
    )
    source = "archive"

    if available < len(nwsf.HORIZONS):
        # Try a live IEM fetch for anchor 12Z. Safety net only.
        runtime = f"{anchor_date_str}T12:00:00Z"
        try:
            raw = mex_mos.fetch_bulletin("KCOS", runtime)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning("live MEX fetch failed for %s: %s", runtime, exc)
            raw = None
        if raw:
            norm = mex_mos.normalize_rows("KCOS", raw)
            entries = mex_mos.derive_daily_entries(norm)
            conn = fdb.get_connection()
            try:
                fetched_at = tz_utils.now_utc()
                for r in norm:
                    conn.execute(
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
                for e in entries:
                    if e.horizon_days < 1:
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO nws_daily_forecast (
                            station_id, issue_date_local, target_date_local,
                            horizon_days, nws_hi_f, nws_lo_f,
                            nws_pop_day, nws_pop_night, source, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "KCOS", e.issue_date_local, e.target_date_local,
                            e.horizon_days, e.nws_hi_f, e.nws_lo_f,
                            e.nws_pop_day, e.nws_pop_night, "MEX_MOS", fetched_at,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
            lookup = nwsf.load_mex_lookup(
                start_anchor=anchor_date_str, end_anchor=anchor_date_str,
            )
            new_available = sum(
                1 for h in nwsf.HORIZONS if (anchor_date_str, h) in lookup
            )
            if new_available > available:
                source = "live" if available == 0 else "mixed"
                available = new_available

    row = nwsf.build_nws_feature_row(anchor_date_str, lookup, hi_clim, lo_clim)
    n_imputed = len(nwsf.HORIZONS) - available
    if available == 0:
        source = "impute"
        logger.warning(
            "No MEX data for anchor %s; imputing all 7 horizons from climatology.",
            anchor_date_str,
        )
    elif n_imputed > 0:
        logger.warning(
            "Partial MEX coverage for anchor %s: %d/%d horizons imputed.",
            anchor_date_str, n_imputed, len(nwsf.HORIZONS),
        )

    diag = {
        "source": source,
        "n_available": available,
        "n_imputed": n_imputed,
    }
    return row.reshape(1, -1), diag
