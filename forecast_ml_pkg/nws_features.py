"""NWS-as-feature helpers for Phase 7 (nwsfeat) model variant.

Provides the lookup, climatology imputation, and per-anchor feature row
construction needed to augment the existing autoregressive feature matrix
with 28 NWS columns (Hi/Lo/PoP at horizons 1..7, plus availability indicators).

Convention for pairing anchor with MEX runtime:

    For anchor date A, use the MEX bulletin issued at ``A 12:00Z`` -- that is,
    the morning of the anchor day itself. This is the freshest MEX that still
    covers all 7 forward horizons (a bulletin from runtime A+1 12Z has its
    earliest ftime at A+2 12Z, so h=1 is not in the data).

    Horizon from MEX runtime equals horizon from anchor (target = A+h,
    ftime = A+h 12Z for the morning low or A+h+1 00Z for the daytime high).

If no MEX bulletin exists in ``nws_daily_forecast`` for that (anchor, horizon),
we impute with DOY-mean climatology computed from the training-period actuals
and flag the row in the indicator column.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from forecast_ml_pkg import db as fdb
from weather_regime_pkg.state_classifier import doy

logger = logging.getLogger(__name__)

HORIZONS = (1, 2, 3, 4, 5, 6, 7)
POP_IMPUTE_PCT = 15.0  # neutral PoP when imputing; KCOS spring/summer baseline


def nws_feature_names() -> list[str]:
    """Return the 28 NWS feature column names in canonical insertion order.

    Order matters: ``build_anchor_features`` and the trainer both rely on this
    list to slot values into the X matrix. The four groups appear in this
    sequence so blame-by-feature in feature-importance plots groups cleanly.
    """
    names: list[str] = []
    for h in HORIZONS:
        names.append(f"nws_hi_f_h{h}")
    for h in HORIZONS:
        names.append(f"nws_lo_f_h{h}")
    for h in HORIZONS:
        names.append(f"nws_pop_h{h}")
    for h in HORIZONS:
        names.append(f"nws_available_h{h}")
    return names


# Shape sanity (kept as a module-level constant other modules can import).
NWS_FEATURE_COUNT = 4 * len(HORIZONS)
assert NWS_FEATURE_COUNT == 28


def doy_hi_lo_climatology(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(hi_clim, lo_clim)`` arrays of shape (367,) keyed by DOY 1..366.

    ``rows`` is a list of ACIS-style records with ``date``, ``maxt``, ``mint``
    fields. Index 0 is always NaN. DOYs with no training observations get NaN
    and the imputer falls back to the global mean.
    """
    hi_sum = np.zeros(367)
    hi_count = np.zeros(367, dtype=int)
    lo_sum = np.zeros(367)
    lo_count = np.zeros(367, dtype=int)
    for r in rows:
        if r.get("maxt") is None or r.get("mint") is None:
            continue
        d = doy(r["date"])
        hi_sum[d] += r["maxt"]
        hi_count[d] += 1
        lo_sum[d] += r["mint"]
        lo_count[d] += 1
    safe_hi = np.where(hi_count == 0, 1, hi_count)
    safe_lo = np.where(lo_count == 0, 1, lo_count)
    hi_clim = np.where(hi_count > 0, hi_sum / safe_hi, np.nan)
    lo_clim = np.where(lo_count > 0, lo_sum / safe_lo, np.nan)
    return hi_clim, lo_clim


def load_mex_lookup(
    start_anchor: Optional[str] = None,
    end_anchor: Optional[str] = None,
    station: str = "KCOS",
) -> dict[tuple[str, int], dict]:
    """Bulk-load MEX_MOS daily forecasts keyed by ``(issue_date_local, horizon)``.

    Inclusive bounds on ``issue_date_local``. With both bounds ``None`` the
    full archive is loaded (~15k rows; under 100 ms).
    """
    sql = (
        "SELECT issue_date_local, horizon_days, nws_hi_f, nws_lo_f, nws_pop_day "
        "FROM nws_daily_forecast "
        "WHERE source = 'MEX_MOS' AND station_id = ? AND horizon_days BETWEEN 1 AND 7"
    )
    params: list[object] = [station]
    if start_anchor:
        sql += " AND issue_date_local >= ?"
        params.append(start_anchor)
    if end_anchor:
        sql += " AND issue_date_local <= ?"
        params.append(end_anchor)
    conn = fdb.get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    out: dict[tuple[str, int], dict] = {}
    for issue, h, hi, lo, pop in rows:
        out[(issue, int(h))] = {"hi": hi, "lo": lo, "pop": pop}
    return out


def _impute_value(date_str: str, clim: np.ndarray) -> float:
    """Return DOY-mean from ``clim`` for the given date; fall back to global mean."""
    d = doy(date_str)
    v = float(clim[d])
    if math.isnan(v):
        # Global mean as last-resort fallback.
        valid = clim[~np.isnan(clim)]
        return float(np.mean(valid)) if len(valid) else 0.0
    return v


def build_nws_feature_row(
    anchor_date_str: str,
    lookup: dict[tuple[str, int], dict],
    hi_clim: np.ndarray,
    lo_clim: np.ndarray,
) -> np.ndarray:
    """Emit the 28-column NWS feature row for one anchor date.

    Layout matches ``nws_feature_names()``: 7 Hi, then 7 Lo, then 7 PoP, then
    7 availability flags.

    The MEX runtime for this anchor is ``anchor_date 12:00Z`` and horizons
    from runtime equal horizons from anchor (see module docstring).
    """
    anchor = datetime.strptime(anchor_date_str, "%Y-%m-%d").date()
    hi_vals: list[float] = []
    lo_vals: list[float] = []
    pop_vals: list[float] = []
    avail: list[float] = []

    for h in HORIZONS:
        target = anchor + timedelta(days=h)
        target_str = target.isoformat()
        row = lookup.get((anchor_date_str, h))
        if row is None or row.get("hi") is None:
            hi_vals.append(_impute_value(target_str, hi_clim))
            lo_vals.append(_impute_value(target_str, lo_clim))
            pop_vals.append(POP_IMPUTE_PCT)
            avail.append(0.0)
            continue
        hi_vals.append(float(row["hi"]))
        lo_vals.append(
            float(row["lo"]) if row.get("lo") is not None
            else _impute_value(target_str, lo_clim)
        )
        pop_vals.append(
            float(row["pop"]) if row.get("pop") is not None else POP_IMPUTE_PCT
        )
        avail.append(1.0)

    return np.array(hi_vals + lo_vals + pop_vals + avail, dtype=float)


def build_nws_feature_matrix(
    rows: list[dict],
    valid_idx: np.ndarray,
    lookup: dict[tuple[str, int], dict],
    hi_clim: np.ndarray,
    lo_clim: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Build an (n_valid, 28) matrix aligned to ``valid_idx``.

    Returns ``(matrix, n_real, n_imputed_any)`` where ``n_real`` is the count
    of anchor dates with at least one real (non-imputed) MEX horizon and
    ``n_imputed_any`` counts rows with any imputation. These let the caller
    log coverage in retraining logs.
    """
    out = np.zeros((len(valid_idx), NWS_FEATURE_COUNT), dtype=float)
    n_real = 0
    n_imputed_any = 0
    for j, i in enumerate(valid_idx):
        anchor_date = rows[i]["date"]
        row_features = build_nws_feature_row(anchor_date, lookup, hi_clim, lo_clim)
        out[j] = row_features
        # Availability slice is the last 7 columns.
        avail_slice = row_features[-len(HORIZONS):]
        if float(avail_slice.sum()) > 0.0:
            n_real += 1
        if float(avail_slice.sum()) < float(len(HORIZONS)):
            n_imputed_any += 1
    return out, n_real, n_imputed_any
