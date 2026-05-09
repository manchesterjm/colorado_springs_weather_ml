"""Daily forecast runner. Scheduled at 06:30 MST.

Loads the active model from ``model_runs``, builds the live anchor-day
features, predicts regime + Hi + Lo + PoP for h=1..7, fetches the
latest NWS forecast for those target dates, blends the two, and writes:

    * production_forecast (one row per horizon)
    * nws_compare_cache  (one row per horizon if NWS data available)
    * markdown report at reports/YYYY-MM-DD.md
    * stdout summary

Reuses the main weather DB at ``D:\\Scripts\\weather_data\\weather.db`` for
NWS forecast snapshots; the new ``forecast_ml.db`` is the only DB this
script writes to.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg import blend, db as fdb, live_features, output  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.io import setup_utf8_stdout, tz_utils  # noqa: E402  pylint: disable=wrong-import-position

from weather_regime_pkg import STATE_LABELS, doy, fetch_latest_nws_snapshot  # noqa: E402  pylint: disable=wrong-import-position
from weather_regime_pkg.nws import (  # noqa: E402  pylint: disable=wrong-import-position
    estimate_p_heavy_given_wet, nws_to_probs,
)


WEATHER_DB = Path(r"D:\Scripts\weather_data\weather.db") if sys.platform == "win32" \
    else Path("/mnt/d/Scripts/weather_data/weather.db")
REPORTS_DIR = fdb.PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_active_model() -> tuple[dict, str, str]:
    """Read the most recent ``is_active=1`` row in ``model_runs`` and load.

    Returns ``(artifact_dict, model_version, run_id)``.
    """
    conn = fdb.get_connection()
    try:
        row = conn.execute(
            "SELECT run_id, model_version, artifact_path FROM model_runs "
            "WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(
            "No active model in model_runs. Run train_extended.py first."
        )
    run_id, model_version, artifact_path = row
    with open(artifact_path, "rb") as fh:
        artifact = pickle.load(fh)
    return artifact, model_version, run_id


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_regime(artifact: dict, horizon: int, X: np.ndarray) -> np.ndarray:
    """9-state probability vector from the calibrated regime classifier."""
    clf = artifact["regime"][horizon]
    raw = clf.predict_proba(X)[0]
    out = np.zeros(9)
    for k, c in enumerate(clf.classes_):
        out[int(c)] = raw[k]
    return out


def predict_hi_lo(artifact: dict, horizon: int, head: str, X: np.ndarray) -> float:
    return float(artifact[head][horizon].predict(X)[0])


def predict_pop(artifact: dict, horizon: int, X: np.ndarray) -> float:
    clf = artifact["pop"][horizon]
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return 0.0


# ---------------------------------------------------------------------------
# NWS lookup
# ---------------------------------------------------------------------------

def fetch_nws_for_dates(
    target_dates: list[str], as_of_date: str
) -> tuple[Optional[str], dict[str, dict]]:
    """Latest NWS snapshot keyed by local target date string."""
    conn = sqlite3.connect(WEATHER_DB)
    try:
        fetch_time, by_date = fetch_latest_nws_snapshot(conn, as_of_date)
    finally:
        conn.close()
    return fetch_time, {d: by_date[d] for d in target_dates if d in by_date}


def nws_summary_for_date(nws_day: dict) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Pull (hi, lo, pop_pct, text) from one fetch_latest_nws_snapshot row."""
    hi = max(nws_day["highs"]) if nws_day.get("highs") else None
    lo = min(nws_day["lows"]) if nws_day.get("lows") else None
    pop = max(nws_day["pops"]) if nws_day.get("pops") else None
    text = nws_day["shorts"][0] if nws_day.get("shorts") else None
    return hi, lo, pop, text


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_forecast() -> output.ForecastResult:  # pylint: disable=too-many-locals
    artifact, model_version, run_id = load_active_model()
    print(f"Active model: {model_version} (run {run_id})")
    print(f"  Trained through {artifact['trained_through']}, "
          f"horizons {artifact['horizons']}, "
          f"feature count {len(artifact['feature_names'])}")

    rows, neighbor_lookups = live_features.load_classified_data(artifact)
    anchor_idx = live_features.find_anchor_index(rows)
    if anchor_idx is None:
        raise RuntimeError("No suitable anchor day found in ACIS data.")

    anchor = rows[anchor_idx]
    anchor_date = anchor["date"]
    anchor_state_label = STATE_LABELS[anchor["state"]]
    print(f"\nAnchor day: {anchor_date} state={anchor_state_label} "
          f"Hi={anchor['maxt']} Lo={anchor['mint']} "
          f"precip={anchor['pcpn']}")

    X = live_features.build_anchor_features(
        rows, anchor_idx, artifact["doy_clim_temp"], neighbor_lookups,
    )

    # Compute target dates for each horizon (anchor_date + h)
    anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d").date()
    horizons = list(artifact["horizons"])
    target_dates = {h: (anchor_dt + timedelta(days=h)).isoformat() for h in horizons}

    # NWS lookup -- as_of_date is today (UTC). Query for "today" snapshot.
    today_utc = datetime.now(timezone.utc).date().isoformat()
    nws_fetch_time, nws_by_date = fetch_nws_for_dates(
        list(target_dates.values()), today_utc,
    )
    if not nws_by_date:
        # Fallback to most recent available snapshot (no date filter).
        conn = sqlite3.connect(WEATHER_DB)
        try:
            cur = conn.execute(
                "SELECT MAX(substr(fetch_time, 1, 10)) FROM forecast_snapshots"
            )
            latest = cur.fetchone()[0]
        finally:
            conn.close()
        if latest:
            nws_fetch_time, nws_by_date = fetch_nws_for_dates(
                list(target_dates.values()), latest,
            )

    # Estimate P(heavy | wet) climatologically from the same KCOS rows used
    # to train (anchor through trained_through window).
    train_rows = [r for r in rows if r["date"] <= artifact["trained_through"]]
    p_heavy_given_wet = estimate_p_heavy_given_wet(train_rows)

    # Per-horizon predictions
    horizon_payloads: list[output.HorizonForecast] = []
    for h in horizons:
        td = target_dates[h]
        weekday = (anchor_dt + timedelta(days=h)).strftime("%A")

        regime_probs = predict_regime(artifact, h, X)
        hi_f = predict_hi_lo(artifact, h, "hi", X)
        lo_f = predict_hi_lo(artifact, h, "lo", X)
        pop_v = predict_pop(artifact, h, X)

        nws_day = nws_by_date.get(td)
        nws_hi = nws_lo = nws_pop = None
        nws_text = None
        nws_regime_state: Optional[int] = None
        if nws_day:
            nws_hi, nws_lo, nws_pop, nws_text = nws_summary_for_date(nws_day)
            nws_state_probs = nws_to_probs(
                nws_day, artifact["terciles"], td, p_heavy_given_wet,
            )
            if nws_state_probs is not None:
                nws_regime_state = int(np.argmax(nws_state_probs))

        # Blend
        w_ml = blend.get_default_w_ml(h)
        b_hi = blend.blend_temperature(hi_f, nws_hi, w_ml)
        b_lo = blend.blend_temperature(lo_f, nws_lo, w_ml)
        b_pop = blend.blend_pop(pop_v, nws_pop, w_ml)

        top1 = int(np.argmax(regime_probs))
        horizon_payloads.append(output.HorizonForecast(
            horizon_days=h,
            target_date_local=td,
            weekday=weekday,
            regime_probs=[float(p) for p in regime_probs],
            regime_top1=top1,
            regime_top1_label=STATE_LABELS[top1],
            hi_f_ml=hi_f,
            lo_f_ml=lo_f,
            pop_ml=pop_v,
            nws_hi_f=nws_hi,
            nws_lo_f=nws_lo,
            nws_pop=nws_pop,
            nws_regime=nws_regime_state,
            nws_text=nws_text,
            blend_hi_f=b_hi,
            blend_lo_f=b_lo,
            blend_pop=b_pop,
            w_ml=w_ml,
        ))

    return output.ForecastResult(
        issue_time=tz_utils.now_utc(),
        issue_date_local=date.today().isoformat(),
        anchor_date_local=anchor_date,
        anchor_state_label=anchor_state_label,
        anchor_hi_f=anchor.get("maxt"),
        anchor_lo_f=anchor.get("mint"),
        anchor_precip_in=anchor.get("pcpn"),
        model_version=model_version,
        model_run_id=run_id,
        nws_fetch_time=nws_fetch_time,
        horizons=horizon_payloads,
    )


def main() -> int:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description="Daily forecast runner")
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip writing to production_forecast / nws_compare_cache",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Skip writing the markdown report",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Print to stdout only (implies --no-db --no-markdown)",
    )
    args = parser.parse_args()

    fdb.init_db()
    result = run_forecast()
    print()
    print(output.render_text(result))

    if args.print_only or args.no_markdown:
        md_path = None
    else:
        md_path = output.write_markdown_report(result, REPORTS_DIR)
        print(f"Markdown report: {md_path}")

    if args.print_only or args.no_db:
        inserted = 0
    else:
        inserted = output.write_to_db(result)
        print(f"Wrote {inserted}/{len(result.horizons)} production_forecast rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
