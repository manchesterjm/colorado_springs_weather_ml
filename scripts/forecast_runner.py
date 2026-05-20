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
import pickle
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

from forecast_ml_pkg import blend, db as fdb, live_features, output
from forecast_ml_pkg.externals import db_utils, setup_utf8_stdout, tz_utils

from weather_regime_pkg import STATE_LABELS, fetch_latest_nws_snapshot
from weather_regime_pkg.nws import (
    estimate_p_heavy_given_wet, nws_to_probs,
)


REPORTS_DIR = fdb.PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_active_models() -> list[tuple[dict, str, str, str]]:
    """Read every ``is_active=1`` row in ``model_runs``, load each artifact.

    Returns a list of ``(artifact, model_version, run_id, variant)`` ordered by
    variant (``baseline`` first, then ``nwsfeat`` so output is stable).
    Raises if no active models exist.
    """
    conn = fdb.get_connection()
    try:
        rows = conn.execute(
            "SELECT run_id, model_version, artifact_path, variant FROM model_runs "
            "WHERE is_active = 1 ORDER BY variant"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise RuntimeError(
            "No active models in model_runs. Run train_extended.py first."
        )
    out: list[tuple[dict, str, str, str]] = []
    for run_id, model_version, artifact_path, variant in rows:
        with open(artifact_path, "rb") as fh:
            artifact = pickle.load(fh)
        out.append((artifact, model_version, run_id, variant))
    return out


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
    conn = sqlite3.connect(db_utils.DB_PATH)
    try:
        fetch_time, by_date = fetch_latest_nws_snapshot(conn, as_of_date)
    finally:
        conn.close()
    return fetch_time, {d: by_date[d] for d in target_dates if d in by_date}


def nws_summary_for_date(
    nws_day: dict,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Pull (hi, lo, pop_pct, text) from one fetch_latest_nws_snapshot row."""
    hi = max(nws_day["highs"]) if nws_day.get("highs") else None
    lo = min(nws_day["lows"]) if nws_day.get("lows") else None
    pop = max(nws_day["pops"]) if nws_day.get("pops") else None
    text = nws_day["shorts"][0] if nws_day.get("shorts") else None
    return hi, lo, pop, text


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _build_anchor_X(artifact: dict) -> tuple[np.ndarray, list[dict], int]:
    """Build the model-ready X for one artifact.

    Returns ``(X, rows, anchor_idx)``. If the artifact is a nwsfeat variant,
    the 28 NWS columns are appended to X (with archive -> live MEX -> impute
    fallback chain).
    """
    rows, neighbor_lookups = live_features.load_classified_data(artifact)
    anchor_idx = live_features.find_anchor_index(rows)
    if anchor_idx is None:
        raise RuntimeError("No suitable anchor day found in ACIS data.")
    X = live_features.build_anchor_features(
        rows, anchor_idx, artifact["doy_clim_temp"], neighbor_lookups,
    )
    if artifact.get("use_nws_features"):
        anchor_date = rows[anchor_idx]["date"]
        nws_row, diag = live_features.fetch_nws_features_for_anchor(
            anchor_date, artifact,
        )
        print(
            f"  NWS features: source={diag['source']} "
            f"available={diag['n_available']}/7 imputed={diag['n_imputed']}"
        )
        X = np.hstack([X, nws_row])
    return X, rows, anchor_idx


def _predict_one_model(
    artifact: dict,
    model_version: str,
    run_id: str,
    nws_by_date: dict[str, dict],
    nws_fetch_time: Optional[str],
    p_heavy_given_wet: float,
) -> output.ForecastResult:
    """Run one model end-to-end and return its ForecastResult."""
    X, rows, anchor_idx = _build_anchor_X(artifact)
    anchor = rows[anchor_idx]
    anchor_date = anchor["date"]
    anchor_state_label = STATE_LABELS[anchor["state"]]
    anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d").date()
    horizons = list(artifact["horizons"])
    target_dates = {h: (anchor_dt + timedelta(days=h)).isoformat() for h in horizons}

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


def run_forecast() -> list[output.ForecastResult]:
    """Generate one ``ForecastResult`` per active model variant.

    Each variant produces its own predictions, blend with NWS, and ForecastResult.
    The list is ordered ``baseline`` first, ``nwsfeat`` second (matching the
    ordering from ``load_active_models``).
    """
    models = load_active_models()
    print(f"Active models: {len(models)}")
    for artifact, version, run_id, variant in models:
        print(
            f"  [{variant:8s}] {version} (run {run_id}) | "
            f"features={len(artifact['feature_names'])} "
            f"horizons={tuple(artifact['horizons'])}"
        )

    # Compute the anchor date once using the first artifact (both share data).
    primary_artifact = models[0][0]
    primary_rows, _ = live_features.load_classified_data(primary_artifact)
    anchor_idx = live_features.find_anchor_index(primary_rows)
    if anchor_idx is None:
        raise RuntimeError("No suitable anchor day found in ACIS data.")
    anchor_dt = datetime.strptime(primary_rows[anchor_idx]["date"], "%Y-%m-%d").date()
    horizons = list(primary_artifact["horizons"])
    target_dates = [(anchor_dt + timedelta(days=h)).isoformat() for h in horizons]

    # One NWS fetch for all variants.
    today_utc = datetime.now(timezone.utc).date().isoformat()
    nws_fetch_time, nws_by_date = fetch_nws_for_dates(target_dates, today_utc)
    if not nws_by_date:
        conn = sqlite3.connect(db_utils.DB_PATH)
        try:
            cur = conn.execute(
                "SELECT MAX(substr(fetch_time, 1, 10)) FROM forecast_snapshots"
            )
            latest = cur.fetchone()[0]
        finally:
            conn.close()
        if latest:
            nws_fetch_time, nws_by_date = fetch_nws_for_dates(target_dates, latest)

    # One p_heavy estimate using the primary (baseline) artifact's training window.
    train_rows = [
        r for r in primary_rows if r["date"] <= primary_artifact["trained_through"]
    ]
    p_heavy_given_wet = estimate_p_heavy_given_wet(train_rows)

    results: list[output.ForecastResult] = []
    for artifact, version, run_id, variant in models:
        print(f"\n--- predicting with {variant} ---")
        result = _predict_one_model(
            artifact, version, run_id,
            nws_by_date, nws_fetch_time, p_heavy_given_wet,
        )
        results.append(result)
    return results


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
    results = run_forecast()
    print()
    for result in results:
        print(output.render_text(result))

    if args.print_only or args.no_markdown:
        md_path = None
    else:
        # Write one combined markdown per issue date; each variant gets its own
        # section so blame-by-variant comparison is one scroll away.
        md_path = output.write_multi_variant_report(results, REPORTS_DIR)
        print(f"Markdown report: {md_path}")

    if args.print_only or args.no_db:
        inserted = 0
    else:
        inserted = sum(output.write_to_db(r) for r in results)
        total_horizons = sum(len(r.horizons) for r in results)
        print(f"Wrote {inserted}/{total_horizons} production_forecast rows "
              f"({len(results)} variants x {len(results[0].horizons)} horizons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
