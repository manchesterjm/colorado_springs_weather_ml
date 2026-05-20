"""On-demand rolling-window accuracy dashboard.

Reports per-horizon ML / NWS / blend metrics over the most recent
30 / 60 / 90 days of verified forecasts. Useful for spotting drift
between retrains and for sanity-checking that the blend is still adding
value over the base ML and base NWS forecasts.

Run::

    python accuracy_dashboard.py
    python accuracy_dashboard.py --window 60
    python accuracy_dashboard.py --markdown
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import date, timedelta
from typing import Optional

from forecast_ml_pkg import db as fdb, fmt
from forecast_ml_pkg.externals import setup_utf8_stdout

WINDOWS_DEFAULT = (30, 60, 90)


def fetch_window(conn: sqlite3.Connection, start_iso: str) -> list[dict]:
    """Return joined forecast + verification rows whose target landed >= start.

    Includes ``model_version`` so the caller can group results by A/B variant.
    """
    cur = conn.execute(
        """
        SELECT pf.horizon_days, pf.model_version, pf.regime_top1, pf.regime_probs_json,
               pf.hi_f, pf.lo_f, pf.pop, pf.nws_compare_json, pf.ml_blend_json,
               fv.actual_regime, fv.regime_top1_hit, fv.regime_p_actual,
               fv.hi_error_f, fv.lo_error_f, fv.actual_pop_hit, fv.pop_brier,
               fv.nws_regime_hit, fv.nws_hi_error_f, fv.nws_lo_error_f, fv.nws_pop_brier
        FROM forecast_verification fv
        JOIN production_forecast pf ON pf.id = fv.forecast_id
        WHERE fv.target_date_local >= ?
        """,
        (start_iso,),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def variant_for(model_version: str) -> str:
    """Map a model_version string to its A/B variant tag.

    Source of truth is the trainer's suffix convention (``-nwsfeat`` for the
    NWS-as-feature variant). Anything else is treated as baseline.
    """
    return "nwsfeat" if "-nwsfeat" in (model_version or "") else "baseline"


def _safe_mean(values: list[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _safe_mean_abs(values: list[Optional[float]]) -> Optional[float]:
    cleaned = [abs(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def summarize(rows: list[dict]) -> dict[tuple[int, str], dict]:
    """Group by ``(horizon, variant)`` and compute ML / NWS metrics.

    Returns a dict keyed by ``(horizon_days, variant)``. NWS metrics duplicate
    across variants by design -- NWS is variant-invariant, so showing the same
    NWS row twice keeps the table grep-able and the delta row honest.
    """
    by_key: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (row["horizon_days"], variant_for(row["model_version"]))
        by_key.setdefault(key, []).append(row)

    out: dict[tuple[int, str], dict] = {}
    for key, group in sorted(by_key.items()):
        ml_blend_hi = []
        ml_blend_lo = []
        ml_blend_pop = []
        for r in group:
            blend_json = json.loads(r["ml_blend_json"] or "{}")
            actual_pop = r["actual_pop_hit"]
            if blend_json.get("hi_f") is not None and r["hi_error_f"] is not None:
                # Reconstruct blend error from blend Hi minus actual Hi (= hi - hi_error origin).
                # Actual Hi = forecast Hi - hi_error. So blend Hi err = blend_hi - actual.
                actual_hi = r["hi_f"] - r["hi_error_f"]
                ml_blend_hi.append(blend_json["hi_f"] - actual_hi)
            if blend_json.get("lo_f") is not None and r["lo_error_f"] is not None:
                actual_lo = r["lo_f"] - r["lo_error_f"]
                ml_blend_lo.append(blend_json["lo_f"] - actual_lo)
            if blend_json.get("pop") is not None and actual_pop is not None:
                ml_blend_pop.append((blend_json["pop"] - actual_pop) ** 2)

        out[key] = {
            "n": len(group),
            "ml": {
                "regime_acc": _safe_mean([r["regime_top1_hit"] for r in group]),
                "regime_p_actual": _safe_mean([r["regime_p_actual"] for r in group]),
                "hi_mae": _safe_mean_abs([r["hi_error_f"] for r in group]),
                "lo_mae": _safe_mean_abs([r["lo_error_f"] for r in group]),
                "hi_bias": _safe_mean([r["hi_error_f"] for r in group]),
                "lo_bias": _safe_mean([r["lo_error_f"] for r in group]),
                "pop_brier": _safe_mean([r["pop_brier"] for r in group]),
            },
            "nws": {
                "regime_acc": _safe_mean([r["nws_regime_hit"] for r in group]),
                "hi_mae": _safe_mean_abs([r["nws_hi_error_f"] for r in group]),
                "lo_mae": _safe_mean_abs([r["nws_lo_error_f"] for r in group]),
                "hi_bias": _safe_mean([r["nws_hi_error_f"] for r in group]),
                "lo_bias": _safe_mean([r["nws_lo_error_f"] for r in group]),
                "pop_brier": _safe_mean([r["nws_pop_brier"] for r in group]),
            },
            "blend": {
                "hi_mae": _safe_mean_abs(ml_blend_hi),
                "lo_mae": _safe_mean_abs(ml_blend_lo),
                "pop_brier": _safe_mean(ml_blend_pop),
            },
        }
    return out


def _horizon_variants(
    summary: dict[tuple[int, str], dict],
) -> dict[int, dict[str, dict]]:
    """Pivot the (h, variant) summary into ``{h: {variant: metrics}}``."""
    out: dict[int, dict[str, dict]] = {}
    for (h, variant), m in summary.items():
        out.setdefault(h, {})[variant] = m
    return out


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return ``a - b`` if both present, else None."""
    if a is None or b is None:
        return None
    return a - b


def render_text(window_days: int, summary: dict[tuple[int, str], dict]) -> str:
    if not summary:
        return f"\n=== {window_days}-day window: no verified forecasts yet ===\n"
    lines = [f"\n=== {window_days}-day window ==="]
    lines.append(
        f"{'h':>2} {'variant':>9}  {'n':>3}  | {'ML acc':>7} {'ML HiMAE':>8} {'ML LoMAE':>8} {'ML PoPB':>7} "
        f"| {'NWS acc':>7} {'NWSHiMAE':>8} {'NWSLoMAE':>8} {'NWS PoPB':>8} "
        f"| {'BlnHiMAE':>8} {'BlnLoMAE':>8} {'BlnPoPB':>8}"
    )
    by_h = _horizon_variants(summary)
    for h in sorted(by_h):
        variants = by_h[h]
        for variant in ("baseline", "nwsfeat"):
            m = variants.get(variant)
            if m is None:
                continue
            lines.append(
                f"{h:>2} {variant:>9}  {m['n']:>3}  "
                f"| {fmt.pct1(m['ml']['regime_acc']):>7} {fmt.num2(m['ml']['hi_mae']):>8} "
                f"{fmt.num2(m['ml']['lo_mae']):>8} {fmt.num4(m['ml']['pop_brier']):>7} "
                f"| {fmt.pct1(m['nws']['regime_acc']):>7} {fmt.num2(m['nws']['hi_mae']):>8} "
                f"{fmt.num2(m['nws']['lo_mae']):>8} {fmt.num4(m['nws']['pop_brier']):>8} "
                f"| {fmt.num2(m['blend']['hi_mae']):>8} {fmt.num2(m['blend']['lo_mae']):>8} "
                f"{fmt.num4(m['blend']['pop_brier']):>8}"
            )
        if "baseline" in variants and "nwsfeat" in variants:
            base = variants["baseline"]["ml"]
            nws = variants["nwsfeat"]["ml"]
            lines.append(
                f"{h:>2} {'Δ(nf-bl)':>9}  {'':>3}  "
                f"| {fmt.pct1_signed(_delta(nws['regime_acc'], base['regime_acc'])):>7} "
                f"{fmt.signed2(_delta(nws['hi_mae'], base['hi_mae'])):>8} "
                f"{fmt.signed2(_delta(nws['lo_mae'], base['lo_mae'])):>8} "
                f"{fmt.signed4(_delta(nws['pop_brier'], base['pop_brier'])):>7} "
                f"|"
            )
    return "\n".join(lines) + "\n"


def render_markdown(windows: dict[int, dict[tuple[int, str], dict]]) -> str:
    parts = ["# Forecast Accuracy Dashboard\n"]
    for window_days, summary in windows.items():
        parts.append(f"## {window_days}-day window\n")
        if not summary:
            parts.append("_no verified forecasts in this window_\n")
            continue
        parts.append(
            "| h | variant | n | ML acc | ML Hi MAE | ML Lo MAE | ML PoP Brier | "
            "NWS acc | NWS Hi MAE | NWS Lo MAE | NWS PoP Brier | "
            "Blend Hi MAE | Blend Lo MAE | Blend PoP Brier |"
        )
        parts.append(
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )
        by_h = _horizon_variants(summary)
        for h in sorted(by_h):
            variants = by_h[h]
            for variant in ("baseline", "nwsfeat"):
                m = variants.get(variant)
                if m is None:
                    continue
                parts.append(
                    "| "
                    f"{h} | {variant} | {m['n']} | {fmt.pct1(m['ml']['regime_acc'])} | "
                    f"{fmt.num2(m['ml']['hi_mae'])} | {fmt.num2(m['ml']['lo_mae'])} | "
                    f"{fmt.num4(m['ml']['pop_brier'])} | "
                    f"{fmt.pct1(m['nws']['regime_acc'])} | {fmt.num2(m['nws']['hi_mae'])} | "
                    f"{fmt.num2(m['nws']['lo_mae'])} | {fmt.num4(m['nws']['pop_brier'])} | "
                    f"{fmt.num2(m['blend']['hi_mae'])} | {fmt.num2(m['blend']['lo_mae'])} | "
                    f"{fmt.num4(m['blend']['pop_brier'])} |"
                )
            if "baseline" in variants and "nwsfeat" in variants:
                base = variants["baseline"]["ml"]
                nws = variants["nwsfeat"]["ml"]
                parts.append(
                    "| "
                    f"{h} | Δ(nf−bl) | -- | "
                    f"{fmt.pct1_signed(_delta(nws['regime_acc'], base['regime_acc']))} | "
                    f"{fmt.signed2(_delta(nws['hi_mae'], base['hi_mae']))} | "
                    f"{fmt.signed2(_delta(nws['lo_mae'], base['lo_mae']))} | "
                    f"{fmt.signed4(_delta(nws['pop_brier'], base['pop_brier']))} | "
                    "-- | -- | -- | -- | -- | -- | -- |"
                )
        parts.append("")
    return "\n".join(parts)


def main() -> int:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description="Forecast accuracy dashboard")
    parser.add_argument(
        "--windows", default="30,60,90",
        help="Comma-separated rolling windows (days). Default 30,60,90",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Emit markdown instead of plain text",
    )
    args = parser.parse_args()

    windows = tuple(int(x) for x in args.windows.split(","))
    today = date.today().isoformat()

    conn = fdb.get_connection()
    try:
        per_window: dict[int, dict[int, dict]] = {}
        for w in windows:
            start = (date.fromisoformat(today) - timedelta(days=w)).isoformat()
            rows = fetch_window(conn, start)
            per_window[w] = summarize(rows)
    finally:
        conn.close()

    if args.markdown:
        print(render_markdown(per_window))
    else:
        for w, summary in per_window.items():
            print(render_text(w, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
