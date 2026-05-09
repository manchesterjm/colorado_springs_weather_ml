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
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg import db as fdb  # noqa: E402  pylint: disable=wrong-import-position

WINDOWS_DEFAULT = (30, 60, 90)


def fetch_window(conn: sqlite3.Connection, start_iso: str) -> list[dict]:
    """Return joined forecast + verification rows whose target landed >= start."""
    cur = conn.execute(
        """
        SELECT pf.horizon_days, pf.regime_top1, pf.regime_probs_json,
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


def summarize(rows: list[dict]) -> dict[int, dict]:
    """Group by horizon and compute ML / NWS metrics."""
    by_h: dict[int, list[dict]] = {}
    for row in rows:
        by_h.setdefault(row["horizon_days"], []).append(row)

    out: dict[int, dict] = {}
    for h, group in sorted(by_h.items()):
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

        out[h] = {
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


def render_text(window_days: int, summary: dict[int, dict]) -> str:
    if not summary:
        return f"\n=== {window_days}-day window: no verified forecasts yet ===\n"
    lines = [f"\n=== {window_days}-day window ==="]
    lines.append(
        f"{'h':>2}  {'n':>3}  | {'ML acc':>7} {'ML Hi MAE':>9} {'ML Lo MAE':>9} {'ML PoPB':>7} "
        f"| {'NWS acc':>7} {'NWS HiMAE':>9} {'NWS LoMAE':>9} {'NWS PoPB':>8} "
        f"| {'BlnHiMAE':>8} {'BlnLoMAE':>8} {'BlnPoPB':>8}"
    )
    for h, m in sorted(summary.items()):
        lines.append(
            f"{h:>2}  {m['n']:>3}  "
            f"| {_fmt_pct(m['ml']['regime_acc']):>7} {_fmt_f(m['ml']['hi_mae']):>9} "
            f"{_fmt_f(m['ml']['lo_mae']):>9} {_fmt_f4(m['ml']['pop_brier']):>7} "
            f"| {_fmt_pct(m['nws']['regime_acc']):>7} {_fmt_f(m['nws']['hi_mae']):>9} "
            f"{_fmt_f(m['nws']['lo_mae']):>9} {_fmt_f4(m['nws']['pop_brier']):>8} "
            f"| {_fmt_f(m['blend']['hi_mae']):>8} {_fmt_f(m['blend']['lo_mae']):>8} "
            f"{_fmt_f4(m['blend']['pop_brier']):>8}"
        )
    return "\n".join(lines) + "\n"


def render_markdown(windows: dict[int, dict[int, dict]]) -> str:
    parts = ["# Forecast Accuracy Dashboard\n"]
    for window_days, summary in windows.items():
        parts.append(f"## {window_days}-day window\n")
        if not summary:
            parts.append("_no verified forecasts in this window_\n")
            continue
        parts.append(
            "| h | n | ML acc | ML Hi MAE | ML Lo MAE | ML PoP Brier | "
            "NWS acc | NWS Hi MAE | NWS Lo MAE | NWS PoP Brier | "
            "Blend Hi MAE | Blend Lo MAE | Blend PoP Brier |"
        )
        parts.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h, m in sorted(summary.items()):
            parts.append(
                "| "
                f"{h} | {m['n']} | {_fmt_pct(m['ml']['regime_acc'])} | "
                f"{_fmt_f(m['ml']['hi_mae'])} | {_fmt_f(m['ml']['lo_mae'])} | "
                f"{_fmt_f4(m['ml']['pop_brier'])} | "
                f"{_fmt_pct(m['nws']['regime_acc'])} | {_fmt_f(m['nws']['hi_mae'])} | "
                f"{_fmt_f(m['nws']['lo_mae'])} | {_fmt_f4(m['nws']['pop_brier'])} | "
                f"{_fmt_f(m['blend']['hi_mae'])} | {_fmt_f(m['blend']['lo_mae'])} | "
                f"{_fmt_f4(m['blend']['pop_brier'])} |"
            )
        parts.append("")
    return "\n".join(parts)


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{100 * v:.1f}%"


def _fmt_f(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.2f}"


def _fmt_f4(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.4f}"


def main() -> int:
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
