"""Renderers for the daily forecast: text, markdown, DB.

Each renderer takes the same ``ForecastResult`` payload and produces a
different artifact. Keeping them separate (rather than threading a
markdown writer through the runner) makes it cheap to add new outputs
later (Slack, email, RSS).
"""

from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from weather_regime_pkg import STATE_LABELS

from . import db as fdb
from .io import tz_utils


@dataclass
class HorizonForecast:
    """One horizon's full forecast payload."""
    horizon_days: int
    target_date_local: str
    weekday: str
    regime_probs: list[float]
    regime_top1: int
    regime_top1_label: str
    hi_f_ml: Optional[float] = None
    lo_f_ml: Optional[float] = None
    pop_ml: Optional[float] = None
    nws_hi_f: Optional[float] = None
    nws_lo_f: Optional[float] = None
    nws_pop: Optional[float] = None
    nws_regime: Optional[int] = None
    nws_text: Optional[str] = None
    blend_hi_f: Optional[float] = None
    blend_lo_f: Optional[float] = None
    blend_pop: Optional[float] = None
    w_ml: Optional[float] = None


@dataclass
class ForecastResult:
    """Complete output of one forecast run."""
    issue_time: str
    issue_date_local: str
    anchor_date_local: str
    anchor_state_label: str
    anchor_hi_f: Optional[float]
    anchor_lo_f: Optional[float]
    anchor_precip_in: Optional[float]
    model_version: str
    model_run_id: str
    nws_fetch_time: Optional[str]
    horizons: list[HorizonForecast] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text (terminal) renderer
# ---------------------------------------------------------------------------

def render_text(result: ForecastResult) -> str:
    """Plain-text terminal-friendly summary."""
    buf = io.StringIO()
    bar = "=" * 90
    print(bar, file=buf)
    print(f"COLORADO SPRINGS WEATHER ML FORECAST — issued {result.issue_time}", file=buf)
    print(bar, file=buf)
    print(f"Anchor day:   {result.anchor_date_local}  state={result.anchor_state_label}  "
          f"Hi={_fmt_f(result.anchor_hi_f)}  Lo={_fmt_f(result.anchor_lo_f)}  "
          f"precip={_fmt_in(result.anchor_precip_in)}", file=buf)
    print(f"Model:        {result.model_version}  (run {result.model_run_id})", file=buf)
    print(f"NWS issued:   {result.nws_fetch_time or 'n/a'}", file=buf)
    print(file=buf)

    print(f"{'Date':12s} {'Day':10s} h  "
          f"{'ML regime':12s} {'pTop':>5s}  "
          f"{'ML Hi':>5s} {'ML Lo':>5s} {'PoP':>5s}  "
          f"{'NWS Hi':>6s} {'NWS Lo':>6s} {'NWS%':>5s}  "
          f"{'BlendHi':>7s} {'BlendLo':>7s} {'BlendPoP':>8s}",
          file=buf)
    print("-" * 110, file=buf)
    for fc in result.horizons:
        ptop = fc.regime_probs[fc.regime_top1]
        print(f"{fc.target_date_local:12s} {fc.weekday:10s} {fc.horizon_days:1d}  "
              f"{fc.regime_top1_label:12s} {ptop:>5.2f}  "
              f"{_fmt_f(fc.hi_f_ml, w=5):>5s} {_fmt_f(fc.lo_f_ml, w=5):>5s} "
              f"{_fmt_pct(fc.pop_ml):>5s}  "
              f"{_fmt_f(fc.nws_hi_f, w=6):>6s} {_fmt_f(fc.nws_lo_f, w=6):>6s} "
              f"{_fmt_pct_int(fc.nws_pop):>5s}  "
              f"{_fmt_f(fc.blend_hi_f, w=7):>7s} {_fmt_f(fc.blend_lo_f, w=7):>7s} "
              f"{_fmt_pct(fc.blend_pop):>8s}",
              file=buf)
    print(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_markdown(result: ForecastResult) -> str:
    """Markdown report — one self-contained file per issue date."""
    buf = io.StringIO()
    print(f"# Colorado Springs Weather ML Forecast", file=buf)
    print(f"\n**Issued:** {result.issue_time}  ", file=buf)
    print(f"**Anchor:** {result.anchor_date_local} "
          f"({result.anchor_state_label}; "
          f"Hi {_fmt_f(result.anchor_hi_f)}, "
          f"Lo {_fmt_f(result.anchor_lo_f)}, "
          f"precip {_fmt_in(result.anchor_precip_in)})  ", file=buf)
    print(f"**Model:** {result.model_version} (run {result.model_run_id})  ", file=buf)
    print(f"**NWS source:** {result.nws_fetch_time or 'unavailable'}", file=buf)
    print(file=buf)
    print("## Forecast", file=buf)
    print(file=buf)
    print("| Date | Day | h | ML regime | p | ML Hi | ML Lo | ML PoP "
          "| NWS Hi | NWS Lo | NWS PoP | Blend Hi | Blend Lo | Blend PoP |", file=buf)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", file=buf)
    for fc in result.horizons:
        ptop = fc.regime_probs[fc.regime_top1]
        print("| "
              f"{fc.target_date_local} | {fc.weekday} | {fc.horizon_days} | "
              f"{fc.regime_top1_label} | {ptop:.2f} | "
              f"{_fmt_f(fc.hi_f_ml)} | {_fmt_f(fc.lo_f_ml)} | "
              f"{_fmt_pct(fc.pop_ml)} | "
              f"{_fmt_f(fc.nws_hi_f)} | {_fmt_f(fc.nws_lo_f)} | "
              f"{_fmt_pct_int(fc.nws_pop)} | "
              f"{_fmt_f(fc.blend_hi_f)} | {_fmt_f(fc.blend_lo_f)} | "
              f"{_fmt_pct(fc.blend_pop)} |", file=buf)
    print(file=buf)
    print("## Top-3 regime probabilities", file=buf)
    print(file=buf)
    for fc in result.horizons:
        order = np.argsort(fc.regime_probs)[::-1][:3]
        line = ", ".join(f"`{STATE_LABELS[i]}` {fc.regime_probs[i]:.2f}" for i in order)
        print(f"- **{fc.target_date_local}** (h={fc.horizon_days}): {line}", file=buf)
    print(file=buf)
    if any(fc.nws_text for fc in result.horizons):
        print("## NWS short-text", file=buf)
        print(file=buf)
        for fc in result.horizons:
            if fc.nws_text:
                print(f"- **{fc.target_date_local}**: {fc.nws_text}", file=buf)
        print(file=buf)
    return buf.getvalue()


def write_markdown_report(result: ForecastResult, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{result.issue_date_local}.md"
    path.write_text(render_markdown(result), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# DB renderer (production_forecast + nws_compare_cache)
# ---------------------------------------------------------------------------

def write_to_db(result: ForecastResult) -> int:
    """Insert one row per horizon into ``production_forecast`` and one row
    per horizon into ``nws_compare_cache``. Returns rows inserted into
    production_forecast."""
    conn = fdb.get_connection()
    inserted = 0
    try:
        with conn:
            for fc in result.horizons:
                regime_json = json.dumps([float(p) for p in fc.regime_probs])
                nws_compare = {
                    "regime": (
                        STATE_LABELS[fc.nws_regime] if fc.nws_regime is not None else None
                    ),
                    "hi_f": fc.nws_hi_f,
                    "lo_f": fc.nws_lo_f,
                    "pop": fc.nws_pop,
                    "text": fc.nws_text,
                }
                blend_json = {
                    "hi_f": fc.blend_hi_f,
                    "lo_f": fc.blend_lo_f,
                    "pop": fc.blend_pop,
                    "w_ml": fc.w_ml,
                }
                cur = conn.execute(
                    """
                    INSERT OR REPLACE INTO production_forecast (
                        issue_time, issue_date_local, target_date_local,
                        horizon_days, model_version, regime_probs_json, regime_top1,
                        hi_f, lo_f, pop, pop_threshold_in,
                        nws_compare_json, ml_blend_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.issue_time, result.issue_date_local, fc.target_date_local,
                        fc.horizon_days, result.model_version, regime_json,
                        fc.regime_top1_label, fc.hi_f_ml, fc.lo_f_ml, fc.pop_ml, 0.10,
                        json.dumps(nws_compare), json.dumps(blend_json),
                        tz_utils.now_utc(),
                    ),
                )
                inserted += 1 if cur.rowcount and cur.rowcount > 0 else 0

                if fc.nws_hi_f is not None or fc.nws_lo_f is not None:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO nws_compare_cache (
                            issue_date_local, target_date_local, horizon_days,
                            nws_hi_f, nws_lo_f, nws_pop, nws_regime, nws_text, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result.issue_date_local, fc.target_date_local, fc.horizon_days,
                            fc.nws_hi_f, fc.nws_lo_f, fc.nws_pop,
                            STATE_LABELS[fc.nws_regime] if fc.nws_regime is not None else None,
                            fc.nws_text, tz_utils.now_utc(),
                        ),
                    )
    finally:
        conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_f(v: Optional[float], w: int = 0) -> str:
    if v is None:
        return "--"
    return f"{v:.0f}"


def _fmt_in(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.2f}\""


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{int(round(v * 100)):d}%"


def _fmt_pct_int(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{int(round(v)):d}%"


__all__ = [
    "HorizonForecast",
    "ForecastResult",
    "render_text",
    "render_markdown",
    "write_markdown_report",
    "write_to_db",
]
