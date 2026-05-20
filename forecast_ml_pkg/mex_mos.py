"""MEX MOS bulletin client and per-target-date Hi/Lo/PoP derivation.

The Iowa State Mesonet archives parsed MOS bulletins at
``https://mesonet.agron.iastate.edu/api/1/mos.json``. The MEX (GFS extended)
product is issued at 00Z and 12Z, with ~15 forecast rows at 12-hour
granularity reaching roughly 7.5 days out.

Field semantics (relevant subset):
    * ``n_x``: night/day temperature in degrees Fahrenheit. The value's meaning
      depends on the ``ftime`` hour. For our use case we map each row to a
      local-Denver target date and a kind (``high`` or ``low``).
    * ``tmp``, ``dpt``: instantaneous temperature/dewpoint at the ftime.
    * ``p12``: probability of measurable precipitation (>= 0.01") in the 12-hour
      window ending at the ftime.
    * ``q12``: quantitative precipitation forecast category for the same window.
    * ``t12``: thunderstorm probability string (``"NN/NN"``).

Encoding rule (validated against actuals in step 2 of the Phase 7 plan):
    * ftime at 00Z UTC corresponds to ~17/18 hour MDT/MST the **previous** local
      day, so ``n_x`` there is the maximum during the local daytime that just
      ended -- the *high* for the **previous** local date.
    * ftime at 12Z UTC corresponds to ~05/06 hour MDT/MST the **same** local
      day, so ``n_x`` there is the minimum during the overnight period leading
      into that local morning -- the *low* for **that** local date.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from forecast_ml_pkg import constants
from forecast_ml_pkg.dates import DENVER, utc_iso_to_denver_date

logger = logging.getLogger(__name__)

BASE_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"


@dataclass(frozen=True)
class MexRow:
    """One ftime row from a MEX bulletin, normalized for our schema."""

    station_id: str
    runtime_utc: str
    ftime_utc: str
    n_x: Optional[int]
    tmp: Optional[int]
    dpt: Optional[int]
    wsp: Optional[int]
    p12: Optional[int]
    q12: Optional[float]
    t12: Optional[str]
    snw: Optional[float]
    raw_json: str


def fetch_bulletin(station: str, runtime_utc_iso: str) -> Optional[list[dict[str, Any]]]:
    """Fetch one MEX bulletin from the IEM archive.

    ``runtime_utc_iso`` should be ``"YYYY-MM-DDTHH:MM:SSZ"`` (12Z runs for our
    use case). Returns the ``data`` list of rows, or ``None`` if the archive
    has no bulletin for that runtime (an empty ``data`` field).
    """
    url = (
        f"{BASE_URL}?station={station}&model=MEX&runtime={runtime_utc_iso}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": constants.HTTP_USER_AGENT})
    with urllib.request.urlopen(req, timeout=constants.HTTP_TIMEOUT_SEC) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data") or []
    if not rows:
        return None
    return rows


def normalize_rows(station: str, raw_rows: Iterable[dict[str, Any]]) -> list[MexRow]:
    """Convert raw IEM rows to ``MexRow`` records ready for DB insert."""
    out: list[MexRow] = []
    for row in raw_rows:
        runtime_iso = _iem_runtime_to_iso(row.get("runtime_utc"))
        ftime_iso = _iem_runtime_to_iso(row.get("ftime_utc"))
        if not runtime_iso or not ftime_iso:
            continue
        out.append(MexRow(
            station_id=station,
            runtime_utc=runtime_iso,
            ftime_utc=ftime_iso,
            n_x=_int_or_none(row.get("n_x")),
            tmp=_int_or_none(row.get("tmp")),
            dpt=_int_or_none(row.get("dpt")),
            wsp=_int_or_none(row.get("wsp")),
            p12=_int_or_none(row.get("p12")),
            q12=_float_or_none(row.get("q12")),
            t12=row.get("t12"),
            snw=_float_or_none(row.get("snw")),
            raw_json=json.dumps(row, default=str, separators=(",", ":")),
        ))
    return out


def _iem_runtime_to_iso(raw: Any) -> Optional[str]:
    """IEM returns timestamps like ``'2025-05-10T12:00:00.000'`` (no offset)
    for ``*_utc`` fields. Normalize to ``'2025-05-10T12:00:00Z'``."""
    if not raw:
        return None
    s = str(raw)
    if s.endswith("Z"):
        return s
    # Drop fractional seconds, append Z.
    if "." in s:
        s = s.split(".", 1)[0]
    if "T" in s:
        return s + "Z"
    if " " in s:
        return s.replace(" ", "T") + "Z"
    return s + "T00:00:00Z"


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class DailyEntry:
    """Per-local-target-date Hi/Lo/PoP derived from a bulletin's rows."""

    issue_date_local: str
    target_date_local: str
    horizon_days: int
    nws_hi_f: Optional[float]
    nws_lo_f: Optional[float]
    nws_pop_day: Optional[float]
    nws_pop_night: Optional[float]


def derive_daily_entries(rows: list[MexRow]) -> list[DailyEntry]:
    """Collapse a bulletin's per-ftime rows into per-target-date entries.

    Maps each ``ftime_utc`` to a (target_date_local, kind) pair where ``kind``
    is ``'high'`` (daytime) or ``'low'`` (overnight). Then groups by target
    date and emits one ``DailyEntry`` per date with Hi/Lo/PoP_day/PoP_night
    populated where available. Target dates strictly after the runtime's local
    date are kept; dates not strictly after are dropped (no h<=0 rows).
    """
    if not rows:
        return []
    runtime_iso = rows[0].runtime_utc
    issue_local = utc_iso_to_denver_date(runtime_iso)
    # Bucket per target local date.
    buckets: dict[str, dict[str, Any]] = {}
    for r in rows:
        target_date, kind = _ftime_to_target(r.ftime_utc)
        if target_date is None:
            continue
        if target_date <= issue_local:
            continue
        bucket = buckets.setdefault(target_date, {})
        if kind == "high":
            if r.n_x is not None:
                bucket["hi"] = float(r.n_x)
            if r.p12 is not None:
                bucket["pop_day"] = float(r.p12)
        elif kind == "low":
            if r.n_x is not None:
                bucket["lo"] = float(r.n_x)
            if r.p12 is not None:
                bucket["pop_night"] = float(r.p12)
    out: list[DailyEntry] = []
    issue_date_obj = datetime.fromisoformat(issue_local).date()
    for target_date_str in sorted(buckets):
        target_obj = datetime.fromisoformat(target_date_str).date()
        h = (target_obj - issue_date_obj).days
        b = buckets[target_date_str]
        out.append(DailyEntry(
            issue_date_local=issue_local,
            target_date_local=target_date_str,
            horizon_days=h,
            nws_hi_f=b.get("hi"),
            nws_lo_f=b.get("lo"),
            nws_pop_day=b.get("pop_day"),
            nws_pop_night=b.get("pop_night"),
        ))
    return out


def _ftime_to_target(ftime_iso: str) -> tuple[Optional[str], Optional[str]]:
    """Map an ftime (UTC) to (target_date_local, kind).

    See module docstring for the encoding rationale.
    """
    t = datetime.fromisoformat(ftime_iso.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    if t.hour == 0:
        # Backs up one hour to land squarely in the previous Denver local day.
        target = (t - timedelta(hours=1)).astimezone(DENVER).date()
        return target.isoformat(), "high"
    if t.hour == 12:
        return t.astimezone(DENVER).date().isoformat(), "low"
    return None, None
