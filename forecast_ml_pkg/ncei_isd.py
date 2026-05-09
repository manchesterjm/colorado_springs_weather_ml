"""NCEI ISD CSV parser.

NOAA's Integrated Surface Database publishes hourly observations as
per-year, per-station CSV files at::

    https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv

Each row is one observation. Most weather elements are encoded as
comma-separated, integer-scaled fields with explicit quality flags and
sentinel values for missing data. This module parses the subset of fields
the production forecaster needs (temperature, dew point, wind, visibility,
precipitation, pressure, sky cover, plus the raw METAR remark), converts
units to project-standard (Fahrenheit-friendly Celsius / inches / knots /
inHg), and returns dicts ready for ``metar_hourly`` insertion.

References:
    https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.error
import urllib.request
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Unit conversions
MS_TO_KNOTS = 1.94384
M_TO_SMI = 1.0 / 1609.344
MM_TO_IN = 1.0 / 25.4
HPA_TO_INHG = 0.0295300

# We only ingest near-hourly synoptic observations. SHEF (hydrologic),
# SOD (summary-of-day), SOM (summary-of-month) are skipped.
WANTED_REPORT_TYPES = {"FM-15", "FM-16"}  # METAR, SPECI


def _parse_signed_int(text: str, missing: int = 9999) -> Optional[int]:
    """Parse ``+0240`` / ``-0006`` / ``+9999`` (missing) -> int or None."""
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    if abs(value) >= missing:
        return None
    return value


def _parse_unsigned_int(text: str, missing_token: str = "9") -> Optional[int]:
    """Parse plain ints, treating all-nines as missing."""
    if not text or not text.lstrip("+-").isdigit():
        return None
    if text.lstrip("+-") == missing_token * len(text.lstrip("+-")):
        return None
    return int(text)


def _split_field(raw: str, expected_parts: int) -> Optional[list[str]]:
    """Split a comma-separated NCEI element field, validating part count."""
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) < expected_parts:
        return None
    return parts


def parse_temp_c(raw: str) -> Optional[float]:
    """Parse TMP / DEW: ``±tttt,q`` where ttt is tenths of °C, +9999 = missing."""
    parts = _split_field(raw, 2)
    if not parts:
        return None
    value = _parse_signed_int(parts[0])
    if value is None:
        return None
    return value / 10.0


def parse_wind(raw: str) -> tuple[Optional[int], Optional[float]]:
    """Parse WND: ``ddd,q,t,sssm,q`` -> (direction_deg, speed_knots).

    ``ddd`` = 999 means missing/variable; ``sssm`` is m/s × 10, missing=9999.
    """
    parts = _split_field(raw, 5)
    if not parts:
        return None, None
    raw_dir = parts[0]
    raw_speed = parts[3]

    direction: Optional[int]
    if raw_dir == "999" or not raw_dir.isdigit():
        direction = None
    else:
        direction = int(raw_dir)
        if direction < 0 or direction > 360:
            direction = None

    speed: Optional[float]
    if raw_speed == "9999" or not raw_speed.isdigit():
        speed = None
    else:
        speed = int(raw_speed) / 10.0 * MS_TO_KNOTS

    return direction, speed


def parse_visibility_smi(raw: str) -> Optional[float]:
    """Parse VIS: ``dddddd,q,v,vq`` -> visibility in statute miles.

    Distance is in meters. 999999 means missing.
    """
    parts = _split_field(raw, 4)
    if not parts:
        return None
    raw_dist = parts[0]
    if raw_dist == "999999" or not raw_dist.isdigit():
        return None
    meters = int(raw_dist)
    if meters > 160000:  # > 100 miles is unphysical
        return None
    return meters * M_TO_SMI


def parse_pressure_inhg(slp_raw: str) -> Optional[float]:
    """Parse SLP: ``ppppp,q`` -> sea-level pressure in inHg. 99999 = missing."""
    parts = _split_field(slp_raw, 2)
    if not parts:
        return None
    raw_p = parts[0]
    if raw_p == "99999" or not raw_p.isdigit():
        return None
    hpa = int(raw_p) / 10.0
    if hpa < 850 or hpa > 1100:
        return None
    return hpa * HPA_TO_INHG


def parse_precip_in(aa1_raw: str) -> Optional[float]:
    """Parse AA1: ``hh,dddd,t,q`` -> precip depth in inches over the hh hours.

    We accept any AA1 row regardless of its hour-period because the daily
    aggregator sums per-day, and most AA1 rows are 1-hour. Missing depth
    (9999) or trace-only rows (depth=0000 with t=2) -> 0.0 inches.
    """
    parts = _split_field(aa1_raw, 4)
    if not parts:
        return None
    raw_depth = parts[1]
    if raw_depth == "9999":
        return None
    if not raw_depth.isdigit():
        return None
    mm_x10 = int(raw_depth)
    return mm_x10 / 10.0 * MM_TO_IN


def parse_gust_knots(oc1_raw: str) -> Optional[float]:
    """Parse OC1 (wind gust): ``ssss,q`` -> gust in knots. 9999 = missing."""
    parts = _split_field(oc1_raw, 2)
    if not parts:
        return None
    raw = parts[0]
    if raw == "9999" or not raw.isdigit():
        return None
    return int(raw) / 10.0 * MS_TO_KNOTS


SKY_COVER_CODES = {
    "00": "CLR", "01": "FEW", "02": "FEW", "03": "SCT", "04": "SCT",
    "05": "BKN", "06": "BKN", "07": "BKN", "08": "OVC", "09": "OBS",
    "10": "POB", "99": None,
}


def parse_sky_condition(*ga_fields: str) -> Optional[str]:
    """Stitch GA1..GA4 layers into a 'FEW100 SCT200 BKN300' style string."""
    layers: list[str] = []
    for raw in ga_fields:
        parts = _split_field(raw, 6)
        if not parts:
            continue
        cover = SKY_COVER_CODES.get(parts[0])
        if cover is None:
            continue
        height_str = parts[2]
        if height_str == "+99999" or not height_str.lstrip("+-").isdigit():
            layers.append(cover)
            continue
        height_m = int(height_str)
        height_hft = max(0, round(height_m * 3.28084 / 100))
        layers.append(f"{cover}{height_hft:03d}")
    if not layers:
        return None
    return " ".join(layers)


def parse_row(row: dict[str, str]) -> Optional[dict[str, object]]:
    """Convert one NCEI ISD CSV row to our ``metar_hourly`` row dict.

    Returns ``None`` if the row is the wrong report type or carries no
    usable temperature reading (which is the minimum we need).
    """
    report_type = (row.get("REPORT_TYPE") or "").strip()
    if report_type not in WANTED_REPORT_TYPES:
        return None

    date = (row.get("DATE") or "").strip()
    if not date:
        return None
    # ISD timestamps are UTC; normalize to project-standard 'Z' suffix.
    obs_time = date if date.endswith("Z") else f"{date}Z"

    call_sign = (row.get("CALL_SIGN") or "").strip()
    station_icao = call_sign[:4].upper() if call_sign and call_sign != "99999" else ""

    temp_c = parse_temp_c(row.get("TMP", ""))
    if temp_c is None:
        return None  # No usable temperature -> skip

    dew_c = parse_temp_c(row.get("DEW", ""))
    wind_dir, wind_kt = parse_wind(row.get("WND", ""))
    vis_smi = parse_visibility_smi(row.get("VIS", ""))
    pressure_inhg = parse_pressure_inhg(row.get("SLP", ""))
    precip_in = parse_precip_in(row.get("AA1", ""))
    gust_kt = parse_gust_knots(row.get("OC1", ""))
    sky = parse_sky_condition(
        row.get("GA1", ""), row.get("GA2", ""),
        row.get("GA3", ""), row.get("GA4", ""),
    )
    rem = (row.get("REM") or "").strip() or None

    return {
        "station_icao": station_icao,
        "observation_time": obs_time,
        "temperature_c": temp_c,
        "dewpoint_c": dew_c,
        "wind_direction_deg": wind_dir,
        "wind_speed_kt": wind_kt,
        "wind_gust_kt": gust_kt,
        "visibility_sm": vis_smi,
        "precip_in": precip_in,
        "pressure_inhg": pressure_inhg,
        "sky_condition": sky,
        "weather_phenomena": None,  # left empty here; can be derived later
        "raw_data": rem,
    }


def fetch_year_csv(usaf: str, wban: str, year: int, timeout: int = 60) -> Optional[str]:
    """Download a NCEI ISD per-station-year CSV. Returns text, or None on 404.

    Raises ``urllib.error.URLError`` for non-404 errors so the caller can retry.
    """
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return data.decode("utf-8", errors="replace")


def parse_year_csv(csv_text: str) -> Iterator[dict[str, object]]:
    """Iterate parsed rows from a NCEI ISD year CSV text blob."""
    reader = csv.DictReader(io.StringIO(csv_text))
    for src_row in reader:
        parsed = parse_row(src_row)
        if parsed is not None:
            yield parsed
