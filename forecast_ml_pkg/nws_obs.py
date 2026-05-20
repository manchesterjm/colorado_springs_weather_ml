"""NWS observations API client and parser.

The NWS public API exposes the latest observation for a station at::

    https://api.weather.gov/stations/{ICAO}/observations/latest

Each property comes back as ``{value, unitCode, qualityControl}`` where
``unitCode`` follows the WMO scheme. We convert everything to the units
``metar_hourly`` already uses (Celsius, knots, statute miles, inches,
inches Hg) so historical NCEI data and live NWS data are interchangeable.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from forecast_ml_pkg import constants

logger = logging.getLogger(__name__)

# Unit conversions (NWS API native -> our schema)
KMH_TO_KNOTS = 0.539957
PA_TO_INHG = 0.000295300
M_TO_INCHES = 39.3701
M_TO_SMI = 1.0 / 1609.344


def _value_in(prop: Optional[dict], expected_unit: str) -> Optional[float]:
    """Extract ``properties[k]['value']`` and assert the unit matches."""
    if not prop:
        return None
    value = prop.get("value")
    if value is None:
        return None
    unit = prop.get("unitCode") or ""
    # WMO unit codes occasionally pop up as 'wmoUnit:foo' or just 'foo'.
    if expected_unit not in unit:
        logger.debug("Unexpected unit %r (wanted %r)", unit, expected_unit)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_latest(icao: str) -> Optional[dict[str, Any]]:
    """Fetch the latest NWS observation JSON for one station. Returns ``None`` on 404."""
    url = f"https://api.weather.gov/stations/{icao}/observations/latest"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": constants.HTTP_USER_AGENT, "Accept": "application/geo+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=constants.HTTP_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def render_sky_condition(layers: Optional[list[dict]]) -> Optional[str]:
    """Render a NWS cloudLayers list as 'FEW100 SCT200' style string."""
    if not layers:
        return None
    out: list[str] = []
    for layer in layers:
        amount = layer.get("amount")
        if not amount:
            continue
        base = layer.get("base") or {}
        meters = base.get("value")
        if meters is None:
            out.append(amount)
            continue
        hundreds_ft = max(0, round(meters * 3.28084 / 100))
        out.append(f"{amount}{hundreds_ft:03d}")
    return " ".join(out) if out else None


def parse_observation(obs_json: dict[str, Any]) -> Optional[dict[str, object]]:
    """Convert one NWS observation JSON into a ``metar_hourly`` row dict.

    Returns ``None`` if the observation has no usable timestamp.
    """
    props = obs_json.get("properties") or {}
    timestamp = props.get("timestamp")
    if not timestamp:
        return None

    obs_time = timestamp
    # NWS returns timestamps with explicit +00:00 offset; normalize to Z suffix
    # to match the project convention.
    if obs_time.endswith("+00:00"):
        obs_time = obs_time[:-len("+00:00")] + "Z"
    elif not obs_time.endswith("Z"):
        # Last-resort: leave as-is. tz_utils.parse_timestamp can still handle it.
        pass

    # Extract scalars in NWS native units, then convert to ours.
    temp_c = _value_in(props.get("temperature"), "degC")
    dew_c = _value_in(props.get("dewpoint"), "degC")
    wind_dir = _value_in(props.get("windDirection"), "degree")
    wind_kmh = _value_in(props.get("windSpeed"), "km_h-1")
    gust_kmh = _value_in(props.get("windGust"), "km_h-1")
    visibility_m = _value_in(props.get("visibility"), "m")
    pressure_pa = _value_in(props.get("barometricPressure"), "Pa")
    precip_m = _value_in(props.get("precipitationLastHour"), "m")

    raw_msg = (props.get("rawMessage") or "").strip() or None
    sky = render_sky_condition(props.get("cloudLayers"))

    weather = props.get("textDescription") or None
    present = props.get("presentWeather") or []
    if present:
        codes = [
            (item.get("rawString") or item.get("weather"))
            for item in present
            if item.get("rawString") or item.get("weather")
        ]
        if codes:
            weather = " ".join(c for c in codes if c)

    return {
        "station_icao": (props.get("station") or "").rsplit("/", maxsplit=1)[-1].upper(),
        "observation_time": obs_time,
        "temperature_c": temp_c,
        "dewpoint_c": dew_c,
        "wind_direction_deg": int(wind_dir) if wind_dir is not None else None,
        "wind_speed_kt": wind_kmh * KMH_TO_KNOTS if wind_kmh is not None else None,
        "wind_gust_kt": gust_kmh * KMH_TO_KNOTS if gust_kmh is not None else None,
        "visibility_sm": visibility_m * M_TO_SMI if visibility_m is not None else None,
        "precip_in": precip_m * M_TO_INCHES if precip_m is not None else None,
        "pressure_inhg": pressure_pa * PA_TO_INHG if pressure_pa is not None else None,
        "sky_condition": sky,
        "weather_phenomena": weather,
        "raw_data": raw_msg,
    }
