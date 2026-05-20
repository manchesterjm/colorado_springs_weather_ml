"""Station registry for the production weather forecasting project.

Defines the 22-station network used for feature engineering and historical
backfill. Provides ICAO -> USAF-WBAN mapping via NOAA's `isd-history.csv`
(cached locally on first call) so the NCEI ISD backfill can resolve canonical
station IDs without hard-coding them.

Station selection rationale:
    Local/Plains: primary signal (0-50 mi)
    Adjacent: short-range upstream (50-100 mi N)
    Mid-W mountain: frontal-arrival lead (75-200 mi W)
    SW synoptic: 4-Corners low signal (200-300 mi SW)
    Far-W synoptic: 1-2 day Pacific lead (370-550 mi W)
    S synoptic: Gulf moisture / cold-front lead (200-400 mi S/SE)
"""

from __future__ import annotations

import csv
import io as _stdlib_io
import logging
from dataclasses import dataclass
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


@dataclass(frozen=True)
class Station:
    """A weather station in the forecasting network."""

    icao: str            # 4-letter ICAO code (e.g., 'KCOS')
    name: str            # Human-readable name
    lat: float           # Decimal degrees, +N
    lon: float           # Decimal degrees, +E (so US is negative)
    elev_m: float        # Elevation in meters
    role: str            # Group label: local, adjacent, mid_w, sw, far_w, s
    distance_mi: int     # Approximate distance from KCOS in miles (0 for KCOS)


# 22-station network. Distances and elevations are nominal; canonical
# location data comes from NCEI isd-history.csv at runtime when needed.
STATIONS: tuple[Station, ...] = (
    # Local / Plains (primary signal)
    Station("KCOS", "Colorado Springs Muni",  38.8058, -104.7009, 1888,  "local",      0),
    Station("KFLY", "Meadow Lake",            38.9447, -104.5697, 2138,  "local",     12),
    Station("KAFF", "USAF Academy",           38.9697, -104.8133, 2018,  "local",     12),
    Station("KFCS", "Fort Carson",            38.6783, -104.7566, 1788,  "local",     10),
    Station("KAPA", "Centennial",             39.5700, -104.8493, 1791,  "local",     53),
    Station("KPUB", "Pueblo Memorial",        38.2891, -104.4967, 1438,  "local",     40),
    # Adjacent (50-100 mi N)
    Station("KDEN", "Denver Intl",            39.8561, -104.6737, 1655,  "adjacent",  68),
    Station("KCYS", "Cheyenne Regional",      41.1556, -104.8117, 1872,  "adjacent", 162),
    # Mid-W mountain / Western Slope (frontal-arrival lead, 75-200 mi W)
    Station("KLXV", "Lake County / Leadville", 39.2228, -106.3169, 3024, "mid_w",     90),
    Station("KAEJ", "Buena Vista (Central CO)", 38.7050, -106.1244, 2300, "mid_w",     76),
    Station("KASE", "Aspen Pitkin",           39.2231, -106.8688, 2384,  "mid_w",    120),
    Station("KEGE", "Eagle County Regional",  39.6427, -106.9176, 2002,  "mid_w",    130),
    Station("KGUC", "Gunnison-Crested Butte", 38.5339, -106.9333, 2336,  "mid_w",    122),
    Station("KGJT", "Grand Junction Regional", 39.1224, -108.5267, 1475, "mid_w",    206),
    Station("KMTJ", "Montrose Regional",      38.5097, -107.8939, 1758,  "mid_w",    172),
    # SW synoptic (200-300 mi SW)
    Station("KDRO", "Durango-La Plata",       37.1515, -107.7538, 2038,  "sw",       193),
    Station("KFMN", "Four Corners Regional",  36.7411, -108.2300, 1677,  "sw",       240),
    Station("KFLG", "Flagstaff Pulliam",      35.1383, -111.6711, 2135,  "sw",       483),
    # Far-W synoptic (370-550 mi W)
    Station("KSLC", "Salt Lake City Intl",    40.7884, -111.9778, 1288,  "far_w",    430),
    Station("KBOI", "Boise Air Terminal",     43.5644, -116.2228,  858,  "far_w",    700),
    # S synoptic (200-400 mi S/SE; Gulf moisture / cold-front lead)
    Station("KELP", "El Paso International",  31.8111, -106.3779, 1207,  "s",        470),
    Station("KAMA", "Amarillo Rick Husband",  35.2192, -101.7058, 1099,  "s",        320),
    Station("KDDC", "Dodge City Regional",    37.7633,  -99.9656,  790,  "s",        310),
)

ICAO_INDEX: dict[str, Station] = {s.icao: s for s in STATIONS}

# NCEI ISD has multiple records per ICAO when stations were re-registered.
# We pick the one whose end-date is most recent (active or recently active).
# Cached on first lookup.
_isd_cache: dict[str, tuple[str, str]] = {}  # icao -> (usaf, wban); empty = unloaded


def get_station(icao: str) -> Station:
    """Look up a station by ICAO code. Raises KeyError if unknown."""
    return ICAO_INDEX[icao.upper()]


def stations_by_role(role: str) -> tuple[Station, ...]:
    """Return all stations with a given role tag."""
    return tuple(s for s in STATIONS if s.role == role)


def load_isd_history(cache_path: Path, force_refresh: bool = False) -> dict[str, tuple[str, str]]:
    """Load the ICAO -> (USAF, WBAN) mapping from NOAA's isd-history.csv.

    Downloads the file once and caches it under ``cache_path``. Subsequent
    calls reuse the cached file. If multiple records exist for one ICAO,
    the one with the most recent END date wins (handles re-registrations).

    Args:
        cache_path: Where to store ``isd-history.csv`` (e.g. data/isd_history.csv).
        force_refresh: If True, redownload even if the cache exists.

    Returns:
        Dict mapping ICAO (4-letter) -> (USAF, WBAN) string pair.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if force_refresh or not cache_path.exists():
        logger.info("Downloading NCEI isd-history.csv to %s", cache_path)
        with urllib.request.urlopen(ISD_HISTORY_URL, timeout=60) as resp:
            data = resp.read()
        cache_path.write_bytes(data)

    mapping: dict[str, tuple[str, str, str]] = {}  # icao -> (usaf, wban, end)
    with cache_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            icao = (row.get("ICAO") or "").strip().upper()
            usaf = (row.get("USAF") or "").strip()
            wban = (row.get("WBAN") or "").strip()
            end = (row.get("END") or "").strip()
            if not icao or not usaf or not wban:
                continue
            if usaf == "999999" or wban == "99999":
                continue
            existing = mapping.get(icao)
            if existing is None or end > existing[2]:
                mapping[icao] = (usaf, wban, end)

    return {icao: (usaf, wban) for icao, (usaf, wban, _end) in mapping.items()}


def get_isd_id(icao: str, cache_path: Path) -> tuple[str, str]:
    """Resolve an ICAO code to its NCEI ISD (USAF, WBAN) pair.

    Loads and caches the isd-history mapping on first call. Raises
    KeyError if the ICAO has no usable entry in NCEI.

    Args:
        icao: 4-letter ICAO code.
        cache_path: Where to cache ``isd-history.csv``.

    Returns:
        Tuple of (USAF id, WBAN id) — both as strings (preserve leading zeros).
    """
    if not _isd_cache:
        _isd_cache.update(load_isd_history(cache_path))
    icao = icao.upper()
    if icao not in _isd_cache:
        raise KeyError(f"No NCEI ISD entry found for ICAO {icao!r}")
    return _isd_cache[icao]


def isd_csv_url(usaf: str, wban: str, year: int) -> str:
    """Build the NCEI ISD per-year per-station CSV URL.

    NCEI publishes hourly observations at::

        https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{USAF}{WBAN}.csv
    """
    return f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv"


def summary() -> str:
    """Render a compact summary of the station network for log/diagnostic use."""
    buf = _stdlib_io.StringIO()
    by_role: dict[str, list[Station]] = {}
    for s in STATIONS:
        by_role.setdefault(s.role, []).append(s)
    print(f"Station network ({len(STATIONS)} stations)", file=buf)
    for role in ("local", "adjacent", "mid_w", "sw", "far_w", "s"):
        bucket = by_role.get(role, [])
        if not bucket:
            continue
        print(f"  {role:9s}: {', '.join(s.icao for s in bucket)}", file=buf)
    return buf.getvalue()


if __name__ == "__main__":
    print(summary())
