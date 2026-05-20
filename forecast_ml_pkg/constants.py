"""Project-wide constants shared across forecast_ml_pkg modules.

One source of truth for values that were previously re-declared in several
modules (the primary station ICAO, the HTTP client identity and timeout).
"""

from __future__ import annotations

# Primary forecast station: Colorado Springs Municipal Airport.
STATION_KCOS = "KCOS"

# Identity + timeout for every outbound HTTP request (IEM MOS archive,
# NWS observations API, NCEI ISD). Bumping the UA touches one line.
HTTP_USER_AGENT = "forecast_ml_pkg/0.1 (manchesterjm@gmail.com)"
HTTP_TIMEOUT_SEC = 30
