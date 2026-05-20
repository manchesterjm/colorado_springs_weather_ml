"""Date / timezone helpers for the forecast pipeline.

One source of truth for UTC -> America/Denver date conversion, replacing the
near-identical copies that lived in ``mex_mos``, ``digital_forecast_agg`` and
``daily_aggregator``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# America/Denver handles the MST/MDT switch automatically. This matches
# ``tz_utils.DENVER_TZ`` in the shared D:\Scripts tree; it is defined here
# directly so this module carries no import-time dependency on that shim.
DENVER = ZoneInfo("America/Denver")


def utc_iso_to_denver_date(iso: str) -> str:
    """Return the America/Denver local date (``YYYY-MM-DD``) for a UTC ISO string.

    Accepts a trailing ``Z`` or an explicit offset. A naive timestamp (no
    offset) is assumed to be UTC.
    """
    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(DENVER).date().isoformat()
