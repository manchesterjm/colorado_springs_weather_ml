"""Build the anchor-day feature vector for live forecasting.

Wraps ``weather_forecast_now.py``'s anchor logic into a reusable function
the production runner can call. Loads ACIS daily for KCOS and neighbors,
classifies states using the *trained* terciles, and emits a single
feature row matching the artifact's ``feature_names``.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np

from weather_regime_pkg import (avg_temp, classify, doy, fetch_acis)


def load_classified_data(
    artifact: dict,
) -> tuple[list[dict], dict[str, dict]]:
    """Load KCOS + neighbors, classify in place using artifact's terciles.

    Returns ``(kcos_rows, neighbor_lookups)`` where ``neighbor_lookups`` is
    ``{station_id: {date: row}}``.
    """
    rows = fetch_acis("KCOS")
    classify(rows, artifact["terciles"])

    neighbor_lookups: dict[str, dict] = {}
    for sid in artifact.get("neighbor_stations", []):
        nrows = fetch_acis(sid)
        n_terc = artifact["neighbor_terciles"].get(sid)
        if n_terc:
            classify(nrows, n_terc)
        neighbor_lookups[sid] = {r["date"]: r for r in nrows}

    return rows, neighbor_lookups


def find_anchor_index(rows: list[dict]) -> Optional[int]:
    """Last row with a complete contiguous 7-day classified history."""
    for i in range(len(rows) - 1, 5, -1):
        win = rows[i - 6:i + 1]
        if any(r["state"] is None for r in win):
            continue
        contiguous = all(
            (datetime.strptime(win[k]["date"], "%Y-%m-%d")
             - datetime.strptime(win[k - 1]["date"], "%Y-%m-%d")).days == 1
            for k in range(1, len(win))
        )
        if contiguous:
            return i
    return None


def build_anchor_features(
    rows: list[dict],
    anchor_idx: int,
    doy_clim_temp: np.ndarray,
    neighbor_lookups: dict[str, dict],
) -> np.ndarray:
    """Emit a (1, n_features) feature row for the given anchor index.

    Layout matches ``weather_regime_pkg.features.build_features``: KCOS
    base block, then 3 features per neighbor in the same order the
    artifact saved.
    """
    r_t = rows[anchor_idx]
    r_tm1 = rows[anchor_idx - 1]
    r_tm2 = rows[anchor_idx - 2]
    d = doy(r_t["date"])
    avg_t = avg_temp(r_t)
    anom = (
        avg_t - doy_clim_temp[d]
        if not math.isnan(doy_clim_temp[d]) else 0.0
    )
    feat: list[float] = [
        float(r_t["state"]),
        float(r_tm1["state"]),
        float(r_tm2["state"]),
        r_t["maxt"], r_t["mint"], avg_t,
        r_tm1["maxt"], r_tm1["mint"],
        r_t["pcpn"], r_tm1["pcpn"], r_tm2["pcpn"],
        sum(rows[j]["pcpn"] for j in range(anchor_idx - 6, anchor_idx + 1)),
        anom,
        math.sin(2 * math.pi * d / 365.25),
        math.cos(2 * math.pi * d / 365.25),
        float(int(r_t["date"][5:7])),
    ]
    for lookup in neighbor_lookups.values():
        nrow = lookup.get(r_t["date"])
        if nrow is None:
            feat.extend([np.nan, np.nan, np.nan])
            continue
        s = nrow.get("state")
        feat.extend([
            float(s) if s is not None else np.nan,
            nrow["maxt"] if nrow["maxt"] is not None else np.nan,
            nrow["pcpn"] if nrow["pcpn"] is not None else np.nan,
        ])
    return np.array([feat], dtype=float)
