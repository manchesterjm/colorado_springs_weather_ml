"""ML <-> NWS blending for regime probability vectors and Hi/Lo/PoP heads.

For the regime classifier we reuse ``weather_regime_pkg.nws.blend`` -- the
linear-pool combine over 9-state vectors. This module adds:

    * ``blend_temperature``  -- weighted average of ML and NWS Tmax / Tmin
    * ``blend_pop``          -- weighted average of ML PoP probability and
                                NWS PoP fraction (max-of-day-and-night).

Default weights are 0.5/0.5 at h=1 and shift toward NWS at longer
horizons, reflecting the empirical w* values found during the Markov
trainer's blend backtest (see Markov reference doc for context).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from weather_regime_pkg import blend as blend_regime  # re-export

# Empirically optimal w_ML weights from the Phase 3 backtest:
# h=1 -> 0.5 (50/50), h=3 -> 0.3 (NWS-leaning), h=7 -> 0.5 (no NWS data).
# Use a smooth interpolation for unknown horizons.
DEFAULT_W_ML_BY_HORIZON: dict[int, float] = {
    1: 0.50,
    2: 0.40,
    3: 0.30,
    4: 0.30,
    5: 0.40,
    6: 0.50,
    7: 0.50,
}


def get_default_w_ml(horizon: int) -> float:
    """Look up the default ML weight for a given horizon (interpolates if unseen)."""
    if horizon in DEFAULT_W_ML_BY_HORIZON:
        return DEFAULT_W_ML_BY_HORIZON[horizon]
    near = min(DEFAULT_W_ML_BY_HORIZON.keys(), key=lambda k: abs(k - horizon))
    return DEFAULT_W_ML_BY_HORIZON[near]


def blend_temperature(
    ml_value: Optional[float],
    nws_value: Optional[float],
    w_ml: float,
) -> Optional[float]:
    """Weighted average of two temperatures. ``None`` if both inputs are missing.

    If only one input is present, the blend collapses to that input.
    """
    if ml_value is None and nws_value is None:
        return None
    if nws_value is None:
        return float(ml_value)
    if ml_value is None:
        return float(nws_value)
    return w_ml * float(ml_value) + (1.0 - w_ml) * float(nws_value)


def blend_pop(
    ml_pop: Optional[float],
    nws_pop_pct: Optional[float],
    w_ml: float,
) -> Optional[float]:
    """Linear blend of ML probability (0..1) and NWS PoP percentage (0..100)."""
    nws_p = nws_pop_pct / 100.0 if nws_pop_pct is not None else None
    if ml_pop is None and nws_p is None:
        return None
    if nws_p is None:
        return float(np.clip(ml_pop, 0.0, 1.0))
    if ml_pop is None:
        return float(nws_p)
    return float(np.clip(w_ml * ml_pop + (1.0 - w_ml) * nws_p, 0.0, 1.0))


__all__ = [
    "blend_regime",
    "blend_temperature",
    "blend_pop",
    "get_default_w_ml",
    "DEFAULT_W_ML_BY_HORIZON",
]
