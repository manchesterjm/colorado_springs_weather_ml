"""Production weather forecasting ML package.

Sister package to ``weather_regime_pkg`` (research artifact). This one drives
the daily 06:30 MST forecast pipeline for Colorado Springs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the canonical scripts directory importable from any submodule the
# moment the package is loaded -- this is what lets us reuse db_utils,
# tz_utils, script_metrics, and weather_regime_pkg without per-module
# sys.path tinkering.
_SCRIPTS_DIR = Path(r"D:\Scripts") if sys.platform == "win32" else Path("/mnt/d/scripts")
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

__version__ = "0.1.0"
