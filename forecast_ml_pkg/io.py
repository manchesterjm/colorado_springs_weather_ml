"""Shared-utility shim for the forecast_ml package.

Re-exports the canonical helpers that already live under ``D:\\Scripts`` so
this project shares one source of truth for database access, timezone
handling, and script metrics. The first time this module is imported it
prepends the scripts directory to ``sys.path`` if needed.

Note on naming: although this file is named ``io.py``, Python's absolute
import rules ensure ``import io`` elsewhere still resolves to the stdlib
``io`` module. To use this shim, do ``from forecast_ml_pkg import io as fio``
or ``from forecast_ml_pkg.io import db_utils``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the canonical scripts directory regardless of host OS.
if sys.platform == "win32":
    _SCRIPTS_DIR = Path(r"D:\Scripts")
else:
    _SCRIPTS_DIR = Path("/mnt/d/Scripts")

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Re-export the three shared utility modules. Consumers can either use
# ``forecast_ml_pkg.io.db_utils`` or import the module names directly via
# ``import db_utils`` once this module has run.
import db_utils  # noqa: E402  pylint: disable=wrong-import-position
import script_metrics  # noqa: E402  pylint: disable=wrong-import-position
import tz_utils  # noqa: E402  pylint: disable=wrong-import-position
from weather_regime_pkg._io import setup_utf8_stdout  # noqa: E402  pylint: disable=wrong-import-position

__all__ = ["db_utils", "script_metrics", "tz_utils", "setup_utf8_stdout", "SCRIPTS_DIR"]

SCRIPTS_DIR: Path = _SCRIPTS_DIR
