"""Re-exports of the shared ``D:\\Scripts`` helper modules.

``db_utils``, ``script_metrics`` and ``tz_utils`` live under ``D:\\Scripts``,
outside this package. ``forecast_ml_pkg/__init__.py`` puts that directory on
``sys.path`` at import time, so this module simply re-exports them and gives
the rest of the package one stable import location::

    from forecast_ml_pkg.externals import db_utils, tz_utils

This module was previously named ``io.py``; it was renamed because that name
shadowed the standard-library ``io`` module and invited confusion.
"""

from __future__ import annotations

from pathlib import Path

import db_utils
import script_metrics
import tz_utils
from weather_regime_pkg._io import setup_utf8_stdout

# Canonical D:\Scripts location, taken from db_utils so the win32/POSIX split
# is defined in exactly one place.
SCRIPTS_DIR: Path = db_utils.SCRIPTS_DIR

__all__ = ["db_utils", "script_metrics", "tz_utils", "setup_utf8_stdout", "SCRIPTS_DIR"]
