"""Number / percentage formatting helpers for the forecast renderers.

Every helper renders ``None`` as ``"--"``. Names state the precision and the
input scale explicitly, so the daily report (``output.py``) and the accuracy
dashboard -- which need different precisions -- never collide on a shared
name the way the old ``_fmt_f`` copies did.
"""

from __future__ import annotations

from typing import Optional

DASH = "--"


def temp(v: Optional[float]) -> str:
    """Whole-degree temperature: ``78.4`` -> ``'78'``."""
    return DASH if v is None else f"{v:.0f}"


def inches(v: Optional[float]) -> str:
    """Two-decimal inches with a trailing inch mark: ``0.03`` -> ``'0.03"'``."""
    return DASH if v is None else f'{v:.2f}"'


def pct_from_fraction(v: Optional[float]) -> str:
    """Whole percent from a 0..1 fraction: ``0.234`` -> ``'23%'``."""
    return DASH if v is None else f"{int(round(v * 100)):d}%"


def pct_from_whole(v: Optional[float]) -> str:
    """Whole percent from an already-0..100 value: ``30.0`` -> ``'30%'``."""
    return DASH if v is None else f"{int(round(v)):d}%"


def num2(v: Optional[float]) -> str:
    """Two-decimal number: ``1.234`` -> ``'1.23'``."""
    return DASH if v is None else f"{v:.2f}"


def num4(v: Optional[float]) -> str:
    """Four-decimal number: ``0.12345`` -> ``'0.1235'``."""
    return DASH if v is None else f"{v:.4f}"


def pct1(v: Optional[float]) -> str:
    """One-decimal percent from a 0..1 fraction: ``0.234`` -> ``'23.4%'``."""
    return DASH if v is None else f"{100 * v:.1f}%"


def signed2(v: Optional[float]) -> str:
    """Two-decimal signed number: ``1.2`` -> ``'+1.20'``."""
    return DASH if v is None else f"{v:+.2f}"


def signed4(v: Optional[float]) -> str:
    """Four-decimal signed number: ``0.012`` -> ``'+0.0120'``."""
    return DASH if v is None else f"{v:+.4f}"


def pct1_signed(v: Optional[float]) -> str:
    """One-decimal signed percent from a 0..1 fraction: ``-0.02`` -> ``'-2.0%'``."""
    return DASH if v is None else f"{100 * v:+.1f}%"
