"""Nightly retrain entrypoint. Scheduled at 02:00 MST.

Refits all 28 models in **production mode** (no test split, recent
calibration window) so the active forecaster always reflects yesterday's
ACIS daily as a feature. The previous active model is demoted to
``is_active = 0`` and kept on disk for ablation.

Production-mode windows::

    train_end       = today - ~3 yr   (extends as ACIS catches up)
    val_start..end  = (train_end + 1d) ... (today - ~3 mo)
    test            = none

The new artifact is written to ``data/model_runs/extended_<run_id>.pkl``
and a row is added to ``model_runs`` (which the trainer wires up).
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg.io import setup_utf8_stdout  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.trainer_extended import (  # noqa: E402  pylint: disable=wrong-import-position
    ARTIFACT_DIR, ExtendedForecastTrainer, ExtendedTrainerConfig,
)


def production_windows(today_iso: str) -> tuple[str, str, str]:
    """Compute (train_end, val_start, val_end) as offsets from today.

    Defaults: train through ~33 months ago, calibrate over the most
    recent ~30 months. This mirrors the test-evaluated artifact's
    train/cal split so the scoring is roughly comparable.
    """
    today_dt = date.fromisoformat(today_iso)
    train_end_dt = today_dt - timedelta(days=int(33 * 30.4))  # ~33 months
    val_start_dt = train_end_dt + timedelta(days=1)
    val_end_dt = today_dt - timedelta(days=int(3 * 30.4))      # ~3 months
    return (
        train_end_dt.isoformat(),
        val_start_dt.isoformat(),
        val_end_dt.isoformat(),
    )


def main() -> int:
    setup_utf8_stdout()
    today = date.today().isoformat()
    train_end, val_start, val_end = production_windows(today)

    print(f"Production retrain ({today})")
    print(f"  Train end:   {train_end}")
    print(f"  Cal window:  {val_start} ... {val_end}")
    print()

    artifact_path = ARTIFACT_DIR / f"extended_prod_{today}.pkl"
    cfg = ExtendedTrainerConfig(
        horizons=(1, 2, 3, 4, 5, 6, 7),
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=None,
        test_end=None,
        neighbor_stations=("KPUB", "KDEN", "KAPA"),
        classifier_kind="histgbm",
        artifact_path=artifact_path,
        write_model_runs_row=True,
    )

    t0 = time.time()
    trainer = ExtendedForecastTrainer(cfg)
    trainer.run()
    elapsed = time.time() - t0
    print(f"\nRetrain complete in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
