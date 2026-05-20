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

from forecast_ml_pkg.externals import setup_utf8_stdout
from forecast_ml_pkg.trainer_extended import (
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

    # A/B retrain: both variants share the same window so the comparison stays
    # apples-to-apples. The trainer demotes only same-variant rows in
    # model_runs, so each variant ends with exactly one is_active=1 row.
    variants = (
        ("baseline", False, f"extended_prod_{today}.pkl"),
        ("nwsfeat",  True,  f"extended_prod_nwsfeat_{today}.pkl"),
    )
    total_t0 = time.time()
    for label, use_nws, fname in variants:
        print(f"\n=== variant: {label} (use_nws_features={use_nws}) ===")
        artifact_path = ARTIFACT_DIR / fname
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
            use_nws_features=use_nws,
        )
        t0 = time.time()
        trainer = ExtendedForecastTrainer(cfg)
        trainer.run()
        print(f"  {label} retrain complete in {time.time() - t0:.1f} s")
    print(f"\nA/B retrain complete in {time.time() - total_t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
