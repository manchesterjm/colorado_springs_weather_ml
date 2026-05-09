"""Train the extended (regime + Hi + Lo + PoP) forecaster.

Default config trains on ACIS daily data 1948-2020, calibrates 2021-2022,
tests 2023-2025. Override any field via CLI flags. Writes a combined
artifact to ``data/model_runs/`` and inserts a row in the ``model_runs``
table marked ``is_active=1``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg.io import setup_utf8_stdout  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.trainer_extended import (  # noqa: E402  pylint: disable=wrong-import-position
    ARTIFACT_DIR, ExtendedForecastTrainer, ExtendedTrainerConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train extended forecaster")
    p.add_argument(
        "--horizons", default="1,2,3,4,5,6,7",
        help="Comma-separated list of forecast horizons (default 1..7)",
    )
    p.add_argument("--train-end", default="2020-12-31")
    p.add_argument("--val-start", default="2021-01-01")
    p.add_argument("--val-end", default="2022-12-31")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument(
        "--neighbors", default="KPUB,KDEN,KAPA",
        help="Comma-separated ACIS neighbor stations (used by base regime features)",
    )
    p.add_argument(
        "--classifier", default="histgbm",
        choices=("histgbm", "histgbm_tuned", "lgbm"),
        help="Regime classifier kind (default histgbm)",
    )
    p.add_argument(
        "--no-test", action="store_true",
        help="Skip the test split (production retrain mode)",
    )
    p.add_argument(
        "--artifact",
        help="Output artifact path (default model_runs/extended_v1.pkl)",
    )
    return p.parse_args()


def main() -> int:
    setup_utf8_stdout()
    args = parse_args()

    horizons = tuple(int(x) for x in args.horizons.split(","))
    neighbors = tuple(s.strip().upper() for s in args.neighbors.split(",") if s.strip())
    artifact_path = Path(args.artifact) if args.artifact else ARTIFACT_DIR / "extended_v1.pkl"

    cfg = ExtendedTrainerConfig(
        horizons=horizons,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=None if args.no_test else args.test_start,
        test_end=None if args.no_test else args.test_end,
        neighbor_stations=neighbors,
        classifier_kind=args.classifier,
        artifact_path=artifact_path,
    )

    trainer = ExtendedForecastTrainer(cfg)
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
