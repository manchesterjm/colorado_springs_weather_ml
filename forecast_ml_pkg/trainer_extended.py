"""Extended forecast trainer: regime + Hi + Lo + PoP per horizon.

Subclasses ``RegimeForecastTrainer`` from ``weather_regime_pkg`` and adds
three continuous-style heads (Hi, Lo, PoP) at each horizon, fit on the
same daily feature matrix and train/val/test split. Default horizons are
``(1..7)`` -- a strict superset of the regime package's ``(1, 3, 7)``.

The fit artifact is a single dict with this layout::

    {
        "version": str,                       # ISO timestamp
        "horizons": (1, 2, 3, 4, 5, 6, 7),
        "feature_indices": [...],
        "feature_names": [...],
        "doy_clim_temp": np.ndarray,
        "terciles": {...},
        "neighbor_stations": [...],
        "neighbor_terciles": {...},
        "regime":  {h: CalibratedClassifierCV},
        "hi":      {h: HistGradientBoostingRegressor},
        "lo":      {h: HistGradientBoostingRegressor},
        "pop":     {h: CalibratedClassifierCV},
        "pop_threshold_in": 0.10,
        "trained_through": "...",
        "calibration_window": (...),
        "test_window": (...),
        "metrics": {h: {regime: {...}, hi: {...}, lo: {...}, pop: {...}}},
    }

The trainer also writes a row to the ``model_runs`` table in
``forecast_ml.db`` so the production runner can look up the active
model.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Make the project root importable when invoked as a script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecast_ml_pkg import db as fdb  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg import continuous_models as cm  # noqa: E402  pylint: disable=wrong-import-position
from forecast_ml_pkg.io import tz_utils  # noqa: E402  pylint: disable=wrong-import-position

# weather_regime_pkg lives under D:\Scripts; the forecast_ml_pkg.io shim
# puts that on sys.path on import.
from weather_regime_pkg.trainer import (  # noqa: E402  pylint: disable=wrong-import-position
    RegimeForecastTrainer, TrainerConfig,
)
from weather_regime_pkg.features import targets_for_horizon  # noqa: E402  pylint: disable=wrong-import-position


PROJECT_ROOT = fdb.PROJECT_ROOT
ARTIFACT_DIR = PROJECT_ROOT / "data" / "model_runs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExtendedTrainerConfig:
    """Configuration mirror for ``ExtendedForecastTrainer``.

    Holds the regime-trainer config plus extended-trainer-specific knobs.
    """
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    train_end: str = "2020-12-31"
    val_start: str = "2021-01-01"
    val_end: str = "2022-12-31"
    test_start: Optional[str] = "2023-01-01"
    test_end: Optional[str] = "2025-12-31"
    neighbor_stations: tuple[str, ...] = ("KPUB", "KDEN", "KAPA")
    pop_threshold_in: float = cm.POP_THRESHOLD_IN
    classifier_kind: str = "histgbm"
    artifact_path: Path = ARTIFACT_DIR / "extended_v1.pkl"
    write_model_runs_row: bool = True
    extra_artifact_fields: dict = field(default_factory=dict)


class ExtendedForecastTrainer(RegimeForecastTrainer):
    """Train regime + Hi + Lo + PoP at each configured horizon."""

    def __init__(self, ext_config: ExtendedTrainerConfig):
        # Adapt our extended config into the parent's TrainerConfig, then
        # override only the fields we need.
        base_cfg = TrainerConfig(
            horizons=ext_config.horizons,
            train_end=ext_config.train_end,
            val_start=ext_config.val_start,
            val_end=ext_config.val_end,
            test_start=ext_config.test_start,
            test_end=ext_config.test_end,
            neighbor_stations=ext_config.neighbor_stations,
            classifier_kind=ext_config.classifier_kind,
            artifact_path=ext_config.artifact_path,
        )
        super().__init__(base_cfg)
        self.ext_config = ext_config
        self.hi_models: dict[int, object] = {}
        self.lo_models: dict[int, object] = {}
        self.pop_models: dict[int, object] = {}
        self.metrics_per_h: dict[int, dict] = {}

    # ----------------------------------------------------------------
    # Per-horizon Hi/Lo/PoP fit + evaluate
    # ----------------------------------------------------------------

    def fit_continuous_heads_for_horizon(self, h: int) -> dict:
        """Fit Hi, Lo, PoP heads for one horizon. Returns the metrics dict."""
        y_hi, y_lo, y_pop, hi_lo_mask = cm.build_continuous_targets(
            self.rows, self.valid_idx, h, self.ext_config.pop_threshold_in,
        )
        train_idx = self.train_m & hi_lo_mask
        val_idx = self.val_m & hi_lo_mask
        test_idx = self.test_m & hi_lo_mask

        X_train = self.X[train_idx]
        X_val = self.X[val_idx]
        X_test = self.X[test_idx]

        # Hi
        hi_reg = cm.fit_hi_lo_regressor(X_train, y_hi[train_idx], self.cat_indices)
        hi_metrics = cm.evaluate_hi_lo(hi_reg, X_test, y_hi[test_idx])

        # Lo
        lo_reg = cm.fit_hi_lo_regressor(X_train, y_lo[train_idx], self.cat_indices)
        lo_metrics = cm.evaluate_hi_lo(lo_reg, X_test, y_lo[test_idx])

        # PoP
        pop_cal, pop_base = cm.fit_pop_classifier(
            X_train, y_pop[train_idx], X_val, y_pop[val_idx], self.cat_indices,
        )
        pop_metrics = cm.evaluate_pop(pop_cal, pop_base, X_test, y_pop[test_idx])

        self.hi_models[h] = hi_reg
        self.lo_models[h] = lo_reg
        self.pop_models[h] = pop_cal

        return {
            "hi": hi_metrics,
            "lo": lo_metrics,
            "pop": pop_metrics,
            "n_train": int(train_idx.sum()),
            "n_val": int(val_idx.sum()),
            "n_test": int(test_idx.sum()),
        }

    # ----------------------------------------------------------------
    # Orchestration
    # ----------------------------------------------------------------

    def run(self) -> dict:  # pylint: disable=too-many-locals
        cfg = self.config
        ext_cfg = self.ext_config
        self.load_data()
        self.build_dataset()

        regime_models: dict[int, object] = {}
        regime_results: dict[int, dict] = {}

        for h in cfg.horizons:
            print(f"\n--- horizon h={h} ---")
            cal, meta = self.fit_one_horizon(h)
            regime_models[h] = cal
            r = self.evaluate(h, cal, meta)
            if r:
                regime_results[h] = r

            cont = self.fit_continuous_heads_for_horizon(h)
            self.metrics_per_h[h] = {
                "regime": _summarize_regime_metrics(r),
                "hi": cont["hi"],
                "lo": cont["lo"],
                "pop": cont["pop"],
                "n_train": cont["n_train"],
                "n_val": cont["n_val"],
                "n_test": cont["n_test"],
            }
            self._print_horizon_summary(h, self.metrics_per_h[h])

        artifact_path = self._save_combined_artifact(regime_models)
        if ext_cfg.write_model_runs_row:
            run_id = self._record_model_run_db(artifact_path)
            print(f"\nmodel_runs row written: run_id={run_id}")

        return {
            "regime_results": regime_results,
            "metrics_per_h": self.metrics_per_h,
            "artifact_path": artifact_path,
        }

    # ----------------------------------------------------------------
    # Reporting + persistence helpers
    # ----------------------------------------------------------------

    def _print_horizon_summary(self, h: int, metrics: dict) -> None:
        regime = metrics["regime"]
        hi = metrics["hi"]
        lo = metrics["lo"]
        pop_cal = metrics["pop"].get("calibrated", {})
        print(
            f"  h={h}: regime top1={regime.get('acc', float('nan')):.3f} "
            f"Brier={regime.get('brier', float('nan')):.3f} "
            f"ECE={regime.get('ece', float('nan')):.3f} | "
            f"Hi MAE={hi.get('mae', float('nan')):.2f}F "
            f"Lo MAE={lo.get('mae', float('nan')):.2f}F | "
            f"PoP Brier={pop_cal.get('brier', float('nan')):.3f} "
            f"ECE={pop_cal.get('ece', float('nan')):.3f}"
        )

    def _save_combined_artifact(self, regime_models: dict[int, object]) -> Path:
        cfg = self.config
        ext_cfg = self.ext_config
        artifact = {
            "version": tz_utils.now_utc(),
            "horizons": list(cfg.horizons),
            "feature_indices": self.cat_indices,
            "feature_names": self.feature_names,
            "doy_clim_temp": self.doy_clim_temp,
            "terciles": self.terciles,
            "neighbor_stations": list(cfg.neighbor_stations),
            "neighbor_terciles": self.neighbor_terciles,
            "regime": regime_models,
            "hi": self.hi_models,
            "lo": self.lo_models,
            "pop": self.pop_models,
            "pop_threshold_in": ext_cfg.pop_threshold_in,
            "trained_through": cfg.train_end,
            "calibration_window": (cfg.val_start, cfg.val_end),
            "test_window": (cfg.test_start, cfg.test_end) if cfg.test_start else None,
            "metrics": self.metrics_per_h,
            "classifier_kind": cfg.classifier_kind,
        }
        artifact.update(ext_cfg.extra_artifact_fields)

        ext_cfg.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ext_cfg.artifact_path, "wb") as fh:
            pickle.dump(artifact, fh)
        size_mb = ext_cfg.artifact_path.stat().st_size / 1024 / 1024
        print(f"\nCombined artifact saved: {ext_cfg.artifact_path} ({size_mb:.1f} MB)")
        return ext_cfg.artifact_path

    def _record_model_run_db(self, artifact_path: Path) -> str:
        cfg = self.config
        ext_cfg = self.ext_config
        run_id = uuid.uuid4().hex[:12]
        model_version = f"extended-{ext_cfg.classifier_kind}-{tz_utils.now_utc()[:10]}"

        hyperparams = {
            "regime_params": (
                "tuned" if cfg.classifier_kind == "histgbm_tuned" else "default"
            ),
            "hi_lo_params": cm.HI_LO_PARAMS,
            "pop_params": cm.POP_PARAMS,
            "horizons": list(cfg.horizons),
            "neighbor_stations": list(cfg.neighbor_stations),
            "pop_threshold_in": ext_cfg.pop_threshold_in,
        }
        val_metrics = {
            str(h): {
                "regime_acc": metrics["regime"].get("acc"),
                "regime_brier": metrics["regime"].get("brier"),
                "regime_ece": metrics["regime"].get("ece"),
                "hi_mae": metrics["hi"].get("mae"),
                "lo_mae": metrics["lo"].get("mae"),
                "pop_brier": metrics["pop"].get("calibrated", {}).get("brier"),
                "pop_ece": metrics["pop"].get("calibrated", {}).get("ece"),
            }
            for h, metrics in self.metrics_per_h.items()
        }

        conn = fdb.get_connection()
        try:
            with conn:
                conn.execute("UPDATE model_runs SET is_active = 0")
                conn.execute(
                    """
                    INSERT INTO model_runs (
                        run_id, model_version, training_start_date, training_end_date,
                        val_start_date, val_end_date,
                        hyperparams_json, val_metrics_json, feature_names_json,
                        artifact_path, created_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        run_id, model_version,
                        "1948-04-01",  # ACIS earliest
                        cfg.train_end,
                        cfg.val_start, cfg.val_end,
                        json.dumps(hyperparams, default=str),
                        json.dumps(val_metrics),
                        json.dumps(self.feature_names),
                        str(artifact_path),
                        tz_utils.now_utc(),
                    ),
                )
        finally:
            conn.close()
        return run_id


def _summarize_regime_metrics(eval_result: Optional[dict]) -> dict:
    if not eval_result:
        return {}
    ml_cal = eval_result.get("ml_cal", {})
    return {
        "acc": ml_cal.get("acc"),
        "brier": ml_cal.get("brier"),
        "logloss": ml_cal.get("logloss"),
        "ece": ml_cal.get("ece"),
    }


# Suppress unused-import lint nag: targets_for_horizon is re-exported
# for any caller that wants it without poking at weather_regime_pkg.
__all__ = [
    "ExtendedTrainerConfig",
    "ExtendedForecastTrainer",
    "targets_for_horizon",
]
