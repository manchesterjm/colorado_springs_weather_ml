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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from forecast_ml_pkg import db as fdb
from forecast_ml_pkg import continuous_models as cm
from forecast_ml_pkg import nws_features as nwsf
from forecast_ml_pkg.externals import tz_utils

# weather_regime_pkg lives under D:\Scripts; forecast_ml_pkg/__init__.py
# puts that directory on sys.path at import time.
from weather_regime_pkg.trainer import (
    RegimeForecastTrainer, TrainerConfig,
)
from weather_regime_pkg.features import targets_for_horizon

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
    # Phase 7 (NWS-as-feature). When True, appends 28 NWS columns to the
    # autoregressive feature matrix: 7 Hi + 7 Lo + 7 PoP + 7 availability
    # indicators. The model version label gains a "-nwsfeat" suffix so the
    # registry can host both variants side-by-side.
    use_nws_features: bool = False


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
        # Phase 7 climatology arrays (populated by augment_with_nws_features).
        self.nws_hi_clim: Optional[np.ndarray] = None
        self.nws_lo_clim: Optional[np.ndarray] = None
        # Built by the parent's build_dataset(); augment_with_nws_features()
        # may extend them. Declared here so they are not assigned only
        # "outside __init__".
        self.X: Optional[np.ndarray] = None
        self.feature_names: list = []

    # ----------------------------------------------------------------
    # Per-horizon Hi/Lo/PoP fit + evaluate
    # ----------------------------------------------------------------

    def fit_continuous_heads_for_horizon(self, h: int) -> dict:
        """Fit Hi, Lo, PoP heads for one horizon. Returns the metrics dict.

        Scores are computed on the held-out **test** window when one is
        configured. Production retrain runs with ``test_start=None`` (no test
        split, so the active model calibrates on the widest recent window), and
        in that mode the heads are scored on the **validation/calibration**
        window instead -- otherwise the metrics are taken over an empty array
        and come out NaN, which is what previously left ``val_metrics_json``
        unusable. Hi/Lo are genuinely held out (only the train split is fit);
        PoP is calibrated on the val window, so its val-fallback Brier/ECE are
        mildly optimistic in absolute terms but stay apples-to-apples across the
        baseline/nwsfeat A/B variants (both scored the same way). The model
        artifact itself is unaffected -- this only governs what gets scored.
        """
        y_hi, y_lo, y_pop, hi_lo_mask = cm.build_continuous_targets(
            self.rows, self.valid_idx, h, self.ext_config.pop_threshold_in,
        )
        train_idx = self.train_m & hi_lo_mask
        val_idx = self.val_m & hi_lo_mask
        test_idx = self.test_m & hi_lo_mask

        # Prefer the held-out test window; fall back to val when none exists.
        eval_idx = test_idx if int(test_idx.sum()) > 0 else val_idx

        X_train = self.X[train_idx]
        X_val = self.X[val_idx]
        X_eval = self.X[eval_idx]

        # Hi
        hi_reg = cm.fit_hi_lo_regressor(X_train, y_hi[train_idx], self.cat_indices)
        hi_metrics = cm.evaluate_hi_lo(hi_reg, X_eval, y_hi[eval_idx])

        # Lo
        lo_reg = cm.fit_hi_lo_regressor(X_train, y_lo[train_idx], self.cat_indices)
        lo_metrics = cm.evaluate_hi_lo(lo_reg, X_eval, y_lo[eval_idx])

        # PoP
        pop_cal, pop_base = cm.fit_pop_classifier(
            X_train, y_pop[train_idx], X_val, y_pop[val_idx], self.cat_indices,
        )
        pop_metrics = cm.evaluate_pop(pop_cal, pop_base, X_eval, y_pop[eval_idx])

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
            "n_eval": int(eval_idx.sum()),
            "eval_on": "test" if int(test_idx.sum()) > 0 else "val",
        }

    def _regime_metrics_on_val(self, h: int, cal, meta: dict) -> dict:
        """Score the regime classifier on the validation window.

        The parent ``evaluate`` only scores against a configured test window and
        returns ``{}`` otherwise; production retrain has no test window, so we
        mirror its calibrated-probability scoring on the val split to keep the
        regime metrics populated instead of null. Returns ``{"ml_cal": {...}}``
        to match what ``_summarize_regime_metrics`` reads, or ``{}`` if val is
        empty. Like the PoP head, the calibrator is fit on val, so these regime
        scores are mildly optimistic but symmetric across A/B variants.
        """
        _, mask = targets_for_horizon(self.rows, self.valid_idx, h)
        val_idx = self.val_m & mask
        if int(val_idx.sum()) == 0:
            return {}
        y = meta["y"]
        x_val, y_val = self.X[val_idx], y[val_idx]
        probs_cal = self._expand_classes(
            cal.predict_proba(x_val), cal.classes_, len(y_val),
        )
        return {"ml_cal": self._scores(probs_cal, y_val)}

    # ----------------------------------------------------------------
    # Orchestration
    # ----------------------------------------------------------------

    def augment_with_nws_features(self) -> None:
        """Append the 28 NWS columns to ``self.X`` and ``self.feature_names``.

        Climatology is computed from training-period rows only (rows whose date
        is <= ``cfg.train_end``) to prevent leakage. Coverage stats are logged
        so downstream callers know how many anchors fell back to imputation.
        """
        cfg = self.config
        train_rows = [r for r in self.rows if r["date"] <= cfg.train_end]
        hi_clim, lo_clim = nwsf.doy_hi_lo_climatology(train_rows)
        self.nws_hi_clim = hi_clim
        self.nws_lo_clim = lo_clim

        anchor_dates = [self.rows[i]["date"] for i in self.valid_idx]
        if anchor_dates:
            lookup = nwsf.load_mex_lookup(
                start_anchor=min(anchor_dates),
                end_anchor=max(anchor_dates),
            )
        else:
            lookup = {}

        nws_X, n_real, n_imp = nwsf.build_nws_feature_matrix(
            self.rows, self.valid_idx, lookup, hi_clim, lo_clim,
        )
        self.X = np.hstack([self.X, nws_X])
        self.feature_names = list(self.feature_names) + nwsf.nws_feature_names()
        n_total = len(self.valid_idx)
        coverage = (n_real / n_total) if n_total else 0.0
        print(
            f"NWS features appended: +{nws_X.shape[1]} cols; "
            f"anchors covered={n_real}/{n_total} ({coverage:.1%}), "
            f"any-imputation rows={n_imp}"
        )

    def run(self) -> dict:  # pylint: disable=too-many-locals
        cfg = self.config
        ext_cfg = self.ext_config
        self.load_data()
        self.build_dataset()
        if ext_cfg.use_nws_features:
            self.augment_with_nws_features()

        regime_models: dict[int, object] = {}
        regime_results: dict[int, dict] = {}

        for h in cfg.horizons:
            print(f"\n--- horizon h={h} ---")
            cal, meta = self.fit_one_horizon(h)
            regime_models[h] = cal
            r = self.evaluate(h, cal, meta)
            if not r:
                # No test window (production retrain): score regime on val so
                # the regime metrics are populated rather than null.
                r = self._regime_metrics_on_val(h, cal, meta)
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
                "n_eval": cont["n_eval"],
                "eval_on": cont["eval_on"],
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
            "use_nws_features": ext_cfg.use_nws_features,
            "nws_hi_clim": self.nws_hi_clim,
            "nws_lo_clim": self.nws_lo_clim,
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
        suffix = "-nwsfeat" if ext_cfg.use_nws_features else ""
        variant = "nwsfeat" if ext_cfg.use_nws_features else "baseline"
        model_version = (
            f"extended-{ext_cfg.classifier_kind}{suffix}-{tz_utils.now_utc()[:10]}"
        )

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
                # Demote only the current variant; the other variant (if any)
                # stays active so A/B keeps running.
                conn.execute(
                    "UPDATE model_runs SET is_active = 0 WHERE variant = ?",
                    (variant,),
                )
                conn.execute(
                    """
                    INSERT INTO model_runs (
                        run_id, model_version, training_start_date, training_end_date,
                        val_start_date, val_end_date,
                        hyperparams_json, val_metrics_json, feature_names_json,
                        artifact_path, created_at, is_active, variant
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
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
                        variant,
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
