"""Hi / Lo / PoP forecasting heads.

In addition to the 9-state regime classifier from ``weather_regime_pkg``,
the production project predicts three continuous-style targets per
horizon:

    * Tmax (Hi) -- HistGradientBoostingRegressor, squared_error loss
    * Tmin (Lo) -- HistGradientBoostingRegressor, squared_error loss
    * PoP      -- HistGradientBoostingClassifier, calibrated, target = 1
                  if precip_t+h > POP_THRESHOLD_IN

The same daily feature matrix and train/val/test splits used for the
regime classifier are reused here, so callers fit all four heads from
one X and one set of valid_idx / split masks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator

POP_THRESHOLD_IN: float = 0.10  # Plan calls for "precip > 0.10 in" definition

HI_LO_PARAMS: dict = {
    "loss": "squared_error",
    "max_depth": 6,
    "max_iter": 400,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 15,
}

POP_PARAMS: dict = {
    "max_depth": 6,
    "max_iter": 400,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 15,
}


def build_continuous_targets(
    rows: list[dict],
    valid_idx: np.ndarray,
    horizon: int,
    pop_threshold: float = POP_THRESHOLD_IN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each row in ``valid_idx`` build (y_hi, y_lo, y_pop, mask) at h.

    The ``mask`` is True iff the +h target row exists, is contiguous, and
    has all three of (maxt, mint, pcpn) populated.
    """
    n = len(valid_idx)
    y_hi = np.full(n, np.nan)
    y_lo = np.full(n, np.nan)
    y_pop = np.full(n, 0, dtype=int)
    mask = np.zeros(n, dtype=bool)

    for j, i in enumerate(valid_idx):
        ti = i + horizon
        if ti >= len(rows):
            continue
        anchor = rows[i]
        target = rows[ti]
        d_diff = (
            datetime.strptime(target["date"], "%Y-%m-%d")
            - datetime.strptime(anchor["date"], "%Y-%m-%d")
        ).days
        if d_diff != horizon:
            continue
        if (target.get("maxt") is None or target.get("mint") is None
                or target.get("pcpn") is None):
            continue
        y_hi[j] = float(target["maxt"])
        y_lo[j] = float(target["mint"])
        y_pop[j] = int(float(target["pcpn"]) > pop_threshold)
        mask[j] = True

    return y_hi, y_lo, y_pop, mask


def fit_hi_lo_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cat_indices: Iterable[int],
) -> HistGradientBoostingRegressor:
    """Fit one Hi-or-Lo regressor on the training fold."""
    reg = HistGradientBoostingRegressor(
        categorical_features=list(cat_indices),
        **HI_LO_PARAMS,
    )
    reg.fit(X_train, y_train)
    return reg


def fit_pop_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_indices: Iterable[int],
) -> tuple[CalibratedClassifierCV, HistGradientBoostingClassifier]:
    """Fit a binary PoP classifier and isotonic-calibrate it on the val fold.

    Returns ``(calibrated, base_uncalibrated)`` so callers can compare both.
    """
    base = HistGradientBoostingClassifier(
        categorical_features=list(cat_indices),
        **POP_PARAMS,
    )
    base.fit(X_train, y_train)

    # Need at least two classes in y_val for isotonic calibration; fall back
    # to identity if the val fold happened to be all-dry.
    if len(np.unique(y_val)) < 2:
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic", cv=2)
        calibrated.fit(X_train, y_train)  # use train if val is degenerate
    else:
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic", cv=5)
        calibrated.fit(X_val, y_val)
    return calibrated, base


def regression_metrics(
    pred: np.ndarray, true: np.ndarray
) -> dict[str, float]:
    """MAE, RMSE, bias for a Hi/Lo head."""
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "n": int(len(true)),
    }


def pop_metrics(
    p_pred: np.ndarray, true: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Brier score, log-loss, ECE, and base rate for the PoP head."""
    eps = 1e-9
    p = np.clip(p_pred, eps, 1 - eps)
    brier = float(np.mean((p - true) ** 2))
    ll = float(-np.mean(true * np.log(p) + (1 - true) * np.log(1 - p)))

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(true)
    for k in range(n_bins):
        in_bin = (p >= bins[k]) & (p < bins[k + 1] if k < n_bins - 1 else p <= bins[k + 1])
        if in_bin.sum() == 0:
            continue
        avg_p = float(np.mean(p[in_bin]))
        avg_y = float(np.mean(true[in_bin]))
        ece += in_bin.sum() / total * abs(avg_p - avg_y)

    return {
        "brier": brier,
        "logloss": ll,
        "ece": float(ece),
        "base_rate": float(np.mean(true)),
        "n": int(total),
    }


def evaluate_hi_lo(
    reg, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    if len(y_test) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": 0}
    return regression_metrics(reg.predict(X_test), y_test)


def evaluate_pop(
    calibrated, base, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, dict[str, float]]:
    if len(y_test) == 0:
        empty = {"brier": float("nan"), "logloss": float("nan"),
                 "ece": float("nan"), "base_rate": float("nan"), "n": 0}
        return {"calibrated": empty, "uncalibrated": empty}

    cal_p = _binary_proba(calibrated, X_test)
    raw_p = _binary_proba(base, X_test)
    return {
        "calibrated": pop_metrics(cal_p, y_test),
        "uncalibrated": pop_metrics(raw_p, y_test),
    }


def _binary_proba(clf, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) regardless of which column sklearn places it in."""
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    # Edge case: all-zero training -> predict the only column as zero-prob.
    return np.zeros(proba.shape[0])


__all__ = [
    "POP_THRESHOLD_IN",
    "build_continuous_targets",
    "fit_hi_lo_regressor",
    "fit_pop_classifier",
    "evaluate_hi_lo",
    "evaluate_pop",
    "regression_metrics",
    "pop_metrics",
]
