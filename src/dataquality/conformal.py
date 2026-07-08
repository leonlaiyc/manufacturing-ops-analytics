"""
M11 Stage B - split conformal prediction intervals for the M7 virtual
metrology (VM) model.

Split conformal prediction (Vovk et al.; Papadopoulos et al.) wraps ANY point
predictor with a distribution-free coverage guarantee, using nothing beyond
sorting and a quantile: no library, hand-built with numpy, consistent with
the project's "interpretable, first-principles methods only" rule.

Method
------
1. TRAIN split: fit the point predictor (``virtual_metrology.fit_ols``,
   unchanged from M7) on the earliest lots.
2. CALIBRATION split: a SEPARATE, later slice of lots the model never saw
   during fitting. Compute the model's absolute residual on each calibration
   lot: ``r_i = |y_i - yhat_i|``.
3. Nominal miscoverage ``alpha`` (default 0.10, i.e. 90% nominal coverage).
   The conformal quantile is the ``ceil((n_cal + 1) * (1 - alpha)) / n_cal``
   empirical quantile of the calibration residuals ``{r_i}`` (the standard
   finite-sample-correct split-conformal quantile: the ``+1`` and ceiling
   are what make the coverage guarantee hold at exactly this sample size,
   not just asymptotically).
4. Interval for any new point: ``[yhat - q_alpha, yhat + q_alpha]`` (constant
   half-width, since the calibration step used raw absolute residuals, not a
   normalized/studentized score; this is the plain split-conformal baseline,
   the simplest version of the method).
5. TEST split: a THIRD, still later slice, used only to MEASURE empirical
   coverage (fraction of test points whose true y falls inside the interval)
   and interval width. The test split never touches steps 1-3.

Why calibration must not overlap training
------------------------------------------
The coverage guarantee comes from calibration residuals being EXCHANGEABLE
with the test residuals under the fitted model. If a lot were used both to
fit the model and to compute its own residual, that residual would be
systematically too small (the model was optimized to fit it), so the
resulting q_alpha would UNDERSTATE the true residual spread and the interval
would undercover on genuinely held-out lots. This mirrors the M7 VM
docstring's leakage-safety argument (train/test) one level further: here
train/calibration/test are three DISJOINT, time-ordered slices of the same
"never re-use a fitting lot's own residual" principle.

Split is time-ordered (by ``completion_time``, the same key
``virtual_metrology.time_split`` uses), consistent with the project's stance
that a random split would leak shared tool-state/queue dynamics across
nearby lots in time; splitting three ways in time order keeps calibration
and test both strictly in the model's simulated future relative to training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from virtual_metrology import FEATURE_COLUMNS, fit_ols, predict

DEFAULT_ALPHA = 0.10   # nominal miscoverage -> 90% nominal coverage
DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_CAL_FRACTION = 0.20
# remaining fraction (0.20 by default) is the held-out test split


def three_way_time_split(feat: pd.DataFrame,
                          train_fraction: float = DEFAULT_TRAIN_FRACTION,
                          cal_fraction: float = DEFAULT_CAL_FRACTION
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by ``completion_time`` and split into train / calibration / test.

    Train = first ``train_fraction``; calibration = next ``cal_fraction``;
    test = the remainder. All three are disjoint, time-ordered slices (see
    module docstring for why calibration must not overlap training).
    """
    ordered = feat.sort_values("completion_time", kind="mergesort")
    n = len(ordered)
    n_train = int(round(n * train_fraction))
    n_cal = int(round(n * cal_fraction))
    train = ordered.iloc[:n_train]
    cal = ordered.iloc[n_train:n_train + n_cal]
    test = ordered.iloc[n_train + n_cal:]
    return train, cal, test


def conformal_quantile(residuals: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Finite-sample-correct split-conformal quantile of absolute residuals.

    q_alpha = the k-th smallest of the n calibration residuals, sorted
    ascending, where k = ceil((n + 1) * (1 - alpha)), clipped to n (so with a
    small calibration set the quantile falls back to the maximum residual,
    the most conservative interval available rather than an out-of-range
    index).
    """
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(r[k - 1])


@dataclass
class ConformalResult:
    """Fitted point model + calibrated conformal half-width + test evaluation."""
    coef: np.ndarray
    q_alpha: float
    alpha: float
    n_train: int
    n_cal: int
    n_test: int
    test_coverage: float       # empirical fraction of test points inside the interval
    test_mean_width: float     # 2 * q_alpha (constant for split conformal)
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def fit_and_calibrate(log: pd.DataFrame, route: list, lot_quality: pd.DataFrame,
                       window_hours: float, alpha: float = DEFAULT_ALPHA,
                       train_fraction: float = DEFAULT_TRAIN_FRACTION,
                       cal_fraction: float = DEFAULT_CAL_FRACTION,
                       feature_columns: list = FEATURE_COLUMNS) -> ConformalResult:
    """End-to-end: build features via ``virtual_metrology.build_features``,
    three-way time split, fit OLS on train, calibrate q_alpha on calibration,
    evaluate coverage and width on test.
    """
    from virtual_metrology import build_features

    feat = build_features(log, route, lot_quality, window_hours)
    train, cal, test = three_way_time_split(feat, train_fraction, cal_fraction)

    coef, _ = fit_ols(train, feature_columns)

    cal_pred = predict(coef, cal, feature_columns)
    cal_true = cal["metrology_reading"].to_numpy(dtype=float)
    cal_resid = np.abs(cal_true - cal_pred)
    q_alpha = conformal_quantile(cal_resid, alpha)

    y_test_true = test["metrology_reading"].to_numpy(dtype=float)
    y_test_pred = predict(coef, test, feature_columns)
    lower = y_test_pred - q_alpha
    upper = y_test_pred + q_alpha
    inside = (y_test_true >= lower) & (y_test_true <= upper)
    test_coverage = float(inside.mean()) if len(inside) else float("nan")
    test_mean_width = float(2 * q_alpha)

    return ConformalResult(
        coef=coef, q_alpha=q_alpha, alpha=alpha,
        n_train=len(train), n_cal=len(cal), n_test=len(test),
        test_coverage=test_coverage, test_mean_width=test_mean_width,
        y_test_true=y_test_true, y_test_pred=y_test_pred,
        lower=lower, upper=upper,
    )
