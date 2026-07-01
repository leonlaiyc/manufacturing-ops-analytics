"""
Statistical anomaly detectors (M5, Step 3).

Two simple, explainable detectors — no black-box models:

  - Control chart (Shewhart): flag any day where the KPI leaves
    ``center ± k·sigma``. Sensitive to sudden shifts (breakdown, demand surge).
  - EWMA: exponentially-weighted moving average ``z_t = λ·x_t + (1−λ)·z_{t−1}``
    with the standard time-varying control limits. Sensitive to small, persistent
    offsets — the slow drift (gradual degradation) a Shewhart chart is slow to catch.

They are complementary and both fully interpretable from first principles.

**Leakage-free baseline.** ``center`` and ``sigma`` are estimated ONLY from a
clean, pre-anomaly window (``fit_baseline``). No future or anomalous data enters
the baseline, so the detector never "sees" the anomaly it is meant to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_baseline(series: pd.Series, baseline_days) -> tuple[float, float]:
    """Estimate (center, sigma) from clean pre-anomaly days only (no leakage).

    ``baseline_days`` is an iterable of day indices known to be anomaly-free.
    """
    x = series.reindex(baseline_days).dropna()
    return float(x.mean()), float(x.std(ddof=1))


def control_chart(series: pd.Series, center: float, sigma: float,
                  k: float = 3.0) -> pd.DataFrame:
    """Shewhart chart: alarm when the day's value leaves center ± k·sigma.

    Returns a DataFrame indexed like ``series`` with columns
    value, center, ucl, lcl, alarm (bool). NaN days never alarm.
    """
    ucl = center + k * sigma
    lcl = center - k * sigma
    val = series.astype(float)
    alarm = (val > ucl) | (val < lcl)
    alarm = alarm.fillna(False)
    return pd.DataFrame({
        "value": val,
        "center": center,
        "ucl": ucl,
        "lcl": lcl,
        "alarm": alarm,
    }, index=series.index)


def ewma_chart(series: pd.Series, center: float, sigma: float,
               lam: float = 0.2, L: float = 3.0) -> pd.DataFrame:
    """EWMA chart with the standard time-varying control limits.

    z_t = lam·x_t + (1-lam)·z_{t-1}, z_0 = center.
    limit_t = L·sigma·sqrt( (lam/(2-lam))·(1-(1-lam)^{2t}) ).
    NaN days carry the previous z forward and do not alarm.
    Returns DataFrame with columns value, ewma, center, ucl, lcl, alarm.
    """
    z = center
    ewma_vals, ucls, lcls, alarms = [], [], [], []
    t = 0
    for x in series.astype(float):
        if np.isnan(x):
            ewma_vals.append(z)
            # limits still widen with t; keep last computed step count
            factor = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (t + 1)))
            limit = L * sigma * np.sqrt(factor)
            ucls.append(center + limit)
            lcls.append(center - limit)
            alarms.append(False)
            continue
        t += 1
        z = lam * x + (1 - lam) * z
        factor = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t))
        limit = L * sigma * np.sqrt(factor)
        ucl = center + limit
        lcl = center - limit
        ewma_vals.append(z)
        ucls.append(ucl)
        lcls.append(lcl)
        alarms.append(bool(z > ucl or z < lcl))
    return pd.DataFrame({
        "value": series.astype(float).values,
        "ewma": ewma_vals,
        "center": center,
        "ucl": ucls,
        "lcl": lcls,
        "alarm": alarms,
    }, index=series.index)


def alarm_days(detected: pd.DataFrame) -> list:
    """Day indices where the detector fired."""
    return list(detected.index[detected["alarm"].to_numpy(dtype=bool)])
