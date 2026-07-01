"""
Detection-quality evaluation (M5, Step 4) — the core of M5.

Because the synthetic anomalies are injected with known timestamps, we can score
the detectors objectively:

  - detection delay = first in-window alarm − the anomaly's INJECTION time
    (ground truth). Cleanest possible reference.
  - lead time = the actual KPI first crossing an INDEPENDENT PHYSICAL threshold
    − the first alarm. The physical threshold is the normal operating level
    scaled by a fixed factor (default: clean-baseline median × 1.5). It is defined
    from actual KPI values vs the normal level and NEVER from the detector's own
    statistics / sigma — otherwise lead time would be the detector compared to
    itself (circular). Positive lead = the detector fired before the KPI became
    physically obvious.
  - false-alarm rate = alarms on clean days / clean days.
  - precision / recall over all injected anomalies (across replications).

Units are days (the KPI series is daily). Anomalies raise cycle time / WIP /
bottleneck wait, so evaluation is written for an "increase" KPI; a grace period
after each window absorbs the queue-drain tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 24.0


def _day_windows(windows: list, grace_days: int) -> list:
    """Convert [{t_start,t_end,type}] to day ranges [lo, hi] with a grace tail."""
    out = []
    for w in windows:
        lo = int(np.floor(w["t_start"] / DAY))
        hi = int(np.ceil(w["t_end"] / DAY)) + grace_days
        out.append({"type": w["type"], "inj_day": lo, "lo": lo, "hi": hi})
    return out


def evaluate_single(
    detected: pd.DataFrame,
    windows: list,
    series: pd.Series,
    baseline_days,
    grace_days: int = 12,
    impact_factor: float = 1.5,
    eval_start: int = 6,
) -> dict:
    """Score one detector run against the injected windows of one replication.

    The false-alarm rate is measured on HELD-OUT clean days only: days that are
    warm-up (< ``eval_start``) or part of the baseline-fit window are excluded
    from both the clean-day count and the false-alarm count, so FAR reflects clean
    days the detector was not centred on. Returns:
      per_anomaly : list of {type, inj_day, detected, delay_days, impact_day, lead_days}
      counts      : {detected_anomalies, total_anomalies, in_window_alarms,
                     false_alarms, clean_days}
    """
    dwins = _day_windows(windows, grace_days)
    alarms = list(detected.index[detected["alarm"].to_numpy(dtype=bool)])
    excluded = set(int(d) for d in baseline_days) | set(range(0, eval_start))

    def in_any(day):
        return any(w["lo"] <= day <= w["hi"] for w in dwins)

    # Physical impact threshold (independent of the detector): normal level x factor.
    normal = float(series.reindex(baseline_days).dropna().median())
    threshold = normal * impact_factor

    per_anomaly = []
    for w in dwins:
        win_alarms = [d for d in alarms if w["lo"] <= d <= w["hi"]]
        first_alarm = min(win_alarms) if win_alarms else None
        # first day at/after injection where the ACTUAL KPI crosses the physical bar
        after = series[(series.index >= w["inj_day"]) & (series.index <= w["hi"])]
        crossed = after[after > threshold]
        impact_day = int(crossed.index[0]) if len(crossed) else None
        delay = (first_alarm - w["inj_day"]) if first_alarm is not None else None
        lead = (impact_day - first_alarm) if (first_alarm is not None and impact_day is not None) else None
        per_anomaly.append({
            "type": w["type"],
            "inj_day": w["inj_day"],
            "detected": first_alarm is not None,
            "delay_days": delay,
            "impact_day": impact_day,
            "lead_days": lead,
        })

    clean_days = [d for d in series.index if not in_any(d) and d not in excluded]
    false_alarms = [d for d in alarms if not in_any(d) and d not in excluded]
    in_window_alarms = [d for d in alarms if in_any(d)]

    return {
        "per_anomaly": per_anomaly,
        "counts": {
            "detected_anomalies": sum(a["detected"] for a in per_anomaly),
            "total_anomalies": len(dwins),
            "in_window_alarms": len(in_window_alarms),
            "false_alarms": len(false_alarms),
            "clean_days": len(clean_days),
        },
        "threshold": threshold,
    }


def aggregate(singles: list) -> dict:
    """Aggregate per-replication results into overall detection-quality metrics."""
    tp_anom = sum(s["counts"]["detected_anomalies"] for s in singles)
    tot_anom = sum(s["counts"]["total_anomalies"] for s in singles)
    inwin = sum(s["counts"]["in_window_alarms"] for s in singles)
    fa = sum(s["counts"]["false_alarms"] for s in singles)
    clean = sum(s["counts"]["clean_days"] for s in singles)
    total_alarms = inwin + fa

    delays = [a["delay_days"] for s in singles for a in s["per_anomaly"]
              if a["delay_days"] is not None]
    leads = [a["lead_days"] for s in singles for a in s["per_anomaly"]
             if a["lead_days"] is not None]

    return {
        "recall": tp_anom / tot_anom if tot_anom else float("nan"),
        "precision": inwin / total_alarms if total_alarms else float("nan"),
        "false_alarm_rate": fa / clean if clean else float("nan"),
        "mean_detection_delay_days": float(np.mean(delays)) if delays else float("nan"),
        "mean_lead_days": float(np.mean(leads)) if leads else float("nan"),
        "n_anomalies": tot_anom,
        "n_detected": tp_anom,
        "n_false_alarms": fa,
        "n_clean_days": clean,
    }


def per_anomaly_table(singles: list) -> pd.DataFrame:
    """Flatten per-anomaly rows across replications for inspection/plots."""
    rows = []
    for r, s in enumerate(singles):
        for a in s["per_anomaly"]:
            rows.append({"rep": r, **a})
    return pd.DataFrame(rows)


def sensitivity_sweep(series_list, baseline_list, windows_list, detector_fn,
                      params, param_name) -> pd.DataFrame:
    """Sweep a detector sensitivity parameter and record FAR + detection delay.

    ``detector_fn(series, center, sigma, <param_name>=value)`` builds the detector
    DataFrame; center/sigma come from each replication's clean baseline. Returns a
    DataFrame with columns [param, recall, precision, false_alarm_rate,
    mean_detection_delay_days].
    """
    from detectors import fit_baseline
    out = []
    for val in params:
        singles = []
        for series, base_days, windows in zip(series_list, baseline_list, windows_list):
            center, sigma = fit_baseline(series, base_days)
            det = detector_fn(series, center, sigma, **{param_name: val})
            singles.append(evaluate_single(det, windows, series, base_days))
        agg = aggregate(singles)
        out.append({
            "param": val,
            "recall": agg["recall"],
            "precision": agg["precision"],
            "false_alarm_rate": agg["false_alarm_rate"],
            "mean_detection_delay_days": agg["mean_detection_delay_days"],
        })
    return pd.DataFrame(out)
