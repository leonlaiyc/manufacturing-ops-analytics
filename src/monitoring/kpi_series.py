"""
Daily KPI time series for anomaly monitoring (M5, Step 2).

Turns a simulation run into per-day KPI series that the detectors watch:
  - throughput  : lots completed per day
  - cycle_time  : daily median lot cycle time (hours)
  - wip         : daily time-average work-in-process
  - bn_wait     : daily mean queue wait in front of the bottleneck station (hours)

Reuses the M3 helpers where they already exist
(``kpi.kpi_metrics``: ``daily_throughput``, ``daily_median_ct``,
``wip_timeseries``); adds the two aggregations M3 did not expose (daily mean WIP
and daily bottleneck wait). Consumers must have ``src`` on ``sys.path`` so
``kpi.kpi_metrics`` imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kpi.kpi_metrics import daily_throughput, daily_median_ct, wip_timeseries

DAY = 24.0


def daily_mean_wip(lifecycle: pd.DataFrame, horizon: float) -> pd.Series:
    """Time-average WIP per day by integrating the WIP step function over each day."""
    times, level = wip_timeseries(lifecycle)
    times = np.asarray(times, dtype=float)
    level = np.asarray(level, dtype=float)
    seg_end = np.concatenate([times[1:], [horizon]]) if len(times) else np.array([])
    days = int(np.ceil(horizon / DAY))
    out = np.zeros(days)
    for d in range(days):
        a, b = d * DAY, (d + 1) * DAY
        lo = np.maximum(times, a)
        hi = np.minimum(seg_end, b)
        dur = np.clip(hi - lo, 0.0, None)
        out[d] = (level * dur).sum() / (b - a)
    return pd.Series(out, index=range(days), name="wip")


def daily_bottleneck_wait(log: pd.DataFrame, station: str, horizon: float) -> pd.Series:
    """Daily mean queue wait (process_start - queue_entry) at ``station``.

    Ops are bucketed by the day their processing completes. For a re-entrant
    station both visits per lot are included.
    """
    ops = log[log["station"] == station].copy()
    ops["wait"] = ops["process_start_time"] - ops["queue_entry_time"]
    ops["day"] = np.floor(ops["process_complete_time"] / DAY).astype(int)
    s = ops.groupby("day")["wait"].mean()
    s.name = "bn_wait"
    return s


def daily_kpis(log: pd.DataFrame, lifecycle: pd.DataFrame, cfg,
               bottleneck: str = "S4") -> pd.DataFrame:
    """Assemble the daily KPI panel over the whole run, indexed by day 0..N-1.

    Days with no completed lots are left as NaN (so the caller can decide how to
    treat idle days); the index is dense over ``[0, horizon/24)``.
    """
    t0, t1 = 0.0, cfg.horizon_hours
    thr = daily_throughput(lifecycle, t0, t1).set_index("day")["count"]
    ct = daily_median_ct(lifecycle, t0, t1).set_index("day")["median_ct"]
    wip = daily_mean_wip(lifecycle, t1)
    bn = daily_bottleneck_wait(log, bottleneck, t1)

    days = range(int(np.ceil(t1 / DAY)))
    df = pd.DataFrame(index=days)
    df["throughput"] = thr
    df["cycle_time"] = ct
    df["wip"] = wip
    df["bn_wait"] = bn
    df.index.name = "day"
    return df
