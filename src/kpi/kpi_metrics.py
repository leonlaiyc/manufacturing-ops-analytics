"""
kpi_metrics.py — KPI computation helpers for M3 dashboard.

All functions operate on steady-state data only; callers pass t0/t1 window bounds
read from metadata.json. No Plotly dependency here — pure pandas/numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_throughput(lifecycle: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """Count lots completed per simulation day within [t0, t1].

    Returns a DataFrame with columns: day (int, simulation day index), count (int).
    Day index is floor(completion_time / 24), matching the simulator's hour unit.
    """
    mask = lifecycle["completion_time"].between(t0, t1)
    df = lifecycle.loc[mask].copy()
    df["day"] = np.floor(df["completion_time"] / 24).astype(int)
    return df.groupby("day", as_index=False).size().rename(columns={"size": "count"})


def wip_timeseries(lifecycle: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build WIP step-function from arrival (+1) and completion (-1) events.

    Uses the full lifecycle (all rows, no window filter) so the warm-up transient
    is visible. Returns (times, wip) as parallel 1-D arrays sorted by time.
    """
    arrivals = pd.DataFrame({
        "time": lifecycle["arrival_time"],
        "delta": 1,
    })
    completions = pd.DataFrame({
        "time": lifecycle["completion_time"],
        "delta": -1,
    })
    events = pd.concat([arrivals, completions], ignore_index=True).sort_values("time")
    times = events["time"].to_numpy()
    wip = events["delta"].cumsum().to_numpy()
    return times, wip


def station_utilization(
    event_log: pd.DataFrame,
    t0: float,
    t1: float,
    stations: dict,
    order: list | None = None,
) -> pd.DataFrame:
    """Compute empirical slot utilization per station in [t0, t1].

    Each operation is clipped to [t0, t1] so boundary-crossing ops are counted
    proportionally. Slot utilization =
    total_busy_time / (n_tools * batch_size * window_length): for serial tools
    this is the classic busy-tool fraction; for batch tools (FURNACE) each
    member row carries the run duration, so the ratio measures used lot-slots —
    the fab-standard capacity view.

    Parameters
    ----------
    stations : dict
        ``metadata.json``'s ``stations`` mapping:
        ``{name: {"n_tools": int, "batch_size": int, ...}}``.
    order : list | None
        Station display order; defaults to first-visit route order if omitted
        (i.e. the order of ``stations`` keys, which insertion-preserves route).

    Returns a DataFrame with columns: station (str), utilization (float 0–1).
    """
    window = t1 - t0
    # keep only ops that overlap [t0, t1]
    overlap = event_log[
        (event_log["process_start_time"] < t1) &
        (event_log["process_complete_time"] > t0)
    ].copy()
    overlap["clipped_start"] = overlap["process_start_time"].clip(lower=t0)
    overlap["clipped_end"] = overlap["process_complete_time"].clip(upper=t1)
    overlap["busy"] = overlap["clipped_end"] - overlap["clipped_start"]

    busy_by_station = overlap.groupby("station")["busy"].sum()
    station_order = order or list(stations.keys())
    capacity = {
        s: stations[s]["n_tools"] * stations[s].get("batch_size", 1)
        for s in station_order
    }
    util = pd.Series(
        {s: busy_by_station.get(s, 0.0) / (capacity[s] * window)
         for s in station_order},
        name="utilization",
    )
    result = util.reset_index()
    result.columns = ["station", "utilization"]
    return result


def x_factor(
    event_log: pd.DataFrame,
    lifecycle: pd.DataFrame,
    t0: float,
    t1: float,
) -> tuple[pd.Series, float, float]:
    """Per-lot X-factor for lots completed within [t0, t1].

    X-factor = cycle time / raw process time — the headline fab flow metric
    (how many times longer a lot takes than pure processing; the excess is
    queueing). Raw process time per lot is the sum of its operations'
    (process_complete - process_start); for batch stations that is the lot's
    processing residence in the run, so X = 1 means zero queueing by
    construction.

    Returns (x_series indexed by lot_id, median, p90).
    """
    done = lifecycle.dropna(subset=["completion_time"])
    in_win = done[done["completion_time"].between(t0, t1)].copy()
    ct = (in_win["completion_time"] - in_win["arrival_time"]).to_numpy()

    proc = (
        (event_log["process_complete_time"] - event_log["process_start_time"])
        .clip(lower=0)
        .groupby(event_log["lot_id"])
        .sum()
    )
    raw = in_win["lot_id"].map(proc).to_numpy()
    x = pd.Series(ct / raw, index=in_win["lot_id"], name="x_factor")
    return x, float(x.median()), float(x.quantile(0.90))


def cycle_time_stats(
    lifecycle: pd.DataFrame,
    t0: float,
    t1: float,
) -> tuple[pd.Series, float, float]:
    """Return cycle-time series and summary statistics for steady-state lots.

    A lot is included if its completion_time falls within [t0, t1].
    Returns (ct_series, median, p90) where times are in hours.
    """
    mask = lifecycle["completion_time"].between(t0, t1)
    ct = lifecycle.loc[mask, "completion_time"] - lifecycle.loc[mask, "arrival_time"]
    ct = ct.reset_index(drop=True)
    return ct, float(ct.median()), float(ct.quantile(0.90))


def daily_median_ct(lifecycle: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """Compute daily median cycle time for steady-state lots.

    Lots are bucketed by the simulation day their completion falls on.
    Returns a DataFrame with columns: day (int), median_ct (float, hours).
    """
    mask = lifecycle["completion_time"].between(t0, t1)
    df = lifecycle.loc[mask].copy()
    ct = df["completion_time"] - df["arrival_time"]
    df["ct"] = ct
    df["day"] = np.floor(df["completion_time"] / 24).astype(int)
    return df.groupby("day", as_index=False)["ct"].median().rename(columns={"ct": "median_ct"})
