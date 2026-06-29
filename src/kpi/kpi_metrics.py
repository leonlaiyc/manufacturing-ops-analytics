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
    n_tools: int,
) -> pd.DataFrame:
    """Compute empirical utilization per station in [t0, t1].

    Each operation is clipped to [t0, t1] so boundary-crossing ops are counted
    proportionally. Utilization = total_busy_time / (n_tools * window_length).

    Returns a DataFrame with columns: station (str), utilization (float 0–1).
    Stations are sorted by the canonical route order S1–S7.
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
    util = busy_by_station / (n_tools * window)

    station_order = [f"S{i}" for i in range(1, 8)]
    result = (
        util.reindex(station_order)
        .reset_index()
        .rename(columns={"station": "station", "busy": "utilization"})
    )
    result.columns = ["station", "utilization"]
    return result


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
