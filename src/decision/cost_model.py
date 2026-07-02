"""
Transparent cost model for decision support (M6).

Three components, each with one illustrative rate that can be defended in review.
The model exists to RANK improvement options relative to one another on a common
scale — NOT to predict absolute dollars. All rates are illustrative assumptions
(stated as such in the notebook) and every conclusion is re-checked under a
+/-50% sensitivity sweep.

  1. processing cost = sum of station busy-hours * proc_rate      ($/tool-hour)
  2. holding cost    = sum of queue-wait hours * hold_rate        ($/lot-hour)
                       (congestion cost of work waiting in front of stations;
                        a WIP time-in-system basis is an equally valid alternative)
  3. capacity cost   = tools_added * tool_cost                    ($ per added tool,
                       amortised over the simulated horizon)
  (+ optional one-off repair_cost, used by the early-fix improvement option.)

All hour sums are clipped to the steady-state window [t0, t1] so boundary-crossing
operations are counted only for their in-window portion, matching the M2/M3
utilisation convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostRates:
    """Illustrative cost rates (NOT real figures — used for relative ranking)."""
    proc_rate: float = 100.0      # $ per tool-hour of processing
    hold_rate: float = 10.0       # $ per lot-hour of waiting in queue
    tool_cost: float = 20000.0    # $ per added tool, amortised over the horizon
    repair_cost: float = 2000.0   # $ one-off, for the early-degradation-fix option


def _processing_hours(log: pd.DataFrame, t0: float, t1: float) -> float:
    start = log["process_start_time"].clip(lower=t0, upper=t1)
    end = log["process_complete_time"].clip(lower=t0, upper=t1)
    return float((end - start).clip(lower=0).sum())


def _waiting_hours(log: pd.DataFrame, t0: float, t1: float) -> float:
    lo = log["queue_entry_time"].clip(lower=t0, upper=t1)
    hi = log["process_start_time"].clip(lower=t0, upper=t1)
    return float((hi - lo).clip(lower=0).sum())


def cost_components(log: pd.DataFrame, t0: float, t1: float, rates: CostRates,
                    tools_added: int = 0, repairs: int = 0) -> dict:
    """Return the cost broken into {processing, holding, capacity, repair, total}."""
    proc = _processing_hours(log, t0, t1) * rates.proc_rate
    hold = _waiting_hours(log, t0, t1) * rates.hold_rate
    cap = tools_added * rates.tool_cost
    rep = repairs * rates.repair_cost
    return {
        "processing": proc,
        "holding": hold,
        "capacity": cap,
        "repair": rep,
        "total": proc + hold + cap + rep,
    }


def total_cost(log: pd.DataFrame, t0: float, t1: float, rates: CostRates,
               tools_added: int = 0, repairs: int = 0) -> float:
    """Scalar total cost over [t0, t1] (see cost_components for the breakdown)."""
    return cost_components(log, t0, t1, rates, tools_added, repairs)["total"]


def daily_operating_cost(log: pd.DataFrame, horizon: float, rates: CostRates) -> pd.Series:
    """Per-day operating cost (processing + holding) for the cost-over-time view.

    Operations are bucketed by the day their processing completes. Capacity/repair
    are one-off and not part of the per-day operating series.
    """
    day = 24.0
    ops = log.copy()
    ops["proc_h"] = (ops["process_complete_time"] - ops["process_start_time"]).clip(lower=0)
    ops["wait_h"] = (ops["process_start_time"] - ops["queue_entry_time"]).clip(lower=0)
    ops["cost"] = ops["proc_h"] * rates.proc_rate + ops["wait_h"] * rates.hold_rate
    ops["day"] = np.floor(ops["process_complete_time"] / day).astype(int)
    s = ops.groupby("day")["cost"].sum()
    s = s.reindex(range(int(np.ceil(horizon / day))), fill_value=0.0)
    s.name = "daily_cost"
    return s
