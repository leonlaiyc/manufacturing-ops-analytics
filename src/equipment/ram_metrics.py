"""
RAM (Reliability, Availability, Maintainability) metrics on the E10 state
timeline (M8 Stage A).

Naming source: SEMI E10 defines availability as up time / total time and
frames reliability/maintainability around mean time between failures (MTBF)
and mean time to repair (MTTR). This module is a STYLIZED mapping onto that
naming, not a standards-compliant E10/E10-equivalent calculation: only
UNSCHEDULED DOWNTIME counts as a "failure" for MTBF/MTTR (SEMI E10 has finer
distinctions - e.g. engineering time, standby-vs-idle splits - that this
project's DES does not model; see ``e10_states.py``). Formulas are kept
deliberately simple so each one can be explained from first principles:

  state-time decomposition : hours and share of total time per E10 state.
  MTBF (mean time between failures)
        = mean UP time between consecutive UNSCHEDULED DOWNTIME events
        = (sum of up-time gaps between unscheduled-down events) / (count of
          unscheduled-down events), using the "up time preceding each failure"
          convention (the run from the previous recovery, or from t=0 for the
          first failure, to the next failure's start).
  MTTR (mean time to repair)
        = mean duration of UNSCHEDULED DOWNTIME events.
  availability
        = up time / total time, where up time = total time minus (SCHEDULED
          DOWNTIME + UNSCHEDULED DOWNTIME). PRODUCTIVE and STANDBY both count
          as "up" (SEMI E10 availability is about the tool being available to
          run, not about whether it happens to be running right now).
  equipment utilization
        = PRODUCTIVE time / total time (fraction of calendar time actually
          producing; distinct from availability, and distinct from the
          generator's DESIGN-TIME slot utilization used elsewhere in this
          project for station capacity planning).

A tool with zero UNSCHEDULED DOWNTIME events has an undefined MTBF/MTTR by
the formulas above (no failures to measure between); this module reports
``float("nan")`` in that case rather than a misleading 0 or infinity.
"""

from __future__ import annotations

import pandas as pd

UP_STATES = ("PRODUCTIVE", "STANDBY", "ENGINEERING")
DOWN_STATES = ("SCHEDULED DOWNTIME", "UNSCHEDULED DOWNTIME")


def state_time_decomposition(timeline: pd.DataFrame, group_cols) -> pd.DataFrame:
    """Hours and share of total time per E10 state, grouped by ``group_cols``.

    ``group_cols`` is typically ``["tool_id"]`` (per tool) or ``["station"]``
    (per station, i.e. summed across that station's tools).

    Returns a tidy DataFrame: group_cols..., e10_state, hours, share.
    """
    df = timeline.copy()
    df["duration"] = df["t_end"] - df["t_start"]
    totals = df.groupby(list(group_cols))["duration"].transform("sum")
    df["share"] = df["duration"] / totals
    out = (df.groupby(list(group_cols) + ["e10_state"])["duration"]
             .sum()
             .reset_index()
             .rename(columns={"duration": "hours"}))
    group_totals = (df.groupby(list(group_cols))["duration"].sum()
                       .rename("total_hours"))
    out = out.merge(group_totals, on=list(group_cols))
    out["share"] = out["hours"] / out["total_hours"]
    return out.drop(columns="total_hours")


def _mtbf_mttr_availability_one_tool(tool_df: pd.DataFrame) -> dict:
    """MTBF, MTTR, availability, utilization for one tool's state intervals.

    ``tool_df`` must be sorted by t_start (caller's responsibility) and cover
    exactly one tool_id.
    """
    total_time = float(tool_df["t_end"].max() - tool_df["t_start"].min())
    dur = tool_df["t_end"] - tool_df["t_start"]

    up_time = float(dur[tool_df["e10_state"].isin(UP_STATES)].sum())
    productive_time = float(dur[tool_df["e10_state"] == "PRODUCTIVE"].sum())
    availability = up_time / total_time if total_time > 0 else float("nan")
    utilization = productive_time / total_time if total_time > 0 else float("nan")

    failures = tool_df[tool_df["e10_state"] == "UNSCHEDULED DOWNTIME"]
    n_failures = len(failures)
    if n_failures == 0:
        mtbf = float("nan")
        mttr = float("nan")
    else:
        mttr = float((failures["t_end"] - failures["t_start"]).mean())
        # Up-time run preceding each failure: from the previous failure's
        # t_end (or 0.0 for the first) to this failure's t_start.
        prev_end = 0.0
        up_runs = []
        for _, f in failures.sort_values("t_start").iterrows():
            up_runs.append(f["t_start"] - prev_end)
            prev_end = f["t_end"]
        mtbf = float(sum(up_runs) / len(up_runs))

    return {
        "hours_total": total_time,
        "hours_up": up_time,
        "hours_productive": productive_time,
        "n_unscheduled_events": n_failures,
        "mtbf_hours": mtbf,
        "mttr_hours": mttr,
        "availability": availability,
        "utilization": utilization,
    }


def ram_metrics_by_tool(timeline: pd.DataFrame) -> pd.DataFrame:
    """Per-tool RAM metrics table (one row per tool_id). See module docstring
    for formulas."""
    rows = []
    for tool_id, tdf in timeline.sort_values("t_start").groupby("tool_id"):
        station = tdf["station"].iloc[0]
        metrics = _mtbf_mttr_availability_one_tool(tdf)
        rows.append({"tool_id": tool_id, "station": station, **metrics})
    return (pd.DataFrame(rows)
              .sort_values(["station", "tool_id"])
              .reset_index(drop=True))


def ram_metrics_by_station(timeline: pd.DataFrame) -> pd.DataFrame:
    """Per-station RAM metrics: same formulas, applied to each station's
    pooled (summed) tool intervals rather than one tool at a time.

    Station-level "up time between failures" pools all of that station's
    tools' UNSCHEDULED DOWNTIME events into one time-ordered sequence, which
    is the natural station-level reading of MTBF/MTTR (how often does THIS
    STATION suffer an unplanned outage, regardless of which of its tools).
    """
    rows = []
    for station, sdf in timeline.sort_values("t_start").groupby("station"):
        # Total/up/productive hours sum naturally across tools at fixed time
        # span, since each tool independently partitions [0, horizon].
        n_tools = sdf["tool_id"].nunique()
        horizon = float(sdf["t_end"].max() - sdf["t_start"].min())
        total_time = horizon * n_tools
        dur = sdf["t_end"] - sdf["t_start"]
        up_time = float(dur[sdf["e10_state"].isin(UP_STATES)].sum())
        productive_time = float(dur[sdf["e10_state"] == "PRODUCTIVE"].sum())
        availability = up_time / total_time if total_time > 0 else float("nan")
        utilization = productive_time / total_time if total_time > 0 else float("nan")

        failures = sdf[sdf["e10_state"] == "UNSCHEDULED DOWNTIME"].sort_values("t_start")
        n_failures = len(failures)
        if n_failures == 0:
            mtbf = float("nan")
            mttr = float("nan")
        else:
            mttr = float((failures["t_end"] - failures["t_start"]).mean())
            # Pooled across tools: "up time since the previous failure at this
            # station" measured on the wall clock (per-tool up time summed
            # over n_tools tools running in parallel), matching the
            # total_time convention above.
            prev_end = 0.0
            up_runs = []
            for _, f in failures.iterrows():
                up_runs.append((f["t_start"] - prev_end) * n_tools)
                prev_end = max(prev_end, f["t_end"])
            mtbf = float(sum(up_runs) / len(up_runs))

        rows.append({
            "station": station,
            "n_tools": n_tools,
            "hours_total": total_time,
            "hours_up": up_time,
            "hours_productive": productive_time,
            "n_unscheduled_events": n_failures,
            "mtbf_hours": mtbf,
            "mttr_hours": mttr,
            "availability": availability,
            "utilization": utilization,
        })
    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
