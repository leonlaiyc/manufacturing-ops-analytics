"""
Per-tool SEMI E10 state timeline (M8 Stage A).

SEMI E10 ("Standard for Definition and Measurement of Equipment Reliability,
Availability, and Maintainability") is the semiconductor-industry naming
convention for equipment states. This module builds a STYLIZED mapping onto
that naming, not a standards-compliant E10 implementation: only the states
this project's DES can actually observe are populated (see below), and the
mapping is documented so the simplification is never hidden.

State definitions used here
----------------------------
  PRODUCTIVE           : the tool is running a lot (or a batch run), taken
                          directly from the event log's
                          [process_start_time, process_complete_time]
                          intervals for that tool_id.
  SCHEDULED DOWNTIME    : a planned-maintenance injection window
                          (``ScheduledDowntimeAnomaly``, type "scheduled_pm")
                          attributed to a specific tool (see convention below).
  UNSCHEDULED DOWNTIME  : an unplanned-failure injection window
                          (``BreakdownAnomaly``, type "breakdown") attributed
                          the same way.
  STANDBY               : whatever time is left over, i.e. the tool is up
                          (not in a down window) and not running a lot. This
                          is the residual bucket, not independently modeled.
  ENGINEERING           : present in the schema for completeness (SEMI E10
                          includes engineering/qualification runs) but ALWAYS
                          EMPTY in this stylized model - the generator has no
                          concept of an engineering run distinct from a
                          production run. Documented limitation, not a bug.

Tool-attribution convention for down windows (M8)
--------------------------------------------------
``_simulate_injected`` (see ``factory_generator.py``) reduces a station's
capacity by COUNT ("how many tools may run concurrently"), never by specific
tool index - the dispatch loop has no notion of "which" tool is down. To
render a down window onto a per-TOOL timeline, this module therefore ADOPTS A
CONVENTION, stated here once and reused everywhere: for a station with
``n_tools`` tools, the HIGHEST-index tool (e.g. "LITHO-2" for a 2-tool
station) is the one considered down for each unit of ``tools_removed``
during the window. This matches the acquisition rule in ``_ToolPool``
(lowest-index-free is always grabbed first), so under normal load the
highest-index tool is the one most likely to sit idle when capacity is cut,
making it the natural (if arbitrary) attribution choice. With
``tools_removed`` > 1 the convention extends to the top ``tools_removed``
indices; this project only ever injects ``tools_removed=1``, so that case is
documented but not exercised.

Consistency rule for PRODUCTIVE-vs-DOWN overlap (required by spec)
--------------------------------------------------------------------
Because a down window is attributed to a specific tool AFTER the fact (the
simulator itself only ever throttles a COUNT), it is possible for the
attributed tool to have already been mid-run when the window opens (a run
that started before t_start and completes after t_start is still legitimately
PRODUCTIVE for its full duration - the simulator would never have started a
run on a tool it was simultaneously holding down). The resolution rule,
applied by construction rather than by silently clipping either interval:

    PRODUCTIVE WINS. The down window is truncated to exclude any sub-interval
    where the attributed tool has a logged PRODUCTIVE run. A down window can
    therefore be split into multiple DOWNTIME segments (with the interleaved
    PRODUCTIVE segment kept exactly as logged), or fully consumed if a run
    spans it entirely.

This keeps the partition exact (GATE 1 in ``equipment_check.py``: no gaps, no
overlaps, states sum to the horizon per tool) while never inventing or
deleting a logged production interval.
"""

from __future__ import annotations

import pandas as pd

E10_STATES = ("PRODUCTIVE", "STANDBY", "SCHEDULED DOWNTIME",
              "UNSCHEDULED DOWNTIME", "ENGINEERING")

_ANOMALY_TYPE_TO_STATE = {
    "scheduled_pm": "SCHEDULED DOWNTIME",
    "breakdown": "UNSCHEDULED DOWNTIME",
}


def _station_tool_labels(station: str, n_tools: int) -> list:
    """All tool_id labels for a station, e.g. ("LITHO-1", "LITHO-2")."""
    return [f"{station}-{i + 1}" for i in range(n_tools)]


def _down_windows_by_tool(anomalies_meta: list, stations: dict) -> dict:
    """Map tool_id -> list of (t_start, t_end, e10_state, reason_code).

    Applies the highest-index-tool convention documented above: for each
    injection window with ``tools_removed = k`` at ``station``, the top ``k``
    tool indices of that station are attributed the window.
    """
    out: dict = {}
    for a in anomalies_meta:
        a_type = a.get("type")
        state = _ANOMALY_TYPE_TO_STATE.get(a_type)
        if state is None:
            continue  # not a capacity-reduction anomaly (e.g. demand_surge, degradation)
        station = a["station"]
        n_tools = stations[station]["n_tools"]
        k = a.get("tools_removed", 1)
        down_labels = _station_tool_labels(station, n_tools)[n_tools - k:]
        reason = f"{a_type}:{station}:{a['t_start']:.4f}-{a['t_end']:.4f}"
        for tool_id in down_labels:
            out.setdefault(tool_id, []).append(
                (float(a["t_start"]), float(a["t_end"]), state, reason))
    return out


def _productive_intervals(log: pd.DataFrame, tool_id: str) -> list:
    """Sorted, merged PRODUCTIVE (start, end) intervals for one tool_id.

    A batch run produces one log row per member lot with identical
    [process_start_time, process_complete_time], so duplicates are dropped
    before sorting; adjacent/overlapping identical rows must not double-count
    productive time.
    """
    rows = log.loc[log["tool_id"] == tool_id,
                    ["process_start_time", "process_complete_time"]]
    pairs = sorted(set(zip(rows["process_start_time"].astype(float),
                           rows["process_complete_time"].astype(float))))
    merged: list = []
    for s, e in pairs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _subtract_intervals(window: tuple, blockers: list) -> list:
    """window minus every overlapping interval in blockers -> list of sub-windows.

    ``blockers`` need not be sorted or disjoint from each other in general,
    but productive intervals for one tool are already merged/disjoint by
    construction (see ``_productive_intervals``).
    """
    w_start, w_end = window
    segments = [(w_start, w_end)]
    for b_start, b_end in blockers:
        next_segments = []
        for s, e in segments:
            if b_end <= s or b_start >= e:
                next_segments.append((s, e))  # no overlap
                continue
            if b_start > s:
                next_segments.append((s, b_start))
            if b_end < e:
                next_segments.append((b_end, e))
        segments = next_segments
    return [(s, e) for s, e in segments if e > s]


def build_tool_state_timeline(log: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Build the tidy per-tool E10 state timeline over [0, effective horizon].

    Parameters
    ----------
    log : pd.DataFrame
        Tool-level event log from ``simulate()`` (must carry ``tool_id``).
    meta : dict
        The ``meta`` dict returned by ``simulate()``. Reads ``horizon_hours``,
        ``stations`` (for ``n_tools`` per station) and, if present,
        ``anomalies`` (the injection metadata list; absent or empty means a
        clean run with no down windows).

    Effective horizon: ``simulate()`` stops SCHEDULING NEW ARRIVALS at
    ``cfg.horizon_hours`` but then drains every already-arrived lot to
    completion, so the last ``process_complete_time`` in the log can fall
    AFTER the nominal horizon. Partitioning the timeline against the nominal
    horizon would then leave a real logged PRODUCTIVE interval hanging past
    the end of [0, horizon) - a manufactured gap/overlap bug, not a
    consistency error in the data. The state layer therefore partitions
    against ``effective_horizon = max(horizon_hours, log.process_complete_time.max())``
    so every logged interval is always inside the partitioned span.

    Returns
    -------
    pd.DataFrame with columns:
        tool_id, station, t_start, t_end, e10_state, reason_code
    Sorted by (station, tool_id, t_start). For every tool_id, the state
    intervals exactly partition [0, effective_horizon]: no gaps, no overlaps
    (GATE 1 in equipment_check.py asserts this to 1e-9).

    ENGINEERING never appears (see module docstring: not modeled).
    """
    horizon = float(meta["horizon_hours"])
    if len(log):
        horizon = max(horizon, float(log["process_complete_time"].max()))
    stations = meta["stations"]
    anomalies_meta = meta.get("anomalies", []) or []
    down_by_tool = _down_windows_by_tool(anomalies_meta, stations)

    rows = []
    for station, sinfo in stations.items():
        n_tools = sinfo["n_tools"]
        for tool_id in _station_tool_labels(station, n_tools):
            productive = _productive_intervals(log, tool_id)
            down_windows = down_by_tool.get(tool_id, [])

            # Resolution rule: PRODUCTIVE WINS. Truncate each down window by
            # the tool's productive intervals before laying it on the timeline.
            down_segments = []  # (start, end, state, reason)
            for d_start, d_end, state, reason in down_windows:
                for s, e in _subtract_intervals((d_start, d_end), productive):
                    down_segments.append((s, e, state, reason))

            events = []
            for s, e in productive:
                events.append((s, e, "PRODUCTIVE", "run"))
            for s, e, state, reason in down_segments:
                events.append((s, e, state, reason))
            events.sort(key=lambda x: x[0])

            # Fill gaps (including before the first and after the last event,
            # and between consecutive events) with STANDBY so the partition
            # is exact by construction.
            cursor = 0.0
            for s, e, state, reason in events:
                if s > cursor + 1e-12:
                    rows.append((tool_id, station, cursor, s, "STANDBY", "idle"))
                rows.append((tool_id, station, max(s, cursor), e, state, reason))
                cursor = max(cursor, e)
            if cursor < horizon - 1e-12:
                rows.append((tool_id, station, cursor, horizon, "STANDBY", "idle"))

    out = pd.DataFrame(rows, columns=["tool_id", "station", "t_start", "t_end",
                                       "e10_state", "reason_code"])
    return (out.sort_values(["station", "tool_id", "t_start"])
               .reset_index(drop=True))
