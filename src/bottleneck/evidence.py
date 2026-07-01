"""
Multi-evidence bottleneck identification (M4, Step 1).

The claim under test: the bottleneck is the station whose *finite capacity most
constrains* line throughput and cycle time — not simply the busiest, the
highest-traffic, or the longest-processing station. We test it by computing
several INDEPENDENT descriptive signals per station over the steady-state window
and checking whether they CONVERGE on the same station.

Signals (each interpretable from first principles; NO weighted composite score):

  1. utilization        busy-tool fraction = time the station's tools are working
                        / (n_tools * window). A station near 1.0 has no headroom.
  2. avg_queue_len      time-average number of lots waiting in front of the
                        station (Little's Law on the queue: sum of per-lot wait
                        durations / window). Work piles up before a constraint.
  3. avg_wait_hours     mean time a lot waits in queue before this station starts
                        serving it (per-lot queue delay).
  4. wait_share         this station's total waiting time as a fraction of the
                        whole plant's waiting time. Where does the line's waiting
                        actually accumulate?
  5. idle_fraction      OPTIONAL, softer "downstream starvation" proxy: fraction
                        of the window the station has ALL tools idle. Stations
                        downstream of the constraint sit idle waiting to be fed.
                        Kept as secondary evidence; the convergence conclusion
                        rests on signals 1–4.

Deliberately NOT combined into a single ranked score — the owner's method is to
show the signals independently agree, which is more defensible than a weighted
index whose weights would need their own justification.

There are two entry points because the two data sources carry different fields:
  - station_evidence_synthetic(): the synthetic event log has explicit
    queue_entry / process_start / process_complete timestamps and a known
    n_tools per station, so all signals (incl. utilization) are computable.
  - activity_evidence_real(): the real 4TU log has neither tool counts nor an
    arrival model, so utilization is NOT computable; only queue-based signals
    (waiting time, waiting share, wait/processing ratio) are available. See M4
    Step 5 and the honest-scope note in the notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Synthetic event log — full evidence set (utilization + queue signals)
# --------------------------------------------------------------------------- #
def _idle_fraction(ops: pd.DataFrame, n_tools: int, t0: float, t1: float) -> float:
    """Fraction of [t0, t1] during which the station has ALL tools idle.

    Builds the union of busy intervals from each operation's
    [process_start, process_complete] (clipped to the window) and measures how
    much of the window is covered by NO operation at all. Independent of n_tools
    for the "at least one tool busy" notion; a downstream-starved station spends
    much of the window fully idle.
    """
    window = t1 - t0
    intervals = []
    for a, b in zip(ops["process_start_time"], ops["process_complete_time"]):
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            intervals.append((lo, hi))
    if not intervals:
        return 1.0
    intervals.sort()
    busy = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:              # overlapping/contiguous -> extend
            cur_hi = max(cur_hi, hi)
        else:
            busy += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    busy += cur_hi - cur_lo
    return float(1.0 - busy / window)


def station_evidence_synthetic(
    log: pd.DataFrame,
    cfg,
    t0: float,
    t1: float,
) -> pd.DataFrame:
    """Per-station evidence signals over the steady-state window [t0, t1].

    Parameters
    ----------
    log : event log with columns lot_id, station, queue_entry_time,
          process_start_time, process_complete_time.
    cfg : FactoryConfig — provides per-station n_tools and the route order.

    Returns
    -------
    DataFrame indexed by station (route order) with columns:
        utilization, avg_queue_len, avg_wait_hours, wait_share, idle_fraction.
    No composite score is produced.
    """
    window = t1 - t0
    stations = list(cfg.stations.keys())

    # Per-op waiting and service durations, clipped to the window.
    ev = log.copy()
    ev["wait_lo"] = ev["queue_entry_time"].clip(lower=t0, upper=t1)
    ev["wait_hi"] = ev["process_start_time"].clip(lower=t0, upper=t1)
    ev["wait_clipped"] = (ev["wait_hi"] - ev["wait_lo"]).clip(lower=0)

    ev["busy_lo"] = ev["process_start_time"].clip(lower=t0, upper=t1)
    ev["busy_hi"] = ev["process_complete_time"].clip(lower=t0, upper=t1)
    ev["busy_clipped"] = (ev["busy_hi"] - ev["busy_lo"]).clip(lower=0)

    # Per-lot wait (unclipped) for ops that START service inside the window.
    started = ev[ev["process_start_time"].between(t0, t1)].copy()
    started["wait_raw"] = (
        started["process_start_time"] - started["queue_entry_time"]
    ).clip(lower=0)

    total_wait_all = ev.groupby("station")["wait_clipped"].sum()
    plant_wait = total_wait_all.sum()

    rows = {}
    for s in stations:
        st = cfg.stations[s]
        ops = ev[ev["station"] == s]
        busy = ops["busy_clipped"].sum()
        util = busy / (st.n_tools * window)
        avg_queue_len = ops["wait_clipped"].sum() / window           # L_q (time-avg)
        s_started = started[started["station"] == s]["wait_raw"]
        avg_wait = float(s_started.mean()) if len(s_started) else 0.0  # W_q (per-lot)
        wait_share = float(total_wait_all.get(s, 0.0) / plant_wait) if plant_wait else 0.0
        idle_frac = _idle_fraction(ops, st.n_tools, t0, t1)
        rows[s] = {
            "utilization": float(util),
            "avg_queue_len": float(avg_queue_len),
            "avg_wait_hours": avg_wait,
            "wait_share": wait_share,
            "idle_fraction": idle_frac,
        }

    return pd.DataFrame.from_dict(rows, orient="index").reindex(stations)


def converged_station(evidence: pd.DataFrame, signals: list[str] | None = None) -> str:
    """The station that tops the most evidence signals (majority vote, no weights).

    ``idle_fraction`` is excluded by default: it is a "smaller = more constrained"
    signal and is treated as secondary. The remaining signals are all
    "larger = more constrained", so we take the argmax of each and report the
    modal station. This is a convergence check, not a scoring rule.
    """
    if signals is None:
        signals = ["utilization", "avg_queue_len", "avg_wait_hours", "wait_share"]
    picks = [evidence[c].idxmax() for c in signals]
    return pd.Series(picks).mode().iloc[0]


# --------------------------------------------------------------------------- #
# Real 4TU log — queue-based evidence only (NO utilization possible)
# --------------------------------------------------------------------------- #
def activity_evidence_real(df_D: pd.DataFrame) -> pd.DataFrame:
    """Queue-based evidence per activity from the real log's D-type events.

    Waiting time is the gap between a step's Start and the previous step's
    Complete WITHIN a case, attributed to the RECEIVING step (same convention as
    M1 Section 6). Utilization / queue-length / starvation are NOT computed: the
    real log has no declared tool counts, no arrival process, and an irregular
    calendar, so a defensible utilization cannot be derived.

    Expects df_D with columns: 'Case ID', 'Activity', 'Start Timestamp',
    'Complete Timestamp', and a numeric 'processing_hours' column.

    Returns
    -------
    DataFrame indexed by Activity with columns:
        n_ops, total_wait_hours, wait_share, median_wait_hours,
        total_proc_hours, wait_proc_ratio  (sorted by total_wait_hours desc).
    """
    d = df_D.sort_values(["Case ID", "Start Timestamp"]).copy()
    d["prev_complete"] = d.groupby("Case ID")["Complete Timestamp"].shift(1)
    d["waiting_hours"] = (
        (d["Start Timestamp"] - d["prev_complete"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    waits = d.dropna(subset=["waiting_hours"])
    total_wait = waits.groupby("Activity")["waiting_hours"].sum()
    plant_wait = total_wait.sum()
    median_wait = waits.groupby("Activity")["waiting_hours"].median()
    total_proc = df_D.groupby("Activity")["processing_hours"].sum()
    n_ops = df_D.groupby("Activity").size()

    out = pd.DataFrame({
        "n_ops": n_ops,
        "total_wait_hours": total_wait,
        "median_wait_hours": median_wait,
        "total_proc_hours": total_proc,
    }).fillna(0.0)
    out["wait_share"] = out["total_wait_hours"] / plant_wait if plant_wait else 0.0
    out["wait_proc_ratio"] = out["total_wait_hours"] / out["total_proc_hours"].replace(0, np.nan)
    out = out.sort_values("total_wait_hours", ascending=False)
    return out[[
        "n_ops", "total_wait_hours", "wait_share",
        "median_wait_hours", "total_proc_hours", "wait_proc_ratio",
    ]]


# --------------------------------------------------------------------------- #
# Plots (matplotlib static images for GitHub visibility)
# --------------------------------------------------------------------------- #
def plot_evidence_synthetic(
    evidence: pd.DataFrame,
    ground_truth: str,
    save_path: str | None = None,
):
    """Small-multiple bar panels — one per signal — with the constraint highlighted.

    Every "larger = more constrained" signal should peak at the same station; the
    optional idle_fraction panel is the complement (downstream stations sit idle).
    """
    panels = [
        ("utilization", "Utilization (busy-tool fraction)", False),
        ("avg_queue_len", "Avg queue length in front (L_q)", False),
        ("avg_wait_hours", "Avg wait before station (h)", False),
        ("wait_share", "Share of plant-wide waiting", False),
        ("idle_fraction", "Idle fraction (starvation proxy, optional)", True),
    ]
    stations = list(evidence.index)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))

    for ax, (col, title, is_secondary) in zip(axes, panels):
        vals = evidence[col].values
        # Highlight the argmax for primary signals; for idle_fraction the
        # constraint is the MIN, so highlight accordingly.
        target = evidence[col].idxmin() if is_secondary else evidence[col].idxmax()
        colors = [
            ("#9E9E9E" if is_secondary else "#B0BEC5") if s != target else
            ("#EF5350")
            for s in stations
        ]
        # Also outline the ground-truth station so the reader can locate S4.
        edgecolors = ["#1A237E" if s == ground_truth else "none" for s in stations]
        linewidths = [2.2 if s == ground_truth else 0.0 for s in stations]
        ax.bar(stations, vals, color=colors, edgecolor=edgecolors, linewidth=linewidths)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Multi-evidence bottleneck signals (synthetic) — red = signal peak, "
        f"navy outline = ground truth {ground_truth}\n"
        f"No weighted score: signals 1–4 independently converge on the constraint.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_evidence_real(
    evidence_real: pd.DataFrame,
    top_n: int = 10,
    candidate: str | None = None,
    save_path: str | None = None,
):
    """Queue-based evidence ranking for the real log (utilization not available)."""
    top = evidence_real.head(top_n)
    labels = list(top.index)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, title) in zip(axes, [
        ("total_wait_hours", "Total waiting time (h)"),
        ("wait_share", "Share of plant-wide waiting"),
        ("wait_proc_ratio", "Wait / processing ratio"),
    ]):
        vals = top[col].values
        colors = ["#EF5350" if (candidate and a == candidate) else "#90A4AE"
                  for a in labels]
        ax.barh(labels, vals, color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Real 4TU log — queue-based bottleneck evidence "
        "(utilization NOT computable: no tool counts / arrival model; "
        "no counterfactual possible). Candidate = convergence of these signals.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
