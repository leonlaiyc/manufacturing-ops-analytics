"""
Naive bottleneck baselines (M4, Step 2).

Two tempting-but-wrong heuristics, implemented so we can show *why* the
multi-evidence method (Step 1) is needed rather than assumed:

  Naive-1  "highest-frequency station"  — the station that appears in the most
           operations. Confuses traffic volume with constraint.
  Naive-2  "longest-processing station" — the station with the highest mean
           per-operation processing time. Confuses slow-per-op with constraint;
           ignores how many tools a station has and how often it is visited.

Honest narrative (the point of this step):

  - Synthetic log:
      * Naive-1 lands on S4 — but only by COINCIDENCE. S4 is re-entrant (visited
        twice per lot), so it trivially has the most operations. It is right for
        the wrong reason: frequency, not constraint.
      * Naive-2 lands on S3 (pt_mean 1.3 > S4's 0.85) — WRONG. The true
        constraint S4 is faster per operation; its load comes from being visited
        twice with the least tool headroom, which per-op processing time cannot see.
  - Real 4TU log:
      * Naive-1 lands on Final Inspection Q.C. (the highest-traffic activity) —
        WRONG. High traffic, but the waiting accumulates elsewhere.

  Conclusion carried into the notebook: naive methods that happen to be right are
  right by coincidence, not by principle, and break the moment the data changes.
  Recovering the constraint requires the queue/utilization evidence of Step 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def naive_baselines_synthetic(log: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Per-station frequency and mean per-op processing time (synthetic log).

    Returns (table, picks) where table is indexed by station in route order with
    columns [frequency, mean_proc_hours], and picks = {"highest_frequency": s,
    "longest_processing": s}.
    """
    stations = list(cfg.stations.keys())
    ev = log.copy()
    ev["proc_hours"] = ev["process_complete_time"] - ev["process_start_time"]

    freq = ev.groupby("station").size()
    mean_proc = ev.groupby("station")["proc_hours"].mean()

    table = pd.DataFrame({
        "frequency": freq,
        "mean_proc_hours": mean_proc,
    }).reindex(stations)

    picks = {
        "highest_frequency": table["frequency"].idxmax(),
        "longest_processing": table["mean_proc_hours"].idxmax(),
    }
    return table, picks


def naive_baselines_real(df_D: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-activity frequency and mean per-op processing time (real 4TU log).

    Expects df_D with 'Activity' and 'processing_hours' (falls back to computing
    processing_hours from 'Start Timestamp'/'Complete Timestamp' if absent).
    """
    d = df_D.copy()
    if "processing_hours" not in d.columns:
        d["processing_hours"] = (
            (d["Complete Timestamp"] - d["Start Timestamp"]).dt.total_seconds() / 3600
        ).clip(lower=0)

    freq = d.groupby("Activity").size()
    mean_proc = d.groupby("Activity")["processing_hours"].mean()

    table = pd.DataFrame({
        "frequency": freq,
        "mean_proc_hours": mean_proc,
    })
    picks = {
        "highest_frequency": table["frequency"].idxmax(),
        "longest_processing": table["mean_proc_hours"].idxmax(),
    }
    return table, picks


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _bar(ax, series, pick, truth, title, horizontal=False, top_n=None):
    s = series.sort_values(ascending=False)
    if top_n:
        s = s.head(top_n)
    labels = list(s.index)
    colors = []
    for lab in labels:
        if lab == pick and lab == truth:
            colors.append("#8E24AA")          # pick AND truth (coincidence)
        elif lab == pick:
            colors.append("#EF5350")          # naive pick (wrong)
        elif lab == truth:
            colors.append("#43A047")          # true/candidate constraint
        else:
            colors.append("#B0BEC5")
    if horizontal:
        ax.barh(labels, s.values, color=colors)
        ax.invert_yaxis()
    else:
        ax.bar(labels, s.values, color=colors)
        ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(axis=("x" if horizontal else "y"), alpha=0.25)


def plot_naive_baselines(
    syn_table: pd.DataFrame,
    syn_picks: dict,
    real_table: pd.DataFrame,
    real_picks: dict,
    syn_truth: str = "S4",
    real_candidate: str | None = None,
    save_path: str | None = None,
):
    """2x2 figure: rows = {synthetic, real}, cols = {frequency, mean processing}.

    Colour key: red = naive pick (wrong), green = true/candidate constraint,
    purple = naive pick that happens to coincide with the truth (right by luck).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    _bar(axes[0, 0], syn_table["frequency"], syn_picks["highest_frequency"],
         syn_truth, "Synthetic — Naive-1: operation frequency")
    _bar(axes[0, 1], syn_table["mean_proc_hours"], syn_picks["longest_processing"],
         syn_truth, "Synthetic — Naive-2: mean processing time / op (h)")
    _bar(axes[1, 0], real_table["frequency"], real_picks["highest_frequency"],
         real_candidate, "Real 4TU — Naive-1: activity frequency (top 12)",
         horizontal=True, top_n=12)
    _bar(axes[1, 1], real_table["mean_proc_hours"], real_picks["longest_processing"],
         real_candidate, "Real 4TU — Naive-2: mean processing time / op, h (top 12)",
         horizontal=True, top_n=12)

    fig.suptitle(
        "Naive baselines mislead — red = naive pick (wrong), green = true/candidate "
        "constraint, purple = right by coincidence\n"
        f"Synthetic: Naive-1 -> {syn_picks['highest_frequency']} "
        f"(coincidence: S4 is re-entrant), Naive-2 -> {syn_picks['longest_processing']} "
        f"(wrong).  Real: Naive-1 -> {real_picks['highest_frequency']} (wrong).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
