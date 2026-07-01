"""
Counterfactual capacity experiment (M4, Step 3).

Question: if we give one station a +1 tool, how much does line performance
improve? A true constraint should respond far more than a non-constraint.

Design (fixed by the project owner):
  - Four scenarios: baseline, S4 +1, S3 +1, S7 +1.
      * S4 = engineered bottleneck.
      * S3 = STRICT control: 2nd-highest utilization AND the station the naive
        "longest processing" heuristic wrongly flags. Showing S3 +1 helps little
        proves the constraint status comes from being the constraint, not merely
        from high load.
      * S7 = CLEAN control: low utilization, route-distant from S4. Least coupled
        to S4, so it anchors the "improve a non-constraint -> almost nothing" end
        of the gradient.
  - N replications, each with Common Random Numbers (CRN): one pre-drawn
    RandomDraws table per replication; baseline and all three interventions run
    on that SAME table, differing only in n_tools. This makes every comparison a
    PAIRED difference (treatment - baseline within a replication), cancelling the
    simulation noise so the delta reflects capacity alone.
  - Report the paired delta distribution across replications: mean + 95% CI.

Honest scope (stated in the notebook): this quantifies the DECISION IMPACT of a
capacity change and tests the decision logic. It does NOT independently "prove"
S4 is the bottleneck — S4 is a known design input. The counterfactual's value is
showing the improvement gradient S4 >> S3 > S7 is what a correct bottleneck call
predicts.

Note on channels: the synthetic line is a stable open network (every rho < 1),
so in steady state throughput is set by the arrival rate and Delta-throughput is
small for every station. The discriminating signal is Delta-cycle-time (Little's
Law: with lambda fixed, relieving the constraint shows up in W, not in TH). Both
are reported; cycle time is the informative channel.
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from factory_generator import default_config, draw_randoms, simulate


def steady_state_kpis(lifecycle: pd.DataFrame, t0: float, t1: float) -> tuple[float, float]:
    """(throughput_per_hour, mean_cycle_time_hours) over the window [t0, t1]."""
    window = t1 - t0
    done = lifecycle.dropna(subset=["completion_time"])
    in_win = done[(done["completion_time"] >= t0) & (done["completion_time"] <= t1)]
    ct = in_win["completion_time"] - in_win["arrival_time"]
    return len(in_win) / window, float(ct.mean())


def with_extra_tool(base_cfg, station: str, delta: int = 1):
    """Deep-copy the config and add ``delta`` tools at ``station`` (nothing else)."""
    cfg = copy.deepcopy(base_cfg)
    cfg.stations[station].n_tools += delta
    return cfg


def run_counterfactual(
    base_cfg=None,
    interventions: list[str] | None = None,
    n_reps: int = 30,
    seed0: int = 1000,
) -> pd.DataFrame:
    """Run the CRN paired counterfactual.

    For each of ``n_reps`` replications: draw ONE random table, run the baseline
    and every "+1 tool at station X" scenario on that table, and record the paired
    deltas (treatment - baseline) for throughput and cycle time.

    Returns a long-form DataFrame with one row per (replication, intervention):
    columns [rep, seed, intervention, base_throughput, base_cycle_time,
             d_throughput, d_cycle_time].
    """
    if base_cfg is None:
        base_cfg = default_config()
    if interventions is None:
        interventions = ["S4", "S3", "S7"]

    t0, t1 = base_cfg.warmup_hours, base_cfg.horizon_hours
    treat_cfgs = {s: with_extra_tool(base_cfg, s) for s in interventions}

    rows = []
    for rep in range(n_reps):
        seed = seed0 + rep
        draws = draw_randoms(base_cfg, seed=seed)      # shared table for this rep

        _, life_b, _ = simulate(base_cfg, draws)
        th_b, ct_b = steady_state_kpis(life_b, t0, t1)

        for s in interventions:
            _, life_t, _ = simulate(treat_cfgs[s], draws)   # same table, +1 tool at s
            th_t, ct_t = steady_state_kpis(life_t, t0, t1)
            rows.append({
                "rep": rep,
                "seed": seed,
                "intervention": f"{s}+1",
                "base_throughput": th_b,
                "base_cycle_time": ct_b,
                "d_throughput": th_t - th_b,
                "d_cycle_time": ct_t - ct_b,
            })

    return pd.DataFrame(rows)


def summarize(deltas: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean + 95% CI (paired t) of ``metric`` per intervention, preserving order."""
    order = list(dict.fromkeys(deltas["intervention"]))
    out = []
    for name, grp in deltas.groupby("intervention", sort=False):
        x = grp[metric].to_numpy()
        n = len(x)
        mean = float(x.mean())
        sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
        out.append({
            "intervention": name,
            "n": n,
            "mean": mean,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "ci95_half": half,
        })
    return (pd.DataFrame(out)
            .set_index("intervention")
            .reindex(order))


def plot_counterfactual(deltas: pd.DataFrame, save_path: str | None = None):
    """Two panels: Delta-throughput and Delta-cycle-time, mean + 95% CI error bars."""
    th = summarize(deltas, "d_throughput")
    ct = summarize(deltas, "d_cycle_time")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    labels = list(th.index)
    # Emphasise the bottleneck bar.
    colors = ["#EF5350" if lab.startswith("S4") else "#78909C" for lab in labels]

    for ax, tab, title, unit in [
        (axes[0], th, "Δ throughput (lots/hour)", "lots/h"),
        (axes[1], ct, "Δ cycle time (hours)", "h"),
    ]:
        means = tab["mean"].to_numpy()
        err = tab["ci95_half"].to_numpy()
        ax.bar(labels, means, yerr=err, capsize=6, color=colors,
               error_kw=dict(ecolor="#263238", lw=1.4))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        for i, (m, e) in enumerate(zip(means, err)):
            ax.annotate(f"{m:+.4f}\n±{e:.4f}" if unit == "lots/h" else f"{m:+.3f}\n±{e:.3f}",
                        (i, m), ha="center",
                        va="bottom" if m >= 0 else "top", fontsize=8)

    fig.suptitle(
        "CRN paired counterfactual (+1 tool), N="
        f"{summarize(deltas, 'd_throughput')['n'].iloc[0]} replications — "
        "improvement gradient S4 >> S3 > S7\n"
        "Quantifies decision impact / tests decision logic; does NOT independently "
        "prove S4 (S4 is a design input). Δ cycle time is the informative channel.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
