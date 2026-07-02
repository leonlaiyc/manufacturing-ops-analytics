"""
Capacity / demand / cost what-if decision support (M6).

Built on the M4 CRN counterfactual harness (imported, not rewritten): one shared
draw table per replication, baseline vs treatment on that same table, paired Δ,
mean + 95% CI. This module extends it to demand scaling, degradation, and cost,
and assembles the prescriptive improvement ranking.

Reused from bottleneck/counterfactual.py:
  steady_state_kpis, with_extra_tool, summarize.
Reused from generator/factory_generator.py:
  default_config, draw_randoms, simulate, theoretical_utilization, DegradationAnomaly.
Cost from decision/cost_model.py.

Consumers must have src/generator, src/bottleneck, src/monitoring, src/decision on
sys.path.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from factory_generator import (
    default_config, draw_randoms, simulate, theoretical_utilization,
    DegradationAnomaly,
)
from counterfactual import steady_state_kpis, with_extra_tool, summarize
from cost_model import CostRates, cost_components


def with_demand(cfg, factor: float):
    """Deep-copy the config and scale the arrival rate by ``factor`` (nothing else)."""
    c = copy.deepcopy(cfg)
    c.arrival_rate *= factor
    return c


# --------------------------------------------------------------------------- #
# Step 1 — capacity what-if with cost
# --------------------------------------------------------------------------- #
def run_capacity_cost(base_cfg=None, stations=("S4", "S3", "S7"),
                      n_reps: int = 30, seed0: int = 1000,
                      rates: CostRates | None = None) -> pd.DataFrame:
    """Paired Δ (cycle time, throughput, total cost) from +1 tool per station.

    Δcost includes the added tool's capacity cost, so it is the net cost change of
    the decision (machine cost minus any holding-cost savings).
    """
    base_cfg = base_cfg or default_config()
    rates = rates or CostRates()
    t0, t1 = base_cfg.warmup_hours, base_cfg.horizon_hours
    treat = {s: with_extra_tool(base_cfg, s) for s in stations}
    rows = []
    for rep in range(n_reps):
        seed = seed0 + rep
        draws = draw_randoms(base_cfg, seed)
        log_b, life_b, _ = simulate(base_cfg, draws)
        th_b, ct_b = steady_state_kpis(life_b, t0, t1)
        cost_b = cost_components(log_b, t0, t1, rates, tools_added=0)["total"]
        for s in stations:
            log_t, life_t, _ = simulate(treat[s], draws)
            th_t, ct_t = steady_state_kpis(life_t, t0, t1)
            cost_t = cost_components(log_t, t0, t1, rates, tools_added=1)["total"]
            rows.append({"rep": rep, "intervention": f"{s}+1",
                         "d_cycle_time": ct_t - ct_b,
                         "d_throughput": th_t - th_b,
                         "d_cost": cost_t - cost_b})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 2 — demand what-if (descriptive)
# --------------------------------------------------------------------------- #
def utilization_vs_demand(base_cfg, factors) -> pd.DataFrame:
    """Analytic per-station utilization rho = arrival*factor*visits*pt/n_tools.

    Exact (no simulation) — identifies which station reaches rho>=1 first.
    Returns a DataFrame indexed by station, one column per demand factor.
    """
    base_cfg = base_cfg or default_config()
    out = {}
    for f in factors:
        out[f] = theoretical_utilization(with_demand(base_cfg, f))
    return pd.DataFrame(out)


def run_demand_absolute(base_cfg=None, factors=(1.0, 1.10, 1.25, 1.50),
                        n_reps: int = 30, seed0: int = 1000) -> pd.DataFrame:
    """Absolute baseline throughput and mean cycle time at each demand level.

    At factors that push the bottleneck to rho>=1 the line is unstable, so mean
    cycle time is measured on completed lots and rises non-linearly (documented in
    the notebook); throughput saturates at the bottleneck's capacity.
    """
    base_cfg = base_cfg or default_config()
    rows = []
    for f in factors:
        cfg_f = with_demand(base_cfg, f)
        t0, t1 = cfg_f.warmup_hours, cfg_f.horizon_hours
        for rep in range(n_reps):
            draws = draw_randoms(cfg_f, seed0 + rep)
            _, life, _ = simulate(cfg_f, draws)
            th, ct = steady_state_kpis(life, t0, t1)
            rows.append({"factor": f, "rep": rep, "throughput": th, "cycle_time": ct})
    return pd.DataFrame(rows)


def run_demand_capacity(base_cfg=None, factors=(1.0, 1.10, 1.25, 1.50),
                        n_reps: int = 30, seed0: int = 1000,
                        station: str = "S4") -> pd.DataFrame:
    """Paired Δthroughput from +1 tool at the bottleneck, per demand level.

    Closes the M4 thread: Δthroughput ~= 0 while arrival-limited (rho<1), then
    becomes clearly positive once the bottleneck saturates (rho>=1) and capacity,
    not demand, limits output.
    """
    base_cfg = base_cfg or default_config()
    rows = []
    for f in factors:
        cfg_f = with_demand(base_cfg, f)
        cfg_ft = with_extra_tool(cfg_f, station)
        t0, t1 = cfg_f.warmup_hours, cfg_f.horizon_hours
        for rep in range(n_reps):
            draws = draw_randoms(cfg_f, seed0 + rep)     # same table, baseline vs +1 tool
            _, life_b, _ = simulate(cfg_f, draws)
            _, life_t, _ = simulate(cfg_ft, draws)
            th_b, _ = steady_state_kpis(life_b, t0, t1)
            th_t, _ = steady_state_kpis(life_t, t0, t1)
            rows.append({"factor": f, "rep": rep,
                         "base_throughput": th_b, "d_throughput": th_t - th_b})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3 — degradation impact (paired clean vs degrading)
# --------------------------------------------------------------------------- #
def run_degradation_impact(cfg, deg: DegradationAnomaly, n_reps: int = 30,
                           seed0: int = 3000, rates: CostRates | None = None) -> pd.DataFrame:
    """Paired Δ (cost, cycle time, throughput) of a degrading bottleneck vs clean."""
    rates = rates or CostRates()
    t0, t1 = cfg.warmup_hours, cfg.horizon_hours
    rows = []
    for rep in range(n_reps):
        draws = draw_randoms(cfg, seed0 + rep)
        log_c, life_c, _ = simulate(cfg, draws)
        log_d, life_d, _ = simulate(cfg, draws, anomalies=[deg])
        th_c, ct_c = steady_state_kpis(life_c, t0, t1)
        th_d, ct_d = steady_state_kpis(life_d, t0, t1)
        cost_c = cost_components(log_c, t0, t1, rates)["total"]
        cost_d = cost_components(log_d, t0, t1, rates)["total"]
        rows.append({"rep": rep, "intervention": "degradation",
                     "d_cost": cost_d - cost_c,
                     "d_cycle_time": ct_d - ct_c, "d_throughput": th_d - th_c})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 4 — improvement trade-off (fixed output, cost-only). RAW quantities so the
# +/-50% sensitivity re-costs the SAME simulated runs without re-simulating.
# --------------------------------------------------------------------------- #
def _raw_quantities(log, t0, t1, tools_added, repairs, throughput):
    proc_h = float((log["process_complete_time"].clip(lower=t0, upper=t1)
                    - log["process_start_time"].clip(lower=t0, upper=t1)).clip(lower=0).sum())
    wait_h = float((log["process_start_time"].clip(lower=t0, upper=t1)
                    - log["queue_entry_time"].clip(lower=t0, upper=t1)).clip(lower=0).sum())
    return {"proc_hours": proc_h, "wait_hours": wait_h,
            "tools_added": tools_added, "repairs": repairs, "throughput": throughput}


def run_improvement_raw(cfg, deg: DegradationAnomaly, detection_day: float,
                        n_reps: int = 30, seed0: int = 4000,
                        nonbottleneck: str = "S3") -> pd.DataFrame:
    """Simulate the improvement options against a degrading-bottleneck baseline.

    All options deliver the same output (the line stays arrival-limited), so they
    are compared purely on cost (fixed-output framework). Options:
      - do_nothing        : let S4 degrade for the whole horizon.
      - add_S4_capacity   : +1 tool at the bottleneck (compensate the degradation).
      - add_{nonbottleneck}: +1 tool at a non-bottleneck (spend, little benefit).
      - early_fix         : detect (M5) and repair the degradation at detection_day
                            (degradation window ends there) + a one-off repair cost.
    Returns RAW proc/wait hours + tools/repairs + throughput per (rep, option) so
    costs can be applied afterwards for any rate set.
    """
    t0, t1 = cfg.warmup_hours, cfg.horizon_hours
    cfg_s4 = with_extra_tool(cfg, "S4")
    cfg_nb = with_extra_tool(cfg, nonbottleneck)
    deg_fixed = DegradationAnomaly(deg.station, deg.t_onset,
                                   detection_day * 24.0, deg.alpha)  # repaired at detection
    rows = []
    for rep in range(n_reps):
        draws = draw_randoms(cfg, seed0 + rep)
        specs = {
            "do_nothing":  (cfg,    [deg],       0, 0),
            "add_S4":      (cfg_s4, [deg],       1, 0),
            f"add_{nonbottleneck}": (cfg_nb, [deg], 1, 0),
            "early_fix":   (cfg,    [deg_fixed], 0, 1),
        }
        for opt, (c, anoms, tools, reps_) in specs.items():
            log, life, _ = simulate(c, draws, anomalies=anoms)
            th, _ = steady_state_kpis(life, t0, t1)
            rows.append({"rep": rep, "option": opt,
                         **_raw_quantities(log, t0, t1, tools, reps_, th)})
    return pd.DataFrame(rows)


def apply_cost(raw: pd.DataFrame, rates: CostRates) -> pd.DataFrame:
    """Add a total_cost column to raw improvement quantities for a given rate set."""
    df = raw.copy()
    df["total_cost"] = (df["proc_hours"] * rates.proc_rate
                        + df["wait_hours"] * rates.hold_rate
                        + df["tools_added"] * rates.tool_cost
                        + df["repairs"] * rates.repair_cost)
    return df


def tradeoff_summary(raw: pd.DataFrame, rates: CostRates) -> pd.DataFrame:
    """Mean total cost + 95% CI per option (lowest = recommended)."""
    costed = apply_cost(raw, rates)
    costed = costed.rename(columns={"option": "intervention"})
    s = summarize(costed, "total_cost")
    return s.sort_values("mean")


def recommended_option(raw: pd.DataFrame, rates: CostRates) -> str:
    return tradeoff_summary(raw, rates).index[0]


def sensitivity_recommendation(raw: pd.DataFrame, base_rates: CostRates,
                               scales=(0.5, 1.5)) -> pd.DataFrame:
    """Re-cost the SAME runs with each rate scaled +/-50% one at a time.

    Returns a table [rate, scale, recommended_option, <option means...>] so the
    notebook can show whether the recommendation flips.
    """
    fields = ["proc_rate", "hold_rate", "tool_cost", "repair_cost"]
    rows = [{"scenario": "baseline", "recommended": recommended_option(raw, base_rates),
             **tradeoff_summary(raw, base_rates)["mean"].to_dict()}]
    for f in fields:
        for sc in scales:
            r = CostRates(**{**base_rates.__dict__})
            setattr(r, f, getattr(base_rates, f) * sc)
            rows.append({"scenario": f"{f} x{sc}",
                         "recommended": recommended_option(raw, r),
                         **tradeoff_summary(raw, r)["mean"].to_dict()})
    return pd.DataFrame(rows)
