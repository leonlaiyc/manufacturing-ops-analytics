"""
M8 Stage B regression + sanity check (run end to end).

Confirms the maintenance-timing what-if (``maintenance_whatif.py``) and the
alert-priority score (``alert_priority.py``) behave correctly:

  GATE 1 - reproducibility: two runs of ``demo_litho_pm_timing`` (fixed seeds)
           produce an identical comparison table.
  GATE 2 - exact pairing: a baseline-vs-baseline paired comparison (the SAME
           anomaly list, "baseline", passed twice through
           ``compare_pm_timings`` by reusing its own no-degradation reference
           row across independent calls) gives EXACTLY zero delta on cycle
           time, violation rate, and yield in every replication.
  GATE 3 - delay direction: in the reference demonstration, the LATE-PM total
           cost exceeds the IMMEDIATE-PM total cost (directional, seeded;
           strict monotonicity across all three timings is NOT required, see
           spec).
  GATE 4 - priority ordering: LITHO's same-severity priority score exceeds
           METRO's, AND the realized CRN-paired total-cost impact of LITHO
           degradation exceeds that of METRO degradation (the score's ranking
           claim is checked against simulated reality, not assumed).

Run:  py src/equipment/maintenance_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
for sub in ("generator", "quality", "decision", "kpi", "equipment"):
    sys.path.insert(0, str(SRC / sub))

from factory_generator import default_config, draw_randoms, simulate
from cost_model import CostRates, cost_components
from maintenance_whatif import (
    compare_pm_timings, summarize_pm_comparison, demo_litho_pm_timing,
    build_pm_scenario,
)
from alert_priority import compare_bottleneck_vs_nonbottleneck


def main() -> int:
    cfg = default_config()   # locked 60-day, seed 42 line
    ok = True

    print("=" * 64)
    print("GATE 1 - reproducibility (identical comparison table, repeated runs)")
    print("=" * 64)
    summary_a = demo_litho_pm_timing(cfg=cfg, n_reps=15, seed0=6000)
    summary_b = demo_litho_pm_timing(cfg=cfg, n_reps=15, seed0=6000)
    g1 = summary_a.equals(summary_b)
    print(f"  two runs of demo_litho_pm_timing are identical : {g1}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 - exact pairing (baseline vs baseline, paired deltas == 0)")
    print("=" * 64)
    # Two independent calls with the SAME pm_times dict content but scored as
    # two separate "scenario sets" sharing draws: compare the "baseline" row
    # of one call against the "baseline" row of the other, replication by
    # replication. Both come from the exact same anomaly-free simulate() call
    # on the exact same draw table (seed0 + rep), so equality must be exact.
    paired_1 = compare_pm_timings(cfg=cfg, n_reps=5, seed0=6000)
    paired_2 = compare_pm_timings(cfg=cfg, n_reps=5, seed0=6000)
    base_1 = paired_1[paired_1["scenario"] == "baseline"].reset_index(drop=True)
    base_2 = paired_2[paired_2["scenario"] == "baseline"].reset_index(drop=True)
    d_ct = (base_1["mean_cycle_time"] - base_2["mean_cycle_time"]).abs().max()
    d_viol = (base_1["violation_rate"] - base_2["violation_rate"]).abs().max()
    d_yield = (base_1["mean_lot_yield"] - base_2["mean_lot_yield"]).abs().max()
    g2 = (d_ct == 0.0) and (d_viol == 0.0) and (d_yield == 0.0)
    print(f"  max |delta cycle time|      = {d_ct:.2e}")
    print(f"  max |delta violation rate|  = {d_viol:.2e}")
    print(f"  max |delta mean lot yield|  = {d_yield:.2e}")
    print(f"  baseline-vs-baseline exactly zero on all three : {g2}")
    ok &= g2

    print("=" * 64)
    print("GATE 3 - delay direction (LATE total cost > IMMEDIATE total cost)")
    print("=" * 64)
    late_cost = float(summary_a.loc[summary_a["scenario"] == "late_pm", "total_cost"].iloc[0])
    immediate_cost = float(summary_a.loc[summary_a["scenario"] == "immediate_pm",
                                        "total_cost"].iloc[0])
    mid_cost = float(summary_a.loc[summary_a["scenario"] == "mid_pm", "total_cost"].iloc[0])
    baseline_cost = float(summary_a.loc[summary_a["scenario"] == "baseline",
                                        "total_cost"].iloc[0])
    g3 = late_cost > immediate_cost
    print(f"  baseline total cost   = {baseline_cost:10.2f}")
    print(f"  immediate_pm total cost = {immediate_cost:10.2f}")
    print(f"  mid_pm total cost       = {mid_cost:10.2f}")
    print(f"  late_pm total cost      = {late_cost:10.2f}")
    print(f"  late_pm > immediate_pm (directional, seeded) : {g3}")
    ok &= g3

    print("=" * 64)
    print("GATE 4 - priority ordering (score AND realized cost impact agree)")
    print("=" * 64)
    rates = CostRates()
    alpha = 0.01
    p_litho, p_metro = compare_bottleneck_vs_nonbottleneck(
        alpha=alpha, bottleneck_station="LITHO", nonbottleneck_station="METRO",
        cfg=cfg, rates=rates)
    g4a = p_litho.priority > p_metro.priority
    print(f"  LITHO priority = {p_litho.priority:.4f} "
          f"(severity={p_litho.severity:.4f}, "
          f"criticality={p_litho.bottleneck_criticality:.4f}, "
          f"cost/h={p_litho.cost_exposure_per_hour:.2f})")
    print(f"  METRO priority = {p_metro.priority:.4f} "
          f"(severity={p_metro.severity:.4f}, "
          f"criticality={p_metro.bottleneck_criticality:.4f}, "
          f"cost/h={p_metro.cost_exposure_per_hour:.2f})")
    print(f"  LITHO score > METRO score : {g4a}")

    # Realized CRN-paired cost impact: same-severity degradation with a fixed
    # late PM time, LITHO vs METRO, against the same no-degradation baseline,
    # on shared draw tables (rep-by-rep pairing as in compare_pm_timings).
    n_reps_g4 = 15
    seed0_g4 = 7000
    t_onset = 20 * 24.0
    pm_time = 45 * 24.0
    pm_duration = 24.0
    t0, t1 = cfg.warmup_hours, cfg.horizon_hours

    def _mean_cost_impact(station: str) -> float:
        deltas = []
        for rep in range(n_reps_g4):
            draws = draw_randoms(cfg, seed0_g4 + rep)
            log_b, _, _ = simulate(cfg, draws)
            anomalies = build_pm_scenario(station, t_onset, alpha, pm_time, pm_duration)
            log_s, _, _ = simulate(cfg, draws, anomalies=anomalies)
            cost_b = cost_components(log_b, t0, t1, rates)["total"]
            cost_s = cost_components(log_s, t0, t1, rates)["total"]
            deltas.append(cost_s - cost_b)
        return float(sum(deltas) / len(deltas))

    impact_litho = _mean_cost_impact("LITHO")
    impact_metro = _mean_cost_impact("METRO")
    g4b = impact_litho > impact_metro
    print(f"  realized mean cost impact LITHO degradation = {impact_litho:10.2f}")
    print(f"  realized mean cost impact METRO degradation = {impact_metro:10.2f}")
    print(f"  realized LITHO impact > realized METRO impact : {g4b}")

    g4 = g4a and g4b
    print(f"  GATE 4 (score ranking matches realized impact ranking) : {g4}")
    ok &= g4

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
