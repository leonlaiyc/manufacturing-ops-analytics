"""
Precompute the numbers/series behind the three findings for the published
index visuals and related static evidence assets.

Reuses the project's own validated modules (src/bottleneck/counterfactual.py,
src/decision/whatif.py, src/decision/cost_model.py, src/monitoring/*) with the
exact parameters used in notebooks 04 and 06, so the recomputed numbers match
the published ones under CRN + fixed seeds. Nothing here re-derives a method -
it only calls the existing functions and caches their outputs.

Writes data/synthetic/findings_cache.json (gitignored, like the rest of
data/**). Static index assets and legacy finding helpers read this cache.

Run:  py src/kpi/precompute_findings.py

Import-order note: pandas' optional "bottleneck" perf package gets shadowed by
this project's src/bottleneck/ package if src/bottleneck is already on
sys.path when pandas first imports. So pandas/numpy are imported BEFORE any
src/* path insertion (same fix the notebooks use by never putting plain "src"
on the path as a package root).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/generator", "src/bottleneck", "src/monitoring", "src/decision"):
    sys.path.insert(0, str(ROOT / p))
sys.path.append(str(ROOT / "src"))  # for "kpi.kpi_metrics" (no shadowing risk)

from factory_generator import default_config, draw_randoms, simulate, DegradationAnomaly  # noqa: E402
from counterfactual import run_counterfactual, summarize  # noqa: E402
from m5_config import m5_config  # noqa: E402
import kpi_series as ks  # noqa: E402
import detectors as det  # noqa: E402
import whatif as wf  # noqa: E402
from cost_model import CostRates, daily_operating_cost  # noqa: E402

OUT = ROOT / "data" / "synthetic" / "findings_cache.json"

N_REPS = 30
STRESS_TOOL_COSTS = {"LITHO": 40000.0, "FURNACE": 8000.0, "DEPO": 5000.0, "METRO": 2000.0}
ORDER = ["LITHO+1", "FURNACE+1", "DEPO+1", "METRO+1"]
STATIONS = ["LITHO", "FURNACE", "DEPO", "METRO"]

ONSET_DAY = 30
HORIZON_DAYS = 160
WARMUP_DAYS = 6


def finding_01() -> dict:
    """CRN-paired +1 tool counterfactual (mirrors notebook 04, Step 3)."""
    print("  [1/3] CRN counterfactual (+1 tool at LITHO/DEPO/FURNACE/METRO, N=30) ...")
    deltas = run_counterfactual(
        default_config(), interventions=STATIONS, n_reps=N_REPS, seed0=1000,
    )
    ct = summarize(deltas, "d_cycle_time").reindex(ORDER)
    out = {
        "stations": STATIONS,
        "delta_mean": (-ct["mean"]).round(6).tolist(),          # positive = hours saved
        "ci_low": (-ct["ci95_high"]).round(6).tolist(),         # note the sign flip
        "ci_high": (-ct["ci95_low"]).round(6).tolist(),
        "n_reps": N_REPS,
    }
    litho_delta = round(float(-ct.loc["LITHO+1", "mean"]), 2)
    assert litho_delta == 2.46, (
        f"Sanity gate failed: recomputed LITHO delta mean cycle time = {litho_delta} h, "
        f"expected 2.46 h. Parameters have diverged from notebook 04 Step 3 - stop and fix "
        f"before trusting the dashboard numbers."
    )
    print(f"        LITHO delta = {litho_delta} h (matches published 2.46 h)")
    return out


def finding_02() -> dict:
    """Equal-cost vs station-specific-cost net cost change (mirrors notebook 06, Step 1)."""
    print("  [2/3] Capacity cost what-if: base case + investment-stress scenario ...")
    rates = CostRates()
    cap_base = wf.run_capacity_cost(default_config(), n_reps=N_REPS, rates=rates)
    cap_stress = wf.run_capacity_cost(
        default_config(), n_reps=N_REPS, rates=rates, added_tool_costs=STRESS_TOOL_COSTS,
    )
    base_cost = summarize(cap_base, "d_cost").reindex(ORDER)
    stress_cost = summarize(cap_stress, "d_cost").reindex(ORDER)
    break_even = rates.tool_cost - base_cost["mean"]

    litho_stress = round(float(stress_cost.loc["LITHO+1", "mean"]) / 1000, 1)
    furnace_stress = round(float(stress_cost.loc["FURNACE+1", "mean"]) / 1000, 1)
    litho_base = round(float(base_cost.loc["LITHO+1", "mean"]) / 1000, 1)
    assert litho_stress == 8.0, (
        f"Sanity gate failed: recomputed LITHO investment-stress net cost = "
        f"{litho_stress}k, expected 8.0k. Cost model params diverged from notebook 06 Step 1."
    )
    assert furnace_stress == -1.7, (
        f"Sanity gate failed: recomputed FURNACE investment-stress net cost = "
        f"{furnace_stress}k, expected -1.7k."
    )
    print(f"        LITHO base = {litho_base}k, LITHO stress = {litho_stress}k, "
          f"FURNACE stress = {furnace_stress}k (match published)")

    return {
        "stations": STATIONS,
        "base_cost": base_cost["mean"].round(2).tolist(),
        "stress_cost": stress_cost["mean"].round(2).tolist(),
        "break_even": break_even.round(2).tolist(),
        "stress_tool_costs": [STRESS_TOOL_COSTS[s] for s in STATIONS],
        "n_reps": N_REPS,
    }


def finding_03() -> dict:
    """Degradation backtest: EWMA vs output-only monitor (mirrors notebook 06, Step 3)."""
    print("  [3/3] Degradation backtest (160-day horizon, clean-twin CRN pair) ...")
    cfg = m5_config(horizon_days=HORIZON_DAYS, warmup_days=WARMUP_DAYS)
    deg = DegradationAnomaly("LITHO", ONSET_DAY * 24.0, cfg.horizon_hours, alpha=5e-5)
    base_days = list(range(WARMUP_DAYS, ONSET_DAY - 2))

    draws_ref = draw_randoms(cfg, 4000)
    log_c0, life_c0, _ = simulate(cfg, draws_ref)
    log_d0, life_d0, _ = simulate(cfg, draws_ref, anomalies=[deg])
    clean_daily = ks.daily_kpis(log_c0, life_c0, cfg)
    deg_daily = ks.daily_kpis(log_d0, life_d0, cfg)

    dser = deg_daily["cycle_time"]
    c_, sg_ = det.fit_baseline(dser, base_days)
    ew = det.ewma_chart(dser, c_, sg_, lam=0.2, L=3.0)
    alert_days = [int(d) for d in ew.index[ew["alarm"]] if d >= ONSET_DAY]
    det_day = alert_days[0]
    assert det_day == 84, (
        f"Sanity gate failed: recomputed EWMA alert day = {det_day}, expected 84. "
        f"M5 config / degradation params diverged from notebook 06 Step 3."
    )
    print(f"        EWMA alert day = {det_day} (matches published day 84)")

    thr = deg_daily["throughput"]
    out_center, out_sigma = det.fit_baseline(thr, base_days)
    out_chart = det.control_chart(thr, out_center, out_sigma, k=3.0)
    out_lcl = float(out_chart["lcl"].iloc[0])
    output_alerts = [
        int(d) for d in out_chart.index[(thr < out_lcl) & (out_chart.index >= ONSET_DAY)]
    ]

    rates = CostRates()
    clean_cum = daily_operating_cost(log_c0, cfg.horizon_hours, rates).cumsum()
    deg_cum = daily_operating_cost(log_d0, cfg.horizon_hours, rates).cumsum()
    extra_end = float(deg_cum.iloc[-1] - clean_cum.iloc[-1])
    extra_at_det = float(deg_cum.loc[det_day] - clean_cum.loc[det_day])
    avoided = extra_end - extra_at_det
    avoided_pct = avoided / extra_end * 100

    extra_end_k = round(extra_end / 1000)
    assert extra_end_k == 249, (
        f"Sanity gate failed: recomputed extra cost = ${extra_end:,.0f}, expected ~$249k."
    )
    print(f"        extra cost = ${extra_end:,.0f}, avoidable = ${avoided:,.0f} "
          f"({avoided_pct:.1f}%) (match published ~$249k / ~95%)")

    return {
        "onset_day": ONSET_DAY,
        "alert_day": det_day,
        "horizon_days": HORIZON_DAYS,
        "output_only_alert_day": (output_alerts[0] if output_alerts else None),
        "day": list(range(HORIZON_DAYS)),
        "clean_output": [None if pd.isna(v) else float(v) for v in clean_daily["throughput"]],
        "deg_output": [None if pd.isna(v) else float(v) for v in deg_daily["throughput"]],
        "clean_cycle_time": [None if pd.isna(v) else round(float(v), 3) for v in clean_daily["cycle_time"]],
        "deg_cycle_time": [None if pd.isna(v) else round(float(v), 3) for v in deg_daily["cycle_time"]],
        "extra_cost_total": round(extra_end, 2),
        "avoidable_cost": round(avoided, 2),
        "avoidable_pct": round(avoided_pct, 1),
    }


def main() -> None:
    t0 = time.time()
    print("Precomputing findings cache (reusing project's bottleneck/decision/monitoring modules) ...")
    cache = {
        "finding_01": finding_01(),
        "finding_02": finding_02(),
        "finding_03": finding_03(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
