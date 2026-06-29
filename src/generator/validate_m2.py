"""
M2 validation (run end to end).

Checks two things that make the generator defensible:
  1. Little's Law self-consistency:  measured WIP  ~=  throughput * cycle time
  2. Ground-truth bottleneck recovery: the engineered station (S4) shows the
     highest empirical utilization, matching the design.

Also writes the synthetic event log and metadata to data/synthetic/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factory_generator import default_config, simulate, theoretical_utilization

OUT = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)


def empirical_utilization(log, cfg, t0, t1):
    """Busy-time fraction per station within [t0, t1], clipped to the window."""
    window = t1 - t0
    util = {}
    for s, st in cfg.stations.items():
        ops = log[log["station"] == s]
        start = ops["process_start_time"].clip(lower=t0, upper=t1)
        end = ops["process_complete_time"].clip(lower=t0, upper=t1)
        busy = (end - start).clip(lower=0).sum()
        util[s] = busy / (st.n_tools * window)
    return util


def measured_wip(lifecycle, t0, t1):
    """Time-average number of lots in system over [t0, t1]."""
    window = t1 - t0
    arr = lifecycle["arrival_time"].to_numpy()
    comp = lifecycle["completion_time"].fillna(t1).to_numpy()  # still-in-system -> t1
    lo = np.maximum(arr, t0)
    hi = np.minimum(comp, t1)
    overlap = np.clip(hi - lo, 0, None)
    return overlap.sum() / window


def main():
    cfg = default_config()
    log, lifecycle, meta = simulate(cfg)

    t0, t1 = cfg.warmup_hours, cfg.horizon_hours
    window = t1 - t0

    # Lots that COMPLETED within the steady-state window.
    done = lifecycle.dropna(subset=["completion_time"])
    in_win = done[(done["completion_time"] >= t0) & (done["completion_time"] <= t1)].copy()
    in_win["cycle_time"] = in_win["completion_time"] - in_win["arrival_time"]

    throughput = len(in_win) / window           # lots per hour
    avg_ct = in_win["cycle_time"].mean()        # hours
    wip_ll = throughput * avg_ct                # Little's Law prediction
    wip_meas = measured_wip(lifecycle, t0, t1)  # measured time-average WIP
    ll_gap = abs(wip_meas - wip_ll) / wip_meas

    theo = theoretical_utilization(cfg)
    emp = empirical_utilization(log, cfg, t0, t1)
    emp_bottleneck = max(emp, key=emp.get)

    print("=" * 60)
    print("M2 GENERATOR VALIDATION")
    print("=" * 60)
    print(f"Lots completed in window : {len(in_win)}")
    print(f"Throughput (lots/hour)   : {throughput:.3f}")
    print(f"Avg cycle time (hours)   : {avg_ct:.2f}")
    print()
    print("--- Little's Law (WIP = throughput x cycle time) ---")
    print(f"WIP predicted (TH x CT)  : {wip_ll:.2f}")
    print(f"WIP measured (time-avg)  : {wip_meas:.2f}")
    print(f"Relative gap             : {ll_gap*100:.2f}%  "
          f"({'PASS' if ll_gap < 0.05 else 'CHECK'})")
    print()
    print("--- Utilization per station (theoretical vs empirical) ---")
    print(f"{'station':<8}{'planned':>10}{'observed':>10}")
    for s in cfg.stations:
        print(f"{s:<8}{theo[s]:>10.3f}{emp[s]:>10.3f}")
    print()
    print(f"Designed bottleneck   : {meta['ground_truth_bottleneck']}")
    print(f"Empirical bottleneck  : {emp_bottleneck}  "
          f"({'PASS' if emp_bottleneck == meta['ground_truth_bottleneck'] else 'CHECK'})")

    # Persist outputs.
    log.to_csv(OUT / "event_log.csv", index=False)
    lifecycle.to_csv(OUT / "lot_lifecycle.csv", index=False)
    meta_out = dict(meta)
    meta_out["validation"] = {
        "throughput_per_hour": throughput,
        "avg_cycle_time_hours": avg_ct,
        "wip_littles_law": wip_ll,
        "wip_measured": wip_meas,
        "littles_law_gap": ll_gap,
        "empirical_utilization": emp,
        "empirical_bottleneck": emp_bottleneck,
    }
    with open(OUT / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print()
    print(f"Saved: {OUT/'event_log.csv'}  ({len(log)} rows)")
    print(f"Saved: {OUT/'lot_lifecycle.csv'}  ({len(lifecycle)} lots)")
    print(f"Saved: {OUT/'metadata.json'}")


if __name__ == "__main__":
    main()
