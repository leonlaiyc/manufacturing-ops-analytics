"""
M5 injection regression + sanity check (run end to end).

Confirms the anomaly-injection hook does not disturb M2-M4 and behaves correctly:

  GATE 1 — empty-anomaly identity: simulate(cfg, draws, anomalies=[]) is the
           un-injected path and equals simulate(cfg, draws) exactly.
  GATE 2 — inactive-anomaly identity: an anomaly whose window is entirely after
           the horizon drives the INJECTED code path but, being always identity,
           reproduces the clean run physically (proves the injected dispatch
           reduces to the plain one when nothing is active).
  GATE 3 — causality: for a real injected scenario, every event before the first
           anomaly's t_start is identical to the CRN-paired clean twin (no effect
           precedes its injection).
  GATE 4 — effect present & correctly signed: after onset, breakdown and
           degradation raise the bottleneck's cycle time / waiting, and the demand
           surge adds lots.

The byte-identical M2-M4 guarantee itself (Little's Law, S4 recovery, CRN Δ=0) is
covered by src/generator/crn_check.py and validate_m2.py, which are unaffected
because anomalies default to None. Run those too.

Run:  python src/monitoring/monitoring_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / "generator"))
sys.path.insert(0, str(SRC / "monitoring"))

from factory_generator import (
    default_config, draw_randoms, simulate,
    BreakdownAnomaly, DegradationAnomaly, DemandSurgeAnomaly,
)


def _cycle_time(life, t0, t1):
    done = life.dropna(subset=["completion_time"])
    w = done[(done["completion_time"] >= t0) & (done["completion_time"] <= t1)]
    return float((w["completion_time"] - w["arrival_time"]).mean())


def main() -> int:
    cfg = default_config()                       # 60-day line, fast
    T1 = cfg.horizon_hours
    draws = draw_randoms(cfg, seed=42)
    log_clean, life_clean, _ = simulate(cfg, draws)
    ok = True

    print("=" * 64)
    print("GATE 1 — empty-anomaly identity")
    print("=" * 64)
    log_e, _, _ = simulate(cfg, draws, anomalies=[])
    g1 = log_e.equals(log_clean)
    print(f"  simulate(anomalies=[]) == clean : {g1}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 — inactive-anomaly injected path == clean (physical)")
    print("=" * 64)
    inactive = BreakdownAnomaly("S4", T1 + 100, T1 + 200, tools_removed=2)
    log_i, _, _ = simulate(cfg, draws, anomalies=[inactive])
    g2 = log_i.equals(log_clean)
    print(f"  injected-but-inactive == clean  : {g2}")
    ok &= g2

    print("=" * 64)
    print("GATE 3 — causality (identical before first t_start)")
    print("=" * 64)
    ts = 30 * 24.0
    scenario = [
        BreakdownAnomaly("S4", ts, ts + 2 * 24, tools_removed=1),
        DemandSurgeAnomaly(40 * 24.0, 44 * 24.0, extra_rate=0.5, seed=7),
        DegradationAnomaly("S4", 46 * 24.0, 55 * 24.0, alpha=0.0005),
    ]
    log_a, life_a, meta_a = simulate(cfg, draws, anomalies=scenario)
    before_c = log_clean[log_clean["process_complete_time"] < ts].reset_index(drop=True)
    before_a = log_a[log_a["process_complete_time"] < ts].reset_index(drop=True)
    g3 = before_c.equals(before_a)
    print(f"  {len(before_a)} ops before t_start identical : {g3}")
    ok &= g3

    print("=" * 64)
    print("GATE 4 — effect present & correctly signed")
    print("=" * 64)
    ct_c = _cycle_time(life_clean, cfg.warmup_hours, T1)
    ct_a = _cycle_time(life_a, cfg.warmup_hours, T1)
    g4a = ct_a > ct_c
    print(f"  cycle time clean {ct_c:.2f}h -> anomalous {ct_a:.2f}h (up): {g4a}")
    # demand surge adds lots
    life_su_lots = len(simulate(cfg, draws,
                                anomalies=[DemandSurgeAnomaly(40 * 24.0, 44 * 24.0,
                                                              extra_rate=0.5, seed=7)])[1])
    g4b = life_su_lots > len(life_clean)
    print(f"  demand surge lots {len(life_clean)} -> {life_su_lots} (more): {g4b}")
    ok &= g4a and g4b

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    print("Reminder: also run src/generator/crn_check.py and validate_m2.py "
          "(unaffected by anomalies=None).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
