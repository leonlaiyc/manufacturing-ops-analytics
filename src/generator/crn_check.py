"""
CRN refactor verification (M4, Phase A) — run end to end.

Two hard gates required before the counterfactual can be trusted:

  GATE 1 — Legacy byte-identical regression.
      The ``draws=None`` path must be byte-for-byte identical to the pre-refactor
      generator. We regenerate the default-config log/lifecycle on the legacy path
      and compare the serialized CSV bytes against the committed golden files in
      ``data/synthetic/``. (Little's-Law gap < 1% and S4 recovery are re-checked by
      ``validate_m2.py``, which uses the same untouched legacy path.)

  GATE 2 — CRN determinism sanity (the key check).
      Using ONE pre-drawn ``RandomDraws`` table, run baseline twice: the two runs
      must be identical, so Δthroughput and Δcycle-time are EXACTLY 0.0. A nonzero
      delta would mean some RNG source is still escaping the table and CRN pairing
      is not actually in force. As a positive control we also confirm that a real
      intervention (S4 +1 tool) on the SAME table does move the KPIs (Δ != 0), and
      that a CRN run still recovers S4 as the empirical bottleneck.

Run:  python src/generator/crn_check.py
Exit code 0 = all gates pass; nonzero = a gate failed.
"""

from __future__ import annotations

import copy
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from factory_generator import (
    default_config,
    draw_randoms,
    simulate,
    theoretical_utilization,
)

SYN = Path(__file__).resolve().parents[2] / "data" / "synthetic"


# --------------------------------------------------------------------------- #
# Small, self-contained KPI helpers (steady-state window [warmup, horizon]).
# Same formulas as validate_m2.main; duplicated here so the test stands alone.
# --------------------------------------------------------------------------- #
def run_kpis(lifecycle: pd.DataFrame, t0: float, t1: float) -> tuple[float, float]:
    """Return (throughput_per_hour, mean_cycle_time_hours) over [t0, t1]."""
    window = t1 - t0
    done = lifecycle.dropna(subset=["completion_time"])
    in_win = done[(done["completion_time"] >= t0) & (done["completion_time"] <= t1)]
    ct = in_win["completion_time"] - in_win["arrival_time"]
    throughput = len(in_win) / window
    mean_ct = float(ct.mean())
    return throughput, mean_ct


def empirical_bottleneck(log: pd.DataFrame, cfg, t0: float, t1: float) -> str:
    """Station with the highest busy-time fraction in [t0, t1]."""
    window = t1 - t0
    util = {}
    for s, st in cfg.stations.items():
        ops = log[log["station"] == s]
        start = ops["process_start_time"].clip(lower=t0, upper=t1)
        end = ops["process_complete_time"].clip(lower=t0, upper=t1)
        busy = (end - start).clip(lower=0).sum()
        util[s] = busy / (st.n_tools * window)
    return max(util, key=util.get)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# GATE 1 — legacy byte-identical regression
# --------------------------------------------------------------------------- #
def gate1_legacy_byte_identical() -> bool:
    print("=" * 64)
    print("GATE 1 — Legacy (draws=None) byte-identical regression")
    print("=" * 64)

    cfg = default_config()
    log, lifecycle, _ = simulate(cfg)          # legacy path, seed 42

    ok = True
    for name, df in [("event_log.csv", log), ("lot_lifecycle.csv", lifecycle)]:
        golden_path = SYN / name
        if not golden_path.exists():
            print(f"  [SKIP] {name}: no committed golden at {golden_path}")
            continue
        fresh = _csv_bytes(df)
        golden = golden_path.read_bytes()
        # Normalize line endings so the check is about content, not CRLF/LF.
        same = fresh.replace(b"\r\n", b"\n") == golden.replace(b"\r\n", b"\n")
        print(f"  {name:<20} bytes fresh={len(fresh):>7}  "
              f"golden={len(golden):>7}  identical={same}")
        ok = ok and same

    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# GATE 2 — CRN determinism sanity
# --------------------------------------------------------------------------- #
def gate2_crn_sanity() -> bool:
    print("=" * 64)
    print("GATE 2 — CRN determinism sanity (Δ must be exactly 0)")
    print("=" * 64)

    base = default_config()
    t0, t1 = base.warmup_hours, base.horizon_hours

    # One shared table of random draws for this replication.
    draws = draw_randoms(base, seed=base.seed)

    # Baseline run twice on the SAME table.
    log_b1, life_b1, _ = simulate(base, draws)
    log_b2, life_b2, _ = simulate(base, draws)

    th_b1, ct_b1 = run_kpis(life_b1, t0, t1)
    th_b2, ct_b2 = run_kpis(life_b2, t0, t1)

    d_th = th_b2 - th_b1
    d_ct = ct_b2 - ct_b1
    logs_identical = log_b1.equals(log_b2)

    print("  Baseline vs baseline (identical table):")
    print(f"    logs identical         : {logs_identical}")
    print(f"    throughput_1           : {th_b1:.10f}")
    print(f"    throughput_2           : {th_b2:.10f}")
    print(f"    Δthroughput            : {d_th:.3e}   (must be 0.0)")
    print(f"    cycle_time_1 (h)       : {ct_b1:.10f}")
    print(f"    cycle_time_2 (h)       : {ct_b2:.10f}")
    print(f"    Δcycle_time            : {d_ct:.3e}   (must be 0.0)")

    zero_delta = (d_th == 0.0) and (d_ct == 0.0) and logs_identical

    # Positive control: +1 tool at S4 on the SAME table must change the KPIs.
    treat = copy.deepcopy(base)
    treat.stations["S4"].n_tools += 1          # only capacity changes; draws reused
    log_t, life_t, _ = simulate(treat, draws)
    th_t, ct_t = run_kpis(life_t, t0, t1)
    d_th_treat = th_t - th_b1
    d_ct_treat = ct_t - ct_b1
    print("  Positive control — S4 +1 tool on the SAME table:")
    print(f"    Δthroughput            : {d_th_treat:+.5f}   (expect > 0)")
    print(f"    Δcycle_time (h)        : {d_ct_treat:+.5f}   (expect < 0)")
    control_moves = (d_th_treat != 0.0) or (d_ct_treat != 0.0)

    # Property: a CRN run still recovers S4 as the empirical bottleneck.
    bn = empirical_bottleneck(log_b1, base, t0, t1)
    print(f"  Empirical bottleneck on CRN run : {bn}   (expect S4)")
    recovers_s4 = (bn == "S4")

    ok = zero_delta and control_moves and recovers_s4
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    g1 = gate1_legacy_byte_identical()
    g2 = gate2_crn_sanity()
    all_ok = g1 and g2
    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if all_ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
